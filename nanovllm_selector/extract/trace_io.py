"""从原始算子轨迹确定跨节点的输出槽位，以及实际返回的缓存。"""

from collections import Counter, defaultdict

from .decompose import TRANSPARENT
from .schema import Evidence, TensorRef
from .edges import tensor_view
from .runtime_state import StateTensors


def cache_tensors(cache, buffers=()):
    return StateTensors(cache, buffers)


def observe_ports(builder, nodes, returned, cache, external, history=None):
    """只以 dispatch 的实参、返回顺序和实际缓存为证据，不依赖图上已有的边。"""
    owners = {seq: builder.id_of_cid[builder._uf.find(cluster.cid)]
              for cluster in builder._clusters.values() for seq in cluster.op_seqs}
    origins = {id(tensor): (builder.node_prefix + "input." + name, builder.tid(tensor))
               for name, tensor in external.items()}
    slots = {origin: 0 for origin in origins.values()}
    produced = defaultdict(list)
    required = set()
    consumed = defaultdict(dict)
    history = history or {}
    builder.state_reads = defaultdict(dict)
    touched = set()
    builder.port_witnesses = defaultdict(dict)
    builder.computed_origins = set()
    for prior in history.values():
        origins[id(prior.tensor)] = (prior.producer, prior.slot)
    prior_sources = {(s.producer, s.slot): s for s in history.values()}
    for dispatch in builder.rec.dispatches:
        owner = owners.get(dispatch.seq)
        if dispatch.op in TRANSPARENT:
            if dispatch.inputs:
                origin = origins.get(id(dispatch.inputs[0]))
                for tensor in dispatch.outputs:
                    origins[id(tensor)] = origin
            continue
        if owner is None:
            continue
        for tensor in dispatch.inputs:
            binding = dispatch.input_slots.get(id(tensor))
            if binding:
                builder.port_witnesses[owner].setdefault(tensor_view(tensor), binding)
            origin = origins.get(id(tensor))
            if origin is None:
                prior = history.get(tensor_view(tensor))
                if prior:
                    origin = (prior.producer, prior.slot)
            touched.add(tensor_view(tensor))
            if origin in prior_sources:
                builder.state_reads[owner][builder.tid(tensor)] = prior_sources[origin]
                continue
            if origin and origin[0] != owner:
                required.add(origin)
                consumed[owner][builder.tid(tensor)] = origin
        for tensor in dispatch.outputs:
            origin = (owner, builder.tid(tensor))
            origins[id(tensor)] = origin
            produced[owner].append(origin)
            if dispatch.inputs:
                builder.computed_origins.add(origin)

    for tensor in returned:
        origin = origins.get(id(tensor))
        if not origin:
            raise ValueError("模型返回了来源未知的张量")
        required.add(origin)
    writes = {}
    roles = {}
    shared = Counter(origins.get(id(t)) for t in cache.values())
    for key, tensor in cache.items():
        origin = origins.get(id(tensor))
        if origin in prior_sources or (tensor_view(tensor) not in touched and origin is None):
            continue
        if origin and shared[origin] > 1:
            # 同一个底层运算可返回多个不同的缓存切片，每个切片对应独立输出槽。
            view_origin = (origin[0], builder.tid(tensor))
            if view_origin != origin:
                produced[origin[0]].append(view_origin)
                if origin in builder.computed_origins:
                    builder.computed_origins.add(view_origin)
                origin = view_origin
        if not origin or origin in roles:
            raise ValueError(f"无法唯一定位第 {key[0]} 层 {key[1]} 缓存的写入")
        required.add(origin)
        roles[origin] = key
        writes[key] = origin

    for node in nodes:
        observed = list(dict.fromkeys(origin for origin in produced[node.id] if origin in required))
        node.outputs = []
        for index, origin in enumerate(observed):
            role = roles[origin][1] if origin in roles else "out"
            node.outputs.append(TensorRef(origin[1], role=role, index=index))
            slots[origin] = index
        node.attrs["observed_outputs"] = [origin[1] for origin in observed]
        node.attrs["observed_output_roles"] = [ref.role for ref in node.outputs]
        node.evidence.append(Evidence("observed", "实测对外输出（按产生顺序）：" +
                                      ", ".join(f"{r.tensor} ({r.role})" for r in node.outputs)))
    return origins, consumed, slots, writes
