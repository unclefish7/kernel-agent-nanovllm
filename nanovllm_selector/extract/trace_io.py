"""从原始算子轨迹确定跨节点的输出槽位，以及实际返回的缓存。"""

from collections import defaultdict

from .decompose import TRANSPARENT
from .schema import Evidence, TensorRef


def cache_tensors(cache):
    if cache is None:
        return {}
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError(f"无法读取 {type(cache).__name__} 的逐层缓存")
    result = {}
    for index, layer in enumerate(layers):
        for role, field in (("key", "keys"), ("value", "values")):
            tensor = getattr(layer, field, None)
            if tensor is None:
                raise ValueError(f"缓存第 {index} 层缺少 {field}")
            result[(index, role)] = tensor
    return result


def observe_ports(builder, nodes, returned, cache, external):
    """只以 dispatch 的实参、返回顺序和实际缓存为证据，不依赖图上已有的边。"""
    owners = {seq: builder.id_of_cid[builder._uf.find(cluster.cid)]
              for cluster in builder._clusters.values() for seq in cluster.op_seqs}
    origins = {id(tensor): (builder.node_prefix + "input." + name, builder.tid(tensor))
               for name, tensor in external.items()}
    slots = {origin: 0 for origin in origins.values()}
    produced = defaultdict(list)
    required = set()
    consumed = defaultdict(dict)
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
            origin = origins.get(id(tensor))
            if origin and origin[0] != owner:
                required.add(origin)
                consumed[owner][builder.tid(tensor)] = origin
        for tensor in dispatch.outputs:
            origin = (owner, builder.tid(tensor))
            origins[id(tensor)] = origin
            produced[owner].append(origin)

    for tensor in returned:
        origin = origins.get(id(tensor))
        if not origin:
            raise ValueError("模型返回了来源未知的张量")
        required.add(origin)
    writes = {}
    roles = {}
    for key, tensor in cache.items():
        origin = origins.get(id(tensor))
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
