"""把图渲染成人能看懂的形式。

同一份图输出五种视图，各自解决一个问题：

| 形式 | 解决什么问题 |
| --- | --- |
| 终端文本 | 在 SSH 里快速扫一眼：有哪些节点、什么顺序、多少实例 |
| Markdown 表 | 逐个节点核对形状、状态、证据数 |
| Mermaid | 贴进 GitHub / mermaid.live 就能看流程图 |
| SVG / HTML | 双击打开，离线可看，带颜色和 tooltip |
| Graphviz DOT | 需要精细排版或打印时用 |

JSON 给机器，这些给人。
"""

from __future__ import annotations

import html as _html
import os
from collections import defaultdict

from .schema import Graph, Node

FAMILY_COLORS = {
    "memory": ("#dbeafe", "#1e40af"),
    "compute": ("#dcfce7", "#166534"),
    "normalization": ("#fef3c7", "#92400e"),
    "activation": ("#fce7f3", "#9d174d"),
    "elementwise": ("#ede9fe", "#5b21b6"),
    "attention": ("#fee2e2", "#991b1b"),
    "shape": ("#e5e7eb", "#374151"),
    "unknown": ("#fecaca", "#7f1d1d"),
}

FAMILY_LABELS = {
    "memory": "访存",
    "compute": "矩阵计算",
    "normalization": "归一化",
    "activation": "激活",
    "elementwise": "逐元素",
    "attention": "注意力",
    "shape": "形状/索引",
    "unknown": "未识别",
}

OP_CN = {
    "input": "输入",
    "embedding": "词嵌入",
    "matmul": "矩阵乘",
    "lm_head": "输出投影",
    "rmsnorm": "RMSNorm",
    "silu": "SiLU",
    "gelu": "GELU",
    "mul": "逐元素乘",
    "add": "残差加",
    "attention": "注意力",
    "rope": "旋转位置编码",
    "rope_table": "RoPE 频率表",
    "arange": "生成位置索引",
    "unknown": "未知",
}


# ---------------------------------------------------------------- 公共
def node_shape(node: Node, meta: dict) -> tuple[str, str]:
    def fmt(refs, count: bool = False):
        if not refs:
            return "—"
        m = meta.get(refs[0].tensor)
        text = "?" if not m else "×".join(str(x) for x in m.shape)
        if count and len(refs) > 1:
            text += f" 等{len(refs)}个"
        return text

    # 输入不标数量：每个输入来自谁下面已经逐条列了；输出标数量，
    # 因为一个节点吐出多个张量（如 rope 的 cos/sin）是重要信息。
    return fmt(node.inputs), fmt(node.outputs, count=True)


def _state_summary(node: Node) -> str:
    parts = []
    for s in node.state:
        if s.kind == "parameter":
            parts.append("w" + ("×".join(str(x) for x in (s.shape or [])) if s.shape else ""))
        elif s.kind == "scalar":
            parts.append(f"{s.name}={s.dtype}")
    return ",".join(parts) if parts else "—"


def _ranks(nodes: list[Node], edges) -> tuple[dict[str, int], dict[str, int]]:
    ids = [n.id for n in nodes]
    preds: dict[str, set[str]] = {i: set() for i in ids}
    for e in edges:
        if e.kind == "layer":
            continue  # 层间边在模板图里是自环，不参与分层
        if e.from_node in preds and e.to_node in preds and e.from_node != e.to_node:
            preds[e.to_node].add(e.from_node)
    rank: dict[str, int] = {}
    remaining = set(ids)
    while remaining:
        progressed = False
        for nid in sorted(remaining):
            ps = [rank[p] for p in preds[nid] if p in rank]
            if all(p in rank for p in preds[nid]):
                rank[nid] = 0 if not ps else max(ps) + 1
                remaining.discard(nid)
                progressed = True
        if not progressed:  # 有环：剩下的按 0 处理
            for nid in remaining:
                rank[nid] = 0
            break
    order: dict[str, int] = {}
    buckets: dict[int, list[str]] = defaultdict(list)
    for nid in ids:
        buckets[rank[nid]].append(nid)
    for r, group in buckets.items():
        pos = sorted(group)
        for i, nid in enumerate(pos):
            order[nid] = i
    return rank, order


# ---------------------------------------------------------------- 文本
def render_text(graph: Graph, width: int = 110) -> str:
    meta = graph.tensors
    nodes = graph.nodes
    edges = graph.edges
    rank, _ = _ranks(nodes, edges)
    by_id = {n.id: n for n in nodes}
    # 按"目标节点 + 张量"索引生产者：这样一个节点的每条输入都能单独标出
    # 它摆在参数的第几位、叫什么名字（这是生成代码时必须的信息）。
    producers: dict[tuple[str, str], str] = {}
    state_from: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e.from_node == e.to_node or e.kind == "layer":
            continue
        if e.kind == "state":
            state_from[e.to_node].append(e.from_node)
        else:
            producers[(e.to_node, e.tensor)] = e.from_node

    ordered = sorted(nodes, key=lambda n: (rank.get(n.id, 0), n.id))
    lines = []
    m = graph.model
    lines.append("=" * width)
    title = f"{m.get('model_type', '?')} · {m.get('num_layers', '?')} 层 · {graph.env.get('dtype', '?')}/{graph.env.get('device', '?')}"
    lines.append(f"计算图  {title}")
    st = graph.reconcile.get("stats", {})
    folded = any(len(n.instances) > 1 for n in nodes)
    inst = st.get("instance_nodes")
    head = f"模板节点 {len(nodes)}  边 {len(edges)}  图输入 {len([n for n in nodes if n.origin == 'input'])}"
    if folded and inst:
        head += f"  展开后 {inst} 个实例节点"
    lines.append(head)
    lines.append("=" * width)
    for i, n in enumerate(ordered):
        ins, outs = node_shape(n, meta)
        instances = f" ×{len(n.instances)}层" if len(n.instances) > 1 else ""
        head = f"[{i:02d}] {n.id}{instances}"
        pad = max(1, 34 - len(head))
        lines.append(f"{head}{' ' * pad}{n.op:<12} {ins:>16} → {outs:<16} {_state_summary(n)}")
        empty = set(n.attrs.get("empty_inputs") or [])
        for ref in n.inputs:
            src = producers.get((n.id, ref.tensor))
            name = ref.arg_name or "?"
            where = f"  [{ref.path}]" if ref.path else ""
            if src:
                arrow = src
            elif ref.tensor in empty:
                arrow = "空张量（0 元素，prefill 时无内容）"
            else:
                arrow = "（无生产者）"
            lines.append(f"       ↑ in{ref.index} {name} ← {arrow}{where}")
        for src in state_from.get(n.id, []):
            lines.append(f"       ↑ 状态来自 {src}")
    layer_edges = [e for e in edges if e.kind == "layer"]
    if layer_edges:
        lines.append("-" * width)
        lines.append("层间连接（每一层都重复同一条，图上画成自环）：")
        for e in layer_edges:
            lines.append(f"  {e.from_node} → 下一层的 {e.to_node}")
    if graph.conflicts:
        lines.append("-" * width)
        lines.append(f"冲突 {len(graph.conflicts)} 条（详见 JSON）")
    lines.append("-" * width)
    return "\n".join(lines)


# ---------------------------------------------------------------- Markdown
def render_markdown(graph: Graph) -> str:
    meta = graph.tensors
    nodes = graph.nodes
    rank, _ = _ranks(nodes, graph.edges)
    ordered = sorted(nodes, key=lambda n: (rank.get(n.id, 0), n.id))
    st = graph.reconcile.get("stats", {})
    out = []
    m = graph.model
    out.append(f"# 计算图：{m.get('model_type', '?')}（{m.get('num_layers', '?')} 层）\n")
    out.append("> 由 `nanovllm_selector.extract` 从真实执行轨迹重建。"
               "节点标签来自实测 aten 算子，形状来自 hook 实测，证据见 JSON。\n")
    out.append("## 概览\n")
    out.append("| 项 | 值 |")
    out.append("| --- | --- |")
    out.append(f"| 模型 | `{m.get('model_type')}`（`{m.get('architectures')}`） |")
    out.append(f"| 精度 / 设备 | {graph.env.get('dtype')} / {graph.env.get('device')} |")
    out.append(f"| torch / transformers | {graph.env.get('torch')} / {graph.env.get('transformers')} |")
    out.append(f"| 节点 / 边 | {st.get('nodes')} / {st.get('edges')} |")
    out.append(f"| 模块调用 / aten 算子 | {st.get('module_calls')} / {st.get('dispatch_count')} |")
    out.append(f"| 对账 | {'通过' if graph.reconcile.get('ok') else '未通过'} |")
    out.append("")
    out.append("## 逐节点\n")
    out.append("| # | 节点 | 算子 | 输入 | 输出 | 状态 | 证据 | 实例 |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, n in enumerate(ordered):
        ins, outs = node_shape(n, meta)
        inst = f"{len(n.instances)} 层" if n.instances else "—"
        out.append(
            f"| {i} | `{n.id}` | {n.op} | `{ins}` | `{outs}` | {_state_summary(n)} | {len(n.evidence)} | {inst} |"
        )
    out.append("")
    out.append("## 流程\n")
    out.append("```mermaid")
    out.append(render_mermaid(graph, with_header=False))
    out.append("```")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- Mermaid
def _mid(node_id: str) -> str:
    return "n" + "".join(c if c.isalnum() else "_" for c in node_id)


def render_mermaid(graph: Graph, with_header: bool = True) -> str:
    meta = graph.tensors
    lines = []
    if with_header:
        lines.append("%% 由 nanovllm_selector.extract 生成，可贴到 https://mermaid.live 查看")
    lines.append("flowchart TD")
    for n in graph.nodes:
        ins, outs = node_shape(n, meta)
        label = f"{n.id}<br/><b>{n.op}</b>"
        if ins != "—" or outs != "—":
            label += f"<br/><small>{ins} → {outs}</small>"
        # 输入节点画成圆角，其余画成方角。原来的三元表达式把右半边也塞进了
        # 条件里，输入节点会拿到字符串 "]），语法上不报错但画出来是坏的。
        opened, closed = ("([", "])") if n.origin == "input" else ("[", "]")
        lines.append(f'    {_mid(n.id)}{opened}"{label}"{closed}')
    for n in graph.nodes:
        fam = n.family
        lines.append(f"    class {_mid(n.id)} {fam if fam in FAMILY_COLORS else 'unknown'}")
    emitted = set()
    for e in graph.edges:
        if e.kind == "layer":
            continue
        if e.from_node == e.to_node:
            continue
        # 同一个生产者给同一个消费者喂两个不同张量（cos 与 sin）要画两条线，
        # 所以去重键带上张量；折叠后的图里每条边本来就代表一个输入位。
        key = (e.from_node, e.to_node, e.kind, e.tensor)
        if key in emitted:
            continue
        emitted.add(key)
        arrow = "-.->" if e.kind == "state" else "-->"
        lines.append(f"    {_mid(e.from_node)} {arrow} {_mid(e.to_node)}")
    for fam, (bg, fg) in FAMILY_COLORS.items():
        lines.append(f"    classDef {fam} fill:{bg},stroke:{fg},stroke-width:1px,color:#111")
    lines.append("    classDef unknown fill:#fecaca,stroke:#7f1d1d,color:#111")
    return "\n".join(lines)


# ---------------------------------------------------------------- DOT
def render_dot(graph: Graph) -> str:
    meta = graph.tensors
    lines = ["digraph G {", '  rankdir=TD; splines=true; nodesep=0.35; ranksep=0.45;',
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];']
    for n in graph.nodes:
        ins, outs = node_shape(n, meta)
        bg, fg = FAMILY_COLORS.get(n.family, FAMILY_COLORS["unknown"])
        label = f"{n.id}\\n{n.op}\\n{ins} → {outs}"
        lines.append(f'  "{n.id}" [label="{label}", fillcolor="{bg}", color="{fg}"];')
    for e in graph.edges:
        if e.from_node == e.to_node:
            continue
        style = "dashed" if e.kind == "state" else "solid"
        lines.append(f'  "{e.from_node}" -> "{e.to_node}" [style={style}];')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- SVG
def render_svg(graph: Graph, title: str | None = None) -> str:
    meta = graph.tensors
    nodes = graph.nodes
    rank, _ = _ranks(nodes, graph.edges)
    by_id = {n.id: n for n in nodes}
    buckets: dict[int, list[str]] = defaultdict(list)
    for n in nodes:
        buckets[rank[n.id]].append(n.id)
    for r, group in buckets.items():
        group.sort(key=lambda i: (-len([e for e in graph.edges if e.to_node == i]), i))

    pad_x, pad_y = 36, 44
    box_w, box_h = 212, 62
    gap_x, gap_y = 30, 34
    max_cols = max(len(v) for v in buckets.values())
    max_rank = max(buckets)
    width = pad_x * 2 + max_cols * (box_w + gap_x) - gap_x
    height = pad_y * 2 + (max_rank + 1) * (box_h + gap_y) - gap_y + 70

    pos: dict[str, tuple[float, float]] = {}
    for r, group in buckets.items():
        row_w = len(group) * (box_w + gap_x) - gap_x
        x0 = (width - row_w) / 2
        for i, nid in enumerate(group):
            pos[nid] = (x0 + i * (box_w + gap_x), pad_y + 60 + r * (box_h + gap_y))

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                 f'viewBox="0 0 {width} {height}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
    parts.append("<defs>")
    parts.append('<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                 'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/></marker>')
    parts.append('<marker id="arrow-state" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                 'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#c2410c"/></marker>')
    parts.append("</defs>")
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    hdr = title or f"{graph.model.get('model_type')} · {graph.model.get('num_layers')} 层 · {graph.env.get('dtype')}/{graph.env.get('device')}"
    parts.append(f'<text x="{pad_x}" y="28" font-size="15" font-weight="600" fill="#111827">计算图 · {_esc(hdr)}</text>')
    parts.append(f'<text x="{pad_x}" y="46" font-size="11" fill="#6b7280">'
                 f'节点 {len(nodes)} · 边 {len([e for e in graph.edges if e.from_node != e.to_node])} · '
                 f'对账 {"通过" if graph.reconcile.get("ok") else "未通过"} · 实线=数据  虚线=状态/跨层</text>')

    for e in graph.edges:
        if e.kind == "layer":
            continue
        if e.from_node == e.to_node or e.from_node not in pos or e.to_node not in pos:
            continue
        x1, y1 = pos[e.from_node]
        x2, y2 = pos[e.to_node]
        sx, sy = x1 + box_w / 2, y1 + box_h
        tx, ty = x2 + box_w / 2, y2
        if abs(sx - tx) < 1:
            d = f"M {sx} {sy} L {tx} {ty}"
        else:
            my = (sy + ty) / 2
            d = f"M {sx} {sy} C {sx} {my}, {tx} {my}, {tx} {ty}"
        color = "#c2410c" if e.kind == "state" else "#9ca3af"
        dash = ' stroke-dasharray="5 4"' if e.kind == "state" else ""
        marker = "arrow-state" if e.kind == "state" else "arrow"
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.4"{dash} '
                     f'marker-end="url(#{marker})"/>')

    for n in nodes:
        x, y = pos[n.id]
        bg, fg = FAMILY_COLORS.get(n.family, FAMILY_COLORS["unknown"])
        ins, outs = node_shape(n, meta)
        inst = f" ×{len(n.instances)}" if len(n.instances) > 1 else ""
        dash = ' stroke-dasharray="5 4"' if n.origin == "input" else ""
        tip = f"{n.id} | op={n.op} | in={ins} | out={outs} | evidence={len(n.evidence)}"
        parts.append(f'<g><title>{_esc(tip)}</title>')
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="9" ry="9" '
                     f'fill="{bg}" stroke="{fg}" stroke-width="1.2"{dash}/>')
        label = _truncate(n.id + inst, 30)
        parts.append(f'<text x="{x + 10}" y="{y + 21}" font-size="11" font-weight="600" fill="#111827">{_esc(label)}</text>')
        parts.append(f'<text x="{x + 10}" y="{y + 38}" font-size="11" fill="{fg}">{_esc(OP_CN.get(n.op, n.op))} · {_esc(n.op)}</text>')
        if ins != "—" or outs != "—":
            parts.append(f'<text x="{x + 10}" y="{y + 53}" font-size="9.5" fill="#4b5563">{_esc(ins + " → " + outs)}</text>')
        parts.append("</g>")

    # 图例
    ly = height - 42
    lx = pad_x
    for fam in ["memory", "compute", "normalization", "activation", "elementwise", "attention", "shape"]:
        bg, fg = FAMILY_COLORS[fam]
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="11" height="11" rx="2" fill="{bg}" stroke="{fg}"/>')
        parts.append(f'<text x="{lx + 16}" y="{ly}" font-size="10.5" fill="#374151">{FAMILY_LABELS[fam]}</text>')
        lx += 16 + len(FAMILY_LABELS[fam]) * 11 + 18
    parts.append("</svg>")
    return "\n".join(parts)


def _esc(s: str) -> str:
    return _html.escape(str(s), quote=True)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------- HTML
def render_html(graph: Graph) -> str:
    svg = render_svg(graph)
    tmeta = graph.tensors
    md = render_markdown(graph)
    st = graph.reconcile.get("stats", {})
    checks = graph.reconcile.get("checks", [])
    rows = []
    for c in checks:
        mark = "✅" if c["ok"] else "❌"
        rows.append(
            f"<tr><td>{mark}</td><td>{_esc(c['name'])}</td><td>{_esc(str(c['expected']))}</td>"
            f"<td>{_esc(str(c['actual']))}</td><td>{_esc(c.get('note') or '')}</td></tr>"
        )
    ops = "".join(
        f"<li><code>{_esc(k)}</code> × {v}</li>" for k, v in (st.get("op_counts") or {}).items()
    )
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>计算图 · {_esc(graph.model.get('model_type', ''))}</title>
<style>
 body{{margin:0;padding:24px 32px;background:#f9fafb;color:#111827;
      font:14px/1.6 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
 h1{{font-size:19px;margin:0 0 4px}} h2{{font-size:15px;margin:28px 0 10px}}
 .sub{{color:#6b7280;font-size:12px;margin-bottom:16px}}
 .card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;overflow:auto}}
 table{{border-collapse:collapse;font-size:12.5px;width:100%}}
 th,td{{border-bottom:1px solid #e5e7eb;padding:5px 8px;text-align:left;vertical-align:top}}
 th{{background:#f3f4f6;position:sticky;top:0}}
 code{{background:#f3f4f6;padding:1px 4px;border-radius:3px;font-size:12px}}
 ul{{columns:4;margin:0;padding-left:18px;font-size:12.5px}}
</style></head><body>
<h1>计算图 · {_esc(graph.model.get('model_type', ''))}</h1>
<div class="sub">{_esc(graph.model.get('architectures'))} · {_esc(graph.env.get('dtype'))}/{_esc(graph.env.get('device'))}
 · torch {_esc(graph.env.get('torch'))} · transformers {_esc(graph.env.get('transformers'))}</div>
<div class="card">{svg}</div>
<h2>映射覆盖（每个算子出现多少次）</h2>
<div class="card"><ul>{ops}</ul></div>
<h2>对账</h2>
<div class="card"><table><tr><th></th><th>检查项</th><th>期望</th><th>实际</th><th>说明</th></tr>{''.join(rows)}</table></div>
<h2>逐节点明细</h2>
<div class="card"><table>
<tr><th>#</th><th>节点</th><th>算子</th><th>输入</th><th>输出</th><th>状态</th><th>证据</th><th>实例</th></tr>
{''.join(f"<tr><td>{i}</td><td><code>{_esc(n.id)}</code></td><td>{_esc(n.op)}</td><td><code>{_esc(node_shape(n, tmeta)[0])}</code></td><td><code>{_esc(node_shape(n, tmeta)[1])}</code></td><td>{_esc(_state_summary(n))}</td><td>{len(n.evidence)}</td><td>{len(n.instances) or '—'}</td></tr>" for i, n in enumerate(graph.nodes))}
</table></div>
</body></html>
"""


# ---------------------------------------------------------------- 落盘
def write_all(graph: Graph, outdir: str, stem: str = "graph") -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "json": os.path.join(outdir, f"{stem}.json"),
        "md": os.path.join(outdir, f"{stem}.md"),
        "mmd": os.path.join(outdir, f"{stem}.mmd"),
        "dot": os.path.join(outdir, f"{stem}.dot"),
        "svg": os.path.join(outdir, f"{stem}.svg"),
        "html": os.path.join(outdir, f"{stem}.html"),
    }
    with open(paths["json"], "w", encoding="utf-8") as f:
        f.write(graph.to_json())
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write(render_markdown(graph))
    with open(paths["mmd"], "w", encoding="utf-8") as f:
        f.write(render_mermaid(graph))
    with open(paths["dot"], "w", encoding="utf-8") as f:
        f.write(render_dot(graph))
    with open(paths["svg"], "w", encoding="utf-8") as f:
        f.write(render_svg(graph))
    with open(paths["html"], "w", encoding="utf-8") as f:
        f.write(render_html(graph))
    return paths
