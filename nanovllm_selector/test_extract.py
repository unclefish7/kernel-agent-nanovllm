"""计算图抽取的单元测试（CPU / fp32，不需要 GPU 和权重）。

从仓库根目录运行：

    python -m unittest -v nanovllm_selector.test_extract

默认使用 tmp/models/Qwen3-0.6B，可用环境变量 EXTRACT_MODEL 指向别的目录；
模型目录不存在时整组跳过。
"""

import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from nanovllm_selector.extract import extract_graph
from nanovllm_selector.extract.render import (
    render_dot,
    render_html,
    render_markdown,
    render_mermaid,
    render_svg,
    render_text,
    write_all,
)
from nanovllm_selector.extract.schema import Edge, Graph, Node, TensorMeta, SCHEMA_VERSION

MODEL = os.environ.get("EXTRACT_MODEL", "tmp/models/Qwen3-0.6B")


class TestSchema(unittest.TestCase):
    def test_round_trip(self):
        g = Graph(
            model={"model_type": "toy", "num_layers": 2},
            env={"dtype": "float32", "device": "cpu"},
            nodes=[
                Node(id="a", op="input", origin="input",
                     outputs=[{"tensor": "t0", "role": "out"}],
                     evidence=[{"kind": "observed", "detail": "测试构造", "ref": None}]),
                Node(id="b", op="matmul",
                     inputs=[{"tensor": "t0", "role": "in", "index": 0,
                              "path": "args[0]", "arg_name": "x"}],
                     outputs=[{"tensor": "t1", "role": "out"}],
                     evidence=[{"kind": "observed", "detail": "测试构造", "ref": None}]),
            ],
            edges=[Edge(from_node="a", to_node="b", tensor="t0")],
            tensors={"t0": TensorMeta(id="t0", shape=[8, 4], dtype="float32"),
                     "t1": TensorMeta(id="t1", shape=[8, 4], dtype="float32")},
        )
        again = Graph.from_json(g.to_json())
        self.assertEqual(again.schema_version, SCHEMA_VERSION)
        self.assertEqual([n.id for n in again.nodes], ["a", "b"])
        self.assertEqual(again.validate(), [])
        self.assertEqual(json.loads(g.to_json())["edges"][0]["from_node"], "a")
        # 输入位置信息必须原样穿过 JSON：path / arg_name 丢了，生成代码时
        # 就分不出 `SiluAndMul` 的前后两半、`store_kvcache` 的 key/value。
        ref = again.nodes[1].inputs[0]
        self.assertEqual((ref.index, ref.path, ref.arg_name), (0, "args[0]", "x"))

    def test_validate_catches_dangling_edge(self):
        g = Graph(nodes=[Node(id="a", op="input")],
                  edges=[Edge(from_node="a", to_node="ghost", tensor="t0")])
        self.assertTrue(g.validate())


@unittest.skipUnless(os.path.isdir(MODEL), f"模型目录不存在：{MODEL}")
class TestExtractQwen3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = extract_graph(MODEL, dtype="float32", device="cpu", seq_len=8)

    # ---- 结构 -----------------------------------------------------
    def test_reconcile_passes(self):
        self.assertTrue(self.graph.reconcile["ok"], self.graph.reconcile)
        self.assertEqual(self.graph.reconcile["validate_problems"], [])
        failed = [c for c in self.graph.reconcile["checks"] if not c["ok"]]
        self.assertEqual(failed, [])

    def test_op_counts(self):
        counts = self.graph.op_counts(expand=True)
        self.assertEqual(counts.get("matmul"), 196)
        self.assertEqual(counts.get("lm_head"), 1)
        self.assertEqual(counts.get("rmsnorm"), 113)
        self.assertEqual(counts.get("silu"), 28)
        self.assertEqual(counts.get("add"), 56)
        self.assertEqual(counts.get("embedding"), 1)
        self.assertNotIn("unknown", counts)

    def test_input_slots(self):
        """每个输入都要说清"摆在第几个参数位"（导师指出的缺口）。"""
        for n in self.graph.nodes:
            for i, t in enumerate(n.inputs):
                self.assertEqual(t.index, i, f"{n.id} 的输入下标不对")
                self.assertTrue(t.path, f"{n.id} 的输入 {t.tensor} 没有 path")
                self.assertTrue(t.arg_name, f"{n.id} 的输入 {t.tensor} 没有 arg_name")

    def test_replay_passes(self):
        """逐节点重放：用独立实现复算每个节点，必须和原模型一致。"""
        rep = self.graph.reconcile["replay"]
        self.assertTrue(rep["ok"], rep["failures"][:3])
        self.assertEqual(rep["stats"]["failed"], 0)
        self.assertGreaterEqual(rep["stats"]["checked"], 400)

    def test_invariants_pass(self):
        """三条机械不变量：dispatch 全覆盖、张量有主、入参有槽。"""
        rec = self.graph.reconcile
        for key in ("I1", "I2", "I3"):
            hits = [c for c in rec["checks"] if c["name"].startswith(key + " ")]
            self.assertEqual(len(hits), 1, f"{key} 没有报告")
            self.assertTrue(hits[0]["ok"], hits[0])
        self.assertEqual(rec["invariants"], [])

    def test_dispatch_accounting_is_closed(self):
        s = self.graph.reconcile["stats"]
        self.assertEqual(s["dispatch_unassigned"], 0)
        self.assertEqual(
            s["dispatch_total"],
            s["dispatch_in_nodes"] + s["dispatch_transparent_only"] + s["dispatch_both"],
        )

    def test_rope_split_and_config_are_present(self):
        """cos/sin 是两张表、rope 只在 attention 里做、eps 与 config 不能缺。"""
        # 旋转发生在 attention 簇内部，所以 cos 与 sin 都是 attention 的两个输入。
        attn = self.graph.node("L*.self_attn")
        emb = [t for t in attn.inputs if (t.arg_name or "") == "position_embeddings"]
        self.assertEqual(len(emb), 2, "cos 和 sin 必须各算一个输入，不能按生产者去重")
        self.assertEqual(
            [t.path for t in emb],
            ["kwargs['position_embeddings'][0]", "kwargs['position_embeddings'][1]"],
        )
        # 频率表节点只算 cos/sin，不旋转任何张量，所以不能叫 rope。
        table = self.graph.node("rotary_emb")
        self.assertEqual(table.op, "rope_table")
        self.assertIn("cos", table.semantics["formula"])
        self.assertNotIn("rotate_half", table.semantics["formula"])
        self.assertNotIn("rope", self.graph.op_counts())
        # eps 在每个 norm 节点上，config 在 model 段里。
        rms = self.graph.node("L*.input_layernorm")
        self.assertIn("variance_epsilon", [s.name for s in rms.state])
        cfg = self.graph.model["config"]
        self.assertEqual(cfg["rms_norm_eps"], 1e-06)
        self.assertEqual(cfg["rope_theta"], 1000000)

    def test_mul_input_order_is_recoverable(self):
        """SwiGLU 的两路必须能区分先后，否则换成 SiluAndMul 的 chunk 就错。"""
        node = self.graph.node("L*.mlp.mul")
        self.assertEqual(node.semantics["bindings"], {"a": "in[0]", "b": "in[1]"})
        self.assertEqual([t.path for t in node.inputs], ["args[0]", "args[1]"])
        self.assertEqual([t.arg_name for t in node.inputs], ["self", "other"])
        # in[0] 必须是 silu(gate) 那一路，in[1] 是 up 那一路
        producers = {
            e.tensor: e.from_node
            for e in self.graph.edges
            if e.to_node == "L*.mlp.mul"
        }
        self.assertEqual(producers[node.inputs[0].tensor], "L*.mlp.act_fn")
        self.assertEqual(producers[node.inputs[1].tensor], "L*.mlp.up_proj")

    def test_every_node_has_evidence(self):
        for n in self.graph.nodes:
            self.assertTrue(n.evidence, f"{n.id} 没有证据")

    def test_edges_resolve(self):
        ids = {n.id for n in self.graph.nodes}
        for e in self.graph.edges:
            self.assertIn(e.from_node, ids)
            self.assertIn(e.to_node, ids)

    def test_template_folding(self):
        layer_nodes = [n for n in self.graph.nodes if n.template_id and n.template_id.startswith("L*")]
        self.assertTrue(layer_nodes)
        for n in layer_nodes:
            self.assertEqual(len(n.instances), 28, n.id)
            self.assertTrue(all(i < 28 for i in n.instances))

    def test_layer_edges_are_marked(self):
        # 折叠图上不允许出现假环：层间边单独标 kind=layer
        layer_edges = [e for e in self.graph.edges if e.kind == "layer"]
        self.assertEqual(len(layer_edges), 2)
        for e in layer_edges:
            self.assertNotEqual(e.from_node, e.to_node)

    def test_weights_are_state_not_inputs(self):
        # 权重只能出现在 state 里；图上不该有几百个"权重输入节点"
        inputs = [n for n in self.graph.nodes if n.origin == "input"]
        self.assertLessEqual(len(inputs), 3)
        ids = {n.id for n in inputs}
        self.assertIn("input.input_ids", ids)

    def test_no_cycle(self):
        # 拓扑排序必须能跑完，否则渲染会退化成一条线
        indeg = {n.id: 0 for n in self.graph.nodes}
        adj = {n.id: [] for n in self.graph.nodes}
        for e in self.graph.edges:
            if e.kind == "layer" or e.from_node == e.to_node:
                continue
            adj[e.from_node].append(e.to_node)
            indeg[e.to_node] += 1
        queue = [k for k, v in indeg.items() if v == 0]
        seen = 0
        while queue:
            cur = queue.pop()
            seen += 1
            for nxt in adj[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        self.assertEqual(seen, len(indeg), "图上还有环")


@unittest.skipUnless(os.path.isdir(MODEL), f"模型目录不存在：{MODEL}")
class TestDecode(unittest.TestCase):
    """decode 走的是另一条分支，必须单独抽、单独验。"""

    @classmethod
    def setUpClass(cls):
        cls.graph = extract_graph(
            MODEL, dtype="float32", device="cpu", seq_len=8, phases=("prefill", "decode")
        )

    def test_decode_nodes_exist(self):
        self.assertEqual(self.graph.phases, ["prefill", "decode"])
        dec = [n for n in self.graph.nodes if n.phase == "decode"]
        self.assertGreater(len(dec), 100)
        self.assertTrue(any(n.id.startswith("D.") for n in dec))

    def test_decode_replays(self):
        rep = self.graph.reconcile["replay"]
        self.assertTrue(rep["ok"], rep["failures"][:3])
        self.assertEqual(rep["stats"]["failed"], 0)
        self.assertGreater(rep["stats"]["checked"], 800)

    def test_reconcile_per_phase(self):
        """模块树只对得上一次前向：两个阶段混在一起数会翻倍。"""
        self.assertTrue(self.graph.reconcile["ok"])
        names = {c["name"] for c in self.graph.reconcile["checks"]}
        self.assertIn("matmul[prefill]", names)
        self.assertIn("matmul[decode]", names)
        for c in self.graph.reconcile["checks"]:
            if c["name"].startswith(("matmul[", "rmsnorm[", "silu[")):
                self.assertEqual(c["expected"], c["actual"], c["name"])

    def test_kv_cache_is_cross_step_state(self):
        """KV cache 不是普通输入：它是节点自己的、跨 step 活着的状态。"""
        n = self.graph.node("D.L0.self_attn")
        kinds = [s.kind for s in n.state]
        self.assertIn("runtime_tensor", kinds)
        self.assertEqual(sum(1 for t in n.inputs if t.role == "state"), 2)


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = extract_graph(MODEL, dtype="float32", device="cpu", seq_len=8)

    def test_text(self):
        text = render_text(self.graph)
        self.assertIn("计算图", text)
        self.assertIn("L*.mlp.act_fn", text)
        self.assertIn("层间连接", text)

    def test_markdown(self):
        md = render_markdown(self.graph)
        self.assertIn("## 逐节点", md)
        self.assertIn("```mermaid", md)
        self.assertIn("| 对账 | 通过 |", md)

    def test_mermaid(self):
        mmd = render_mermaid(self.graph)
        self.assertTrue(mmd.strip().endswith("classDef unknown fill:#fecaca,stroke:#7f1d1d,color:#111"))
        self.assertIn("flowchart TD", mmd)
        self.assertIn("classDef compute", mmd)
        self.assertEqual(mmd.count("-->") , len([e for e in self.graph.edges
                                                 if e.kind != "layer" and e.from_node != e.to_node]))

    def test_dot(self):
        dot = render_dot(self.graph)
        self.assertTrue(dot.startswith("digraph G {"))
        self.assertIn("L*.mlp.act_fn", dot)

    def test_svg_is_valid_xml(self):
        svg = render_svg(self.graph)
        root = ET.fromstring(svg)
        self.assertTrue(root.tag.endswith("svg"))
        self.assertNotIn("NaN", svg)

    def test_html_embeds_svg(self):
        html = render_html(self.graph)
        self.assertIn("<svg", html)
        self.assertIn("逐节点明细", html)

    def test_write_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_all(self.graph, tmp, stem="g")
            self.assertEqual(len(paths), 6)
            for kind, path in paths.items():
                self.assertTrue(os.path.getsize(path) > 0, kind)
            with open(paths["json"], encoding="utf-8") as f:
                self.assertEqual(json.load(f)["schema_version"], SCHEMA_VERSION)



LLAMA_FIXTURE = "nanovllm_selector/fixtures/llama-3.2-1b"


class TestExtractLlama(unittest.TestCase):
    """第二个架构：没有 q/k norm，RoPE 用 llama3 型缩放（频率表被改过）。

    只放 config.json，不下载权重；`from_config` 随机初始化。
    """

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(os.path.join(LLAMA_FIXTURE, "config.json")):
            raise unittest.SkipTest("缺少 llama 的 config.json")
        cls.graph = extract_graph(LLAMA_FIXTURE, dtype="float32", device="cpu", seq_len=8)

    def test_all_checks_pass(self):
        self.assertTrue(self.graph.reconcile["ok"], self.graph.reconcile)
        failed = [c for c in self.graph.reconcile["checks"] if not c["ok"]]
        self.assertEqual(failed, [])

    def test_shape_of_graph(self):
        ids = {n.id for n in self.graph.nodes}
        self.assertEqual(len(self.graph.nodes), 20)
        # 16 层 × 14 个节点，加顶层 5 个（input_ids 不算实例节点）
        self.assertEqual(self.graph.reconcile["stats"]["instance_nodes"], 229)
        self.assertNotIn("L*.self_attn.q_norm", ids)
        self.assertNotIn("L*.self_attn.k_norm", ids)
        counts = self.graph.op_counts(expand=True)
        self.assertEqual(counts.get("rmsnorm"), 33)
        self.assertEqual(counts.get("matmul"), 112)
        self.assertEqual(counts.get("lm_head"), 1)
        self.assertEqual(counts.get("rope_table"), 1)

    def test_rope_uses_recorded_frequency_table(self):
        """llama3 缩放改了 inv_freq；只按 rope_theta 重算会对不上。"""
        n = self.graph.node("rotary_emb")
        kinds = {(s.name, s.kind) for s in n.state}
        self.assertIn(("inv_freq", "buffer"), kinds)
        self.assertEqual(len(n.outputs), 2)  # cos / sin，中间量不算输出

    def test_eps_comes_from_module(self):
        n = self.graph.node("L*.input_layernorm")
        scalars = {s.name for s in n.state if s.kind == "scalar"}
        self.assertIn("variance_epsilon", scalars)


if __name__ == "__main__":
    unittest.main()
