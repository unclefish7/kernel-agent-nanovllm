"""生成入口的证据门控与混合层上下文。只在内存里修改样本。"""
import unittest
import hashlib
import re
from importlib.util import find_spec
from pathlib import Path

from nanovllm_selector.port.generate import generate_model
from nanovllm_selector.port.prompt import _render_graph, _render_spec, build_sections, load_graph


def program_node(layer, shift):
    source, rolled, result = (f"t{layer}{i}" for i in range(3))
    def tensor(name):
        return {"tensor": name}

    return {
        "id": f"L{layer}.mixer", "phase": "prefill", "op": "unknown",
        "impl": {"module_path": f"model.layers.{layer}.mixer"},
        "inputs": [{"tensor": source, "index": 0}],
        "outputs": [{"tensor": result, "index": 0}],
        "attrs": {"program": [
            {"seq": layer * 10, "op": "aten.roll.default",
             "args": [tensor(source), [shift], [-1]], "kwargs": {},
             "result": tensor(rolled), "writes": []},
            {"seq": layer * 10 + 1, "op": "aten.copy_.default",
             "args": [tensor(rolled), tensor(source)], "kwargs": {"non_blocking": False},
             "result": tensor(result), "writes": [rolled]},
        ], "program_replay": {"ok": True, "checked": 2, "total": 2}},
    }


def decode_projection(layer):
    x, weight, y = (f"{prefix}{layer}" for prefix in ("x", "w", "y"))

    def branch(batch, op):
        # 不同样本有意复用张量号，但形状和权重视图不同。
        dims = [1, 1, 4] if batch == 1 else [2, 4]
        shapes = {x: dims, weight: [1, 4, 3] if batch == 1 else [4, 3], y: [*dims[:-1], 3]}
        inputs = [{"index": 0, "tensor": x}]
        outputs = [{"index": 0, "tensor": y}]
        step = {"seq": layer + batch, "op": op, "args": [{"tensor": x}, {"tensor": weight}],
                "kwargs": {}, "result": {"tensor": y}, "writes": []}
        name = f"model.layers.{layer}.out_proj.weight"
        return {"cases": [[batch, 5], [batch, 17 if batch == 1 else 11]],
                "representative_case": [batch, 5], "inputs": inputs, "outputs": outputs,
                "tensors": {t: {"shape": shape, "dtype": "float32"} for t, shape in shapes.items()},
                "program": [step], "program_inputs": {x: {"kind": "input", "tensor": x},
                                                     weight: {"kind": "state", "name": name}},
                "program_state": {weight: name}, "program_replay": {"ok": True, "checked": 1, "total": 1}}

    single, pair = branch(1, "aten.bmm.default"), branch(2, "aten.mm.default")
    attrs = {key: single[key] for key in ("program", "program_inputs", "program_state", "program_replay")}
    return {"id": f"D.L{layer}.out_proj", "phase": "decode", "op": "matmul",
            "inputs": single["inputs"], "outputs": single["outputs"],
            "attrs": {**attrs, "program_variants": [single, pair]}}


class PromptContractsTest(unittest.TestCase):
    def test_no_call_when_graph_lacks_validation(self):
        graph = {"schema_version": "graph.v1", "nodes": [program_node(0, -1)]}
        with self.assertRaisesRegex(ValueError, "重新抽取"):
            generate_model(graph, {}, "index\n", "template\n", lambda _: self.fail("调用了模型"))

    def test_failed_graph_reports_failed_check_and_next_step(self):
        graph = {"reconcile": {"ok": False, "checks": [
            {"name": "I2 missing source at L7", "ok": False, "expected": 0, "actual": 1}
        ]}}
        with self.assertRaisesRegex(ValueError, r"I2 missing source at L7.*重新抽取"):
            generate_model(graph, {}, "index\n", "template\n", lambda _: self.fail("调用了模型"))

    def test_stale_legacy_replay_coverage_blocks_generation(self):
        graph = load_graph(Path(__file__).parents[1] / "outputs/graph_qwen3/graph.json")
        graph["reconcile"]["replay"]["checked"] -= 1
        with self.assertRaisesRegex(ValueError, r"重放.*452.*453.*重新抽取"):
            build_sections(graph, {}, "index\n", "template\n")

    def test_variants_require_probe_evidence_even_on_a_legacy_graph(self):
        graph = load_graph(Path(__file__).parents[1] / "outputs/graph_qwen3/graph.json")
        node = graph["nodes"][0]
        node["attrs"]["program_variants"] = decode_projection(0)["attrs"]["program_variants"]
        with self.assertRaisesRegex(ValueError, "program_variants.*shape_probes"):
            generate_model(graph, {}, "index\n", "template\n", lambda _: self.fail("调用了模型"))

    def test_shared_program_keeps_operand_order_constant_and_writes(self):
        nodes = [program_node(i, shift) for i, shift in enumerate([-1, -1, -2])]
        graph = {"nodes": nodes, "env": {"state_slots": []},
                 "phases": ["prefill"], "model": {"num_layers": 3}}
        rendered = _render_graph(graph)
        self.assertEqual(rendered.count("aten.roll.default"), 2)
        self.assertEqual(rendered.count("aten.copy_.default"), 2)
        self.assertLess(rendered.index("aten.roll.default"), rendered.index("aten.copy_.default"))
        for value in ("[-2]", "[-1]", "non_blocking", "写入", "mixer"):
            self.assertIn(value, rendered)

    def test_state_update_and_readers_survive_grouping(self):
        records = []
        for i, retained in enumerate([2, 2, 1]):
            records.append({
                "layer": i, "phase": "decode", "path": f"cache.windows[{i}]",
                "aliases": [f"store.entries[{i}]"], "role": "window", "shape": [1, 12, 3],
                "dtype": "float32", "producer": f"D.L{i}.mixer", "source_slot": 1,
                "tensor": f"dt{i}", "readers": [{"node": f"D2.L{i}.mixer", "target_slot": 2}],
                "update": {"kind": "slide", "retained_columns": retained},
            })
        rendered = _render_graph({"env": {"state_slots": records}, "nodes": []})
        self.assertEqual(rendered.count("retained_columns"), 2)
        for value in ("decode", "windows", "store.entries", "D2.", "slide", "dt2", "target_slot"):
            self.assertIn(value, rendered)

    def test_engine_gaps_visible_even_with_toolbox_omitted(self):
        spec = {"capabilities": [
            {"id": "fixed_rows", "supported": True, "detail": "按请求槽位读写"},
            {"id": "prefix_snapshots", "supported": False, "detail": "尚无检查点"},
        ]}
        rendered = _render_spec(spec, include_toolbox=False)
        self.assertRegex(rendered, r"支持.*fixed_rows.*按请求槽位读写")
        self.assertRegex(rendered, r"缺口.*prefix_snapshots.*尚无检查点")

    def test_rank_changes_keep_the_case_and_corresponding_shape(self):
        node = program_node(0, -1)
        node["attrs"]["shape_profile"] = {"outputs": [{"alternatives": [
            {"cases": [[1, 5]], "shapes": [[5, 12]]},
            {"cases": [[2, 5]], "shapes": [[2, 5, 12]]}
        ]}]}
        rendered = _render_graph({"nodes": [node], "env": {"state_slots": []}})
        self.assertIn('{"cases":[[1,5]],"shapes":[[5,12]]}', rendered)
        self.assertIn('{"cases":[[2,5]],"shapes":[[2,5,12]]}', rendered)

    def test_decode_cases_select_their_own_program_and_weight_view(self):
        nodes = [decode_projection(i) for i in (0, 3)]
        tensors = {t: meta for n in nodes for t, meta in n["attrs"]["program_variants"][0]["tensors"].items()}
        rendered = _render_graph({"nodes": nodes, "env": {"state_slots": []}, "tensors": tensors})
        for cases, representative, op, shape in (
            ("[[1,5],[1,17]]", "[1,5]", "aten.bmm.default", "[1,4,3]"),
            ("[[2,5],[2,11]]", "[2,5]", "aten.mm.default", "[4,3]"),
        ):
            with self.subTest(cases=cases):
                rows = [row for row in rendered.splitlines() if cases in row]
                self.assertEqual(len(rows), 1)
                self.assertIn(f"代表={representative}", rows[0])
                program = re.search(r"\bP\d+\b", rows[0]).group()
                recipe = re.search(rf"^{program} (.*?)(?=^P\d+ |^## |\Z)", rendered, re.M | re.S)[1]
                self.assertIn(op, recipe)
                layout = re.search(r"out_proj\.weight/(T\d+)", recipe)[1]
                self.assertRegex(rendered, rf'(?m)^{layout} .*"shape":{re.escape(shape)}')
                self.assertEqual(rendered.count(op), 1)

    def test_variant_uses_its_own_producer_even_when_tensor_ids_agree(self):
        node = decode_projection(0)
        pair = node["attrs"]["program_variants"][1]
        pair["program_inputs"]["x0"] = {
            "kind": "producer", "tensor": "x0", "node": "D.L0.alternate_source",
            "seq": 1, "step": 3, "result_index": 2,
        }
        rendered = _render_graph({"nodes": [node], "env": {"state_slots": []}})
        self.assertIn("L.alternate_source.step3[2]", rendered)

    @unittest.skipUnless(find_spec("torch") and find_spec("transformers"), "需要本地 CPU 抽图环境")
    def test_generation_gate_checks_variant_samples_and_replay(self):
        from nanovllm_selector.extract.shapes import profile_shapes
        from nanovllm_selector.test_mixed_state import tiny_mixed_graph

        graph = tiny_mixed_graph()
        profile_shapes(graph, 1, 5, tiny_mixed_graph)
        name = "D.L0.conv.out_proj"
        self.assertIn("program_variants", graph.node(name).attrs)
        sections = build_sections(graph, {}, "index\n", "template\n", exact=False)
        self.assertIn("代表=[2,5]", sections["graph"])
        for change in ("cases", "replay"):
            with self.subTest(change=change):
                data = load_graph(graph.to_json())
                node = next(n for n in data["nodes"] if n["id"] == name)
                pair = node["attrs"]["program_variants"][1]
                if change == "cases":
                    pair["cases"].pop()
                else:
                    pair["program_replay"]["checked"] -= 1
                with self.assertRaisesRegex(ValueError, re.escape(name)):
                    generate_model(data, {}, "index\n", "template\n", lambda _: self.fail("调用了模型"))

    def test_interleaved_nodes_keep_the_global_write_order(self):
        left, right = program_node(0, -1), program_node(0, -7)
        right["id"] = "L0.reader"
        left["attrs"]["program"][1]["seq"] = 3
        for i, step in enumerate(right["attrs"]["program"], 1):
            step["seq"] = i
        rendered = _render_graph({"nodes": [left, right], "env": {"state_slots": []}})
        self.assertIn("N1[0:1] N2 N1[1:2]", rendered)

    def test_external_tensor_literal_participates_in_program_identity(self):
        nodes = [program_node(i, -1) for i in range(3)]
        for node, value in zip(nodes, (17, 17, 21)):
            tid = node["id"] + ".constant"
            node["attrs"]["program"][0]["args"].append({"tensor": tid})
            node["attrs"]["program_inputs"] = {tid: {"kind": "literal", "value": [value]}}
        rendered = _render_graph({"nodes": nodes, "env": {"state_slots": []}})
        self.assertEqual(rendered.count("aten.roll.default"), 2)
        self.assertIn("[17]", rendered)
        self.assertIn("[21]", rendered)

    def test_rounding_evidence_comes_from_matching_local_source(self):
        package = find_spec("transformers")
        if not package:
            self.skipTest("需要本地 Transformers 源码")
        path = Path(package.origin).parent / "models/lfm2/modeling_lfm2.py"
        source = {"qualname": "transformers.models.lfm2.modeling_lfm2.Lfm2RMSNorm",
                  "class": "Lfm2RMSNorm", "file": str(path),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        node = program_node(0, -1)
        node["op"] = "rmsnorm"
        node["impl"]["source"] = source
        graph = {"nodes": [node], "env": {"state_slots": []}}
        expected = "self.weight * hidden_states.to(input_dtype)"
        self.assertIn(expected, _render_graph(graph))
        source["sha256"] = "0" * 64
        self.assertNotIn(expected, _render_graph(graph))

    def test_incomplete_verified_artifact_is_rejected_before_call(self):
        path = Path(__file__).parents[1] / "outputs/lfm25_1_2b/graph_final.json"
        if not path.is_file():
            self.skipTest("需要本地已通过验证的混合图")
        for change in ("program_inputs", "producer", "readers", "update"):
            with self.subTest(change=change):
                graph = load_graph(path)
                if change == "program_inputs":
                    node = next(n for n in graph["nodes"] if n["attrs"].get(change))
                    node["attrs"][change].clear()
                elif change == "producer":
                    binding = next(b for n in graph["nodes"] for b in n["attrs"].get(
                        "program_inputs", {}).values() if b["kind"] == "producer")
                    binding["node"] = "missing_producer"
                else:
                    slot = next(s for s in graph["env"]["state_slots"] if s["readers"])
                    if change == "update":
                        del slot[change]
                    else:
                        slot[change].clear()
                with self.assertRaisesRegex(ValueError, r"重新抽取"):
                    generate_model(graph, {}, "index\n", "template\n", lambda _: self.fail("调用了模型"))
