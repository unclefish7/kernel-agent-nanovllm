"""下游移植流程的单元测试：prompt 组装、围栏剥离、结构校验、CLI dry-run。

运行（容器里，工作目录 /workspace）：

    python -m nanovllm_selector.port.test_port

不访问网络、不需要 GPU、不需要 torch；`yaml` 是唯一第三方依赖。
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nanovllm_selector.port import cli
from nanovllm_selector.port.generate import (
    ValidationError,
    generate_model,
    strip_code_fence,
    validate_model_code,
)
from nanovllm_selector.port.prompt import (
    SECTION_ORDER,
    build_messages,
    build_prompt,
    build_sections,
    load_graph,
    load_spec,
    prompt_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO_ROOT / "nanovllm_selector" / "outputs" / "graph_qwen3" / "graph.json"

MATERIAL_ROOTS = (REPO_ROOT, REPO_ROOT / "tmp" / "port-deps")

GOOD_CODE = '''import torch
from torch import nn

from nanovllm.utils.context import get_context


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, input_ids, positions):
        ctx = get_context()
        return input_ids, ctx
'''


def _materials() -> tuple[Path, Path, Path]:
    """三样材料可能分别来自仓库根和 tmp/port-deps（容器里 contracts/、kernels/ 未挂载）。"""
    resolved: list[Path] = []
    for relative in (
        Path("contracts") / "nanovllm" / "spec.yaml",
        Path("kernels") / "index.md",
        Path("nano-vllm") / "nanovllm" / "models" / "qwen3.py",
    ):
        found = next(
            (root / relative for root in MATERIAL_ROOTS if (root / relative).is_file()), None
        )
        if found is None:
            raise unittest.SkipTest(
                f"找不到 {relative}；容器里先把 contracts/ 与 kernels/ 复制到 tmp/port-deps"
            )
        resolved.append(found)
    return resolved[0], resolved[1], resolved[2]


class FenceTest(unittest.TestCase):
    def test_python_fence_with_prose(self) -> None:
        raw = "先说明\n```python\nx = 1\n```\n后记"
        self.assertEqual(strip_code_fence(raw), "x = 1")

    def test_choose_python_block_over_other_blocks(self) -> None:
        raw = "```text\n一些说明\n```\n```py\ny = 2\n```"
        self.assertEqual(strip_code_fence(raw), "y = 2")

    def test_no_fence_passthrough(self) -> None:
        self.assertEqual(strip_code_fence("  z = 3\n"), "z = 3")

    def test_unterminated_fence(self) -> None:
        raw = "```python\nclass A:\n    pass\n"
        self.assertEqual(strip_code_fence(raw), "class A:\n    pass")

    def test_inline_fence_markers_removed(self) -> None:
        self.assertEqual(strip_code_fence("```Qwen3ForCausalLM```"), "Qwen3ForCausalLM")

    def test_non_string_rejected(self) -> None:
        with self.assertRaises(TypeError):
            strip_code_fence(None)  # type: ignore[arg-type]


class ValidateTest(unittest.TestCase):
    def test_good_code_passes(self) -> None:
        self.assertEqual(validate_model_code(GOOD_CODE), [])

    def test_annotated_class_attribute_passes(self) -> None:
        code = GOOD_CODE.replace(
            "    packed_modules_mapping = {",
            "    packed_modules_mapping: dict = {",
        )
        self.assertEqual(validate_model_code(code), [])

    def test_syntax_error_reports_line(self) -> None:
        reasons = validate_model_code("class X(:\n    pass\n")
        self.assertEqual(len(reasons), 1)
        self.assertIn("不是合法 Python", reasons[0])
        self.assertIn("第 1 行", reasons[0])

    def test_missing_class(self) -> None:
        code = GOOD_CODE.replace("class Qwen3ForCausalLM", "class Qwen3Model")
        reasons = validate_model_code(code)
        self.assertTrue(any("ForCausalLM" in reason for reason in reasons))

    def test_missing_packed_mapping(self) -> None:
        code = re.sub(r"    packed_modules_mapping = \{.*?\n    \}\n", "", GOOD_CODE, flags=re.S)
        reasons = validate_model_code(code)
        self.assertTrue(any("packed_modules_mapping" in reason for reason in reasons))

    def test_missing_context_import(self) -> None:
        code = GOOD_CODE.replace(
            "from nanovllm.utils.context import get_context", "from nanovllm.utils import context"
        )
        reasons = validate_model_code(code)
        self.assertTrue(any("get_context" in reason for reason in reasons))

    def test_empty_code(self) -> None:
        self.assertEqual(validate_model_code("   \n"), ["生成物为空"])


class GenerateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec, cls.kernels, cls.template = _materials()

    def test_fake_llm_round_trip(self) -> None:
        seen: list[list[dict]] = []

        def fake_llm(messages: list[dict]) -> str:
            seen.append(messages)
            return "这是生成的模型：\n```python\n" + GOOD_CODE + "```\n以上。"

        code = generate_model(GRAPH_PATH, self.spec, self.kernels, self.template, fake_llm)
        self.assertEqual(code.strip(), GOOD_CODE.strip())
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0]["role"], "system")
        self.assertEqual(seen[0][1]["role"], "user")
        self.assertIn("一、任务", seen[0][1]["content"])

    def test_invalid_generated_code_raises(self) -> None:
        bad = GOOD_CODE.replace("from nanovllm.utils.context import get_context", "")
        with self.assertRaises(ValidationError) as caught:
            generate_model(GRAPH_PATH, self.spec, self.kernels, self.template, lambda _: bad)
        self.assertTrue(any("get_context" in reason for reason in caught.exception.reasons))

    def test_empty_reply_raises(self) -> None:
        with self.assertRaises(ValidationError):
            generate_model(GRAPH_PATH, self.spec, self.kernels, self.template, lambda _: "```python\n```")

    def test_non_string_reply_raises(self) -> None:
        with self.assertRaises(TypeError):
            generate_model(GRAPH_PATH, self.spec, self.kernels, self.template, lambda _: None)


class PromptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec, cls.kernels, cls.template = _materials()
        cls.sections = build_sections(GRAPH_PATH, cls.spec, cls.kernels, cls.template)
        cls.prompt = "\n\n".join(cls.sections[name] for name in SECTION_ORDER)
        cls.graph = load_graph(GRAPH_PATH)

    def test_all_sections_present_in_order(self) -> None:
        markers = ["一、任务", "二、参考图", "三、引擎规约", "四、算子库索引", "五、同构模板"]
        positions = [self.prompt.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_graph_is_folded(self) -> None:
        nodes = self.graph["nodes"]
        self.assertEqual(len(nodes), 22)
        self.assertEqual(len(self.graph["edges"]), 29)
        headings = re.findall(r"^### \d+\. `", self.sections["graph"], flags=re.M)
        self.assertEqual(len(headings), len(nodes))
        self.assertNotIn("layers.27", self.prompt)
        self.assertIn("×28", self.sections["graph"])

    def test_spec_and_kernels_rendered(self) -> None:
        self.assertIn("M1", self.prompt)
        self.assertIn("get_context", self.prompt)
        self.assertIn("只能看写法", self.prompt)
        self.assertIn("同构同式", self.prompt)

    def test_prompt_length_is_bounded(self) -> None:
        self.assertGreater(len(self.prompt), 20_000)
        self.assertLess(len(self.prompt), 200_000)

    def test_stats(self) -> None:
        stats = prompt_stats(self.sections, GRAPH_PATH, self.spec)
        self.assertEqual(stats["chars"], len(self.prompt))
        self.assertEqual(stats["nodes"], 22)
        self.assertEqual(stats["instance_nodes"], 453)
        spec = load_spec(self.spec)
        self.assertEqual(stats["must"], len(spec["must"]))
        for field, entries in (("toolbox", spec["toolbox"]), ("context_fields", spec["context"]["fields"])):
            self.assertEqual(stats[field], len(entries))
            for entry in entries:
                self.assertIn(f"`{entry['name']}`", self.sections["spec"])

    def test_messages_shape(self) -> None:
        messages = build_messages(GRAPH_PATH, self.spec, self.kernels, self.template)
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], self.prompt)

    def test_toolbox_can_be_omitted(self) -> None:
        sections = build_sections(
            GRAPH_PATH, self.spec, self.kernels, self.template, include_toolbox=False
        )
        self.assertNotIn("3.2 toolbox", sections["spec"])
        self.assertLess(len(sections["spec"]), len(self.sections["spec"]))

    def test_graph_dict_and_path_agree(self) -> None:
        from_path = build_prompt(GRAPH_PATH, self.spec, self.kernels, self.template)
        from_dict = build_prompt(self.graph, self.spec, self.kernels, self.template)
        self.assertEqual(from_path, from_dict)


class CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec, cls.kernels, cls.template = _materials()

    def _argv(self, *extra: str) -> list[str]:
        return [
            "--graph",
            str(GRAPH_PATH),
            "--spec",
            str(self.spec),
            "--kernels",
            str(self.kernels),
            "--template",
            str(self.template),
            *extra,
        ]

    def test_dry_run_prints_stats_without_network(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = cli.main(self._argv("--dry-run"))
        self.assertEqual(code, 0)
        output = stdout.getvalue()
        self.assertIn("chars:", output)
        self.assertIn("nodes: 22", output)
        self.assertIn("edges: 29", output)

    def test_missing_key_returns_2_without_raising(self) -> None:
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {"PORT_TEST_UNSET_KEY": ""}):
            with contextlib.redirect_stderr(stderr):
                code = cli.main(self._argv("--api-key-env", "PORT_TEST_UNSET_KEY"))
        self.assertEqual(code, 2)
        self.assertIn("PORT_TEST_UNSET_KEY", stderr.getvalue())

    def test_generation_writes_file_with_injected_llm(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "tmp") as workdir:
            out = Path(workdir) / "generated.py"
            with mock.patch.object(
                cli, "make_llm", return_value=lambda _: "```python\n" + GOOD_CODE + "```"
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = cli.main(self._argv("--out", str(out)))
            self.assertEqual(code, 0)
            self.assertIn("class Qwen3ForCausalLM", out.read_text(encoding="utf-8"))

    def test_missing_material_errors(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = self._argv("--dry-run")
        argv[argv.index("--template") + 1] = str(REPO_ROOT / "tmp" / "port-missing-template.py")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                cli.main(argv)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("找不到同构模板", stderr.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
