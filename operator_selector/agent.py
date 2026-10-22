"""调用 DeepSeek，输出已有算子的执行方案，不生成 kernel。"""

import json
import os

import requests

def select_plan(context, catalog, output_dir):
    prompt = {
        "task": context,
        "candidates": catalog,
        "instruction": (
            "阅读真实模块源码和激活源码，划分 forward 计算，并选择已有算子。"
            "只返回 JSON，不要 Markdown 或 kernel 源码。保留源码中所有 Linear 的原始权重与计算，不拆分或合并 Linear；"
            "根据源码识别非 Linear 计算片段，用目录中的基础操作或融合候选覆盖它，保持完整 forward 的数学语义。"
            "考虑 dtype、布局和输入适配开销；未测量前不要保证加速。"
            "步骤顺序必须使输入已定义，x 是唯一外部输入，每一步调用目录中的一个 ID。"
            "候选中的 parameters 只是接口形参，不是方案的张量名；方案 inputs 只能引用 x 或此前步骤的 output。"
            "直接读取模型输入的步骤必须引用 x，不能写 input、hidden_states 等源码或形参名称。"
            "JSON 格式为 {\"steps\":[{\"op\":\"候选ID\",\"inputs\":[\"张量名\"],"
            "\"output\":\"结果名\"}],\"output\":\"最终结果名\",\"reason\":\"解释划分与选择的中文理由\"}。"
            "若一个操作返回两个 Tensor，该步骤的 output 写成两个名字的数组，顺序与候选接口一致。"
        ),
    }
    (output_dir / "prompt.json").write_text(json.dumps(prompt, ensure_ascii=False, indent=2))
    body = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        "messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
        "max_tokens": 16384,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    response = requests.post(
        os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]},
        json=body, timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    (output_dir / "response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    plan = json.loads(payload["choices"][0]["message"]["content"])
    candidates = {candidate["id"]: candidate for candidate in catalog}
    defined = {"x"}
    for step in plan["steps"]:
        assert step["op"] in candidates, f"未知候选：{step['op']}"
        assert len(step["inputs"]) == len(candidates[step["op"]]["parameters"]), "候选参数数量不匹配"
        assert set(step["inputs"]) <= defined, f"引用未定义的张量：{step['inputs']}"
        outputs = step["output"] if isinstance(step["output"], list) else [step["output"]]
        defined.update(outputs)
    assert plan["output"] in defined, "最终输出未定义"
    return plan
