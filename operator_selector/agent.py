"""调用 DeepSeek，输出已有算子的执行方案，不生成 kernel。"""

import json
import os

import requests


def validate_plan(plan, catalog, external_inputs=("x",)):
    candidates = {candidate["id"]: candidate for candidate in catalog}
    defined = set(external_inputs)
    for step in plan["steps"]:
        if step["op"] not in candidates:
            raise ValueError(f"未知候选：{step['op']}")
        candidate = candidates[step["op"]]
        if len(step["inputs"]) != len(candidate["parameters"]):
            raise ValueError("候选参数数量不匹配")
        if not set(step["inputs"]) <= defined:
            raise ValueError(f"引用未定义的张量：{step['inputs']}")
        outputs = step["output"] if isinstance(step["output"], list) else [step["output"]]
        if len(outputs) != candidate.get("returns", 2 if step["op"] == "split_halves" else 1):
            raise ValueError("候选返回值数量不匹配")
        if len(set(outputs)) != len(outputs) or set(outputs) & defined:
            raise ValueError("输出张量名重复或覆盖已有张量")
        defined.update(outputs)
    if plan["output"] not in defined:
        raise ValueError("最终输出未定义")


def select_plan(context, catalog, output_dir):
    external = context.get("external_inputs", ["x"])
    prompt = {
        "task": context,
        "candidates": catalog,
        "instruction": (
            "阅读真实模块源码和激活源码，划分 forward 计算，并选择已有算子。"
            "只返回 JSON，不要 Markdown 或 kernel 源码。保留源码中所有 Linear 的原始权重与计算，不拆分或合并 Linear；"
            "根据源码识别非 Linear 计算片段，用目录中的基础操作或融合候选覆盖它，保持完整 forward 的数学语义。"
            "考虑 dtype、布局和输入适配开销；未测量前不要保证加速。"
            f"步骤顺序必须使输入已定义，唯一允许的外部张量名为 {external}，每一步调用目录中的一个 ID。"
            "候选中的 parameters 只是接口形参，不是方案的张量名；方案 inputs 只能引用外部张量或此前步骤的 output。"
            "遵守任务给出的边界映射、执行模式以及真实源码的数据依赖；不得丢失后续还需要的残差。"
            "候选中 bound_parameters 已由执行器绑定，不要把权重或 eps 另写为张量输入；不要覆盖已有张量名。"
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
    validate_plan(plan, catalog, external)
    return plan
