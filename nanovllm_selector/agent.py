"""一次 DeepSeek 决策：只接受候选 ID，不执行模型返回的代码。"""

import json
import os

import requests

from candidates import validate_plan


SYSTEM_PROMPT = """你是推理算子选择器，不是 kernel 编写器。
阅读给出的真实 Transformers 源码、nano-vLLM 接口、运行规格及候选目录。
本次只选择 MLP 的 packed gate/up 激活实现，作用于指定模型的全部同规格 Decoder 层。
禁止修改 Linear、投影权重、Attention、残差、其他 Norm 或采样；禁止返回代码、路径或新候选 ID。
语义必须是 SiLU(gate) * up，不能交换两半，不能返回两个张量，不能丢掉 gate/up 中任何一半。
考虑设备、dtype、shape、stride、输入是否改写及返回规格。不得把未知候选当作已有候选。
这是接口接入试验，不是最快实现评测；无需因为没有 benchmark 就否决接口兼容的候选。
允许选择 native；若选择非原生，说明匹配依据和未验证风险，不宣称加速或严格数值等价。
候选的高精度中间计算可能导致 BF16 舍入差异，说明风险即可；完整数值验收安排在后续。
只返回一个 JSON 对象，字段严格为：
{"target":"mlp_activation","candidate_id":"目录中的 ID","reason":"中文说明选择依据、限制及风险"}
"""


def select(context, output_dir):
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请使用 docker exec --env-file 注入本地配置")
    request = {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
        "thinking": {"type": "enabled"}, "reasoning_effort": "high",
        "response_format": {"type": "json_object"}, "max_tokens": 8192,
    }
    # 只记录请求正文；不记录 Authorization、环境变量或密钥。
    (output_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2))
    response = requests.post(
        os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + key}, json=request, timeout=(15, 180),
    )
    response.raise_for_status()
    payload = response.json()
    (output_dir / "response.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["choices"][0].get("finish_reason") != "stop":
        raise ValueError("Agent 响应未正常结束，拒绝使用截断方案")
    choice = json.loads(payload["choices"][0]["message"]["content"])
    plan = {"schema_version": 1, "binding": context["binding"], "choice": choice}
    validate_plan(plan, context["binding"])
    (output_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan
