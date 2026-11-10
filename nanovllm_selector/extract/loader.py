"""模型加载：M1 只要 config，不要权重。

`from_config` 按 config 随机初始化，省掉几个 GB 的下载和 bf16 舍入误差，
形状、模块树、执行路径与真实权重完全一致。
"""

from __future__ import annotations

import hashlib
import json
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_sha256(model_path: str) -> str | None:
    p = os.path.join(model_path, "config.json")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def weights_sha256(model_path: str) -> str | None:
    """M1 不加载权重；把权重文件的哈希记下来，供 M2 对比环境用。"""
    for name in ("model.safetensors",):
        p = os.path.join(model_path, name)
        if os.path.exists(p):
            from .source import sha256_file

            return sha256_file(p)
    return None


def load_model(
    model_path: str,
    weights: bool = False,
    dtype: str = "float32",
    device: str = "cpu",
    trust_remote_code: bool = False,
):
    """返回 (model, config)。

    weights=False 时不会读取任何 .safetensors，只读 config.json。
    """
    torch_dtype = _DTYPES[dtype]
    config = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    if weights:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
        model = model.to(dtype=torch_dtype)

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    model._extract_meta = {
        "model_path": os.path.abspath(model_path),
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", []) or []),
        "num_layers": getattr(config, "num_hidden_layers", None),
        "dtype": dtype,
        "device": device,
        "weights_loaded": bool(weights),
        "config_sha256": config_sha256(model_path),
        "weights_sha256": weights_sha256(model_path),
        "config_json": json.loads(json.dumps(config.to_dict(), default=str)),
        "config_fingerprint": _sha256_text(json.dumps(config.to_dict(), sort_keys=True, default=str)),
    }
    return model, config
