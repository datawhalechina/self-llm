#!/usr/bin/env python3
"""
Gemma4 教程冒烟测试（不默认加载整模权重，避免 OOM；可按需开启网络拉取 Processor）。

用法:
  /path/to/self-llm/bin/python verify_gemma4_tutorials.py
  GEMMA_PULL_PROCESSOR=1 ...   # 从 Hub 拉取 Gemma4Processor（需已安装 torchvision，与 torch 同 CUDA 版本）

说明：本脚本不做整模加载/推理；真机部署请参考各 .md 并自行下载权重。
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path
from collections import UserDict
from typing import Any
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "dataset" / "huanhuan.json"
MODEL_ID = "google/gemma-4-E4B-it"


def _ok(name: str, detail: str = "") -> None:
    print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, err: BaseException) -> None:
    print(f"[FAIL] {name}: {err}")
    traceback.print_exc()


def test_imports() -> bool:
    import numpy
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    assert hasattr(torch, "cuda")
    _ok(
        "import 栈",
        f"numpy={numpy.__version__}, torch={torch.__version__}, "
        f"cuda={torch.cuda.is_available()}, transformers={transformers.__version__}",
    )
    _ = AutoModelForMultimodalLM
    _ = AutoProcessor
    return True


def test_pull_processor() -> bool:
    if os.environ.get("GEMMA_PULL_PROCESSOR", "").strip() not in ("1", "true", "yes"):
        print("[SKIP] Processor Hub 拉取（设置 GEMMA_PULL_PROCESSOR=1 可开启）")
        return True
    from transformers import AutoProcessor

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    dt = time.time() - t0
    _ok("AutoProcessor.from_pretrained", f"{type(proc).__name__}, {dt:.1f}s")
    return True


def test_fastapi_smoke_with_mock() -> bool:
    """教程 01 路由逻辑：用 Mock 模型验证请求/响应（避免真推理）。"""
    from contextlib import asynccontextmanager

    from fastapi import Body, FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from pydantic import BaseModel, Field, model_validator
    from typing import List, Literal, Optional

    class ContentItem(BaseModel):
        type: Literal["text", "image"]
        text: Optional[str] = Field(None, description="文本")
        image: Optional[str] = Field(None, description="图片 URL 或 base64")

        @model_validator(mode="after")
        def _v(self):
            if self.type == "text" and not (self.text and str(self.text).strip()):
                raise ValueError("文本类型必须提供 text 字段")
            if self.type == "image":
                img = self.image or ""
                if not img.startswith(("http://", "https://", "data:image")):
                    raise ValueError("图片必须是有效的 URL 或 base64(data:image)")
            return self

    class Message(BaseModel):
        role: Literal["system", "user", "assistant"]
        content: List[ContentItem]

    class ProcessRequest(BaseModel):
        messages: List[Message] = Field(..., min_length=1)
        max_new_tokens: int = Field(1000, ge=10, le=4096)

    class ProcessResponse(BaseModel):
        response: str
        status: int
        time: int
        processing_time: float
        tokens_generated: int

    import torch

    class _Batch(UserDict):
        def to(self, _device):
            return self

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate = MagicMock(return_value=torch.tensor([[1, 2, 99, 100]]))

    mock_processor = MagicMock()
    mock_processor.apply_chat_template = MagicMock(
        return_value=_Batch(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[1, 1]]),
        )
    )
    mock_processor.decode = MagicMock(return_value="<mock>")
    mock_processor.parse_response = MagicMock(side_effect=Exception("no parse"))

    model_ref = {"m": mock_model, "p": mock_processor}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(lifespan=lifespan)
    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def _normalize_content_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for it in items:
            if it.get("type") == "text":
                out.append({"type": "text", "text": it.get("text") or ""})
            elif it.get("type") == "image":
                img = it.get("image") or it.get("url")
                if not img:
                    continue
                if str(img).startswith(("http://", "https://")):
                    out.append({"type": "image", "url": img})
                else:
                    out.append({"type": "image", "image": img})
        return out

    @app.post("/chat/completions", response_model=ProcessResponse)
    async def generate_response(chat: ProcessRequest = Body(...)):
        start_time = time.time()
        try:
            model, processor = model_ref["m"], model_ref["p"]
            processed_messages = []
            system_prompt = DEFAULT_SYSTEM_PROMPT
            for msg in chat.messages:
                if msg.role == "system":
                    system_prompt = " ".join([item.text or "" for item in msg.content if item.type == "text"])
                else:
                    d = msg.model_dump()
                    d["content"] = _normalize_content_items(d["content"])
                    processed_messages.append(d)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                *processed_messages,
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            max_token_num = min(4096, int(chat.max_new_tokens))
            with torch.inference_mode():
                generation = model.generate(**inputs, max_new_tokens=max_token_num, do_sample=False)
            response_ids = generation[0][input_len:]
            raw = processor.decode(response_ids, skip_special_tokens=False)
            try:
                parsed = processor.parse_response(raw)
                decoded = parsed.get("content", raw) if isinstance(parsed, dict) else raw
            except Exception:
                decoded = processor.decode(response_ids, skip_special_tokens=True)
            return ProcessResponse(
                response=str(decoded),
                status=200,
                time=int(time.time()),
                processing_time=time.time() - start_time,
                tokens_generated=int(len(response_ids)),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你好，只做连通性测试"},
                ],
            }
        ],
        "max_new_tokens": 64,
    }
    with TestClient(app) as client:
        r = client.post("/chat/completions", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "response" in body
    _ok("FastAPI /chat/completions (Mock)", f"tokens_generated={body.get('tokens_generated')}")
    return True


def test_lora_dataset() -> bool:
    if not DATASET.is_file():
        print(f"[SKIP] 数据集不存在: {DATASET}")
        return True
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    assert isinstance(raw, list) and len(raw) > 0
    first = raw[0]
    for k in ("instruction", "input", "output"):
        assert k in first
    _ok("05 LoRA 数据集", f"{DATASET.name} 条数={len(raw)}")
    return True


def test_evalscope_import() -> bool:
    from evalscope.config import TaskConfig
    from evalscope.run import run_task

    _ = TaskConfig
    _ = run_task
    _ok("evalscope 导入", "TaskConfig / run_task 可用（完整评测需 Ollama 等服务）")
    return True


def main() -> int:
    tests = [
        ("环境导入", test_imports),
        ("05 嬛嬛数据集", test_lora_dataset),
        ("01 FastAPI Mock", test_fastapi_smoke_with_mock),
        ("04 evalscope", test_evalscope_import),
        ("Hub Processor", test_pull_processor),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            failed += 1
            _fail(name, e)
    print("---")
    print(f"完成: {len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
