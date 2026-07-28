# api.py — 与教程 01 一致，已适配 Pydantic v2 / FastAPI，并支持本地模型路径
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModelForMultimodalLM, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda"
DEVICE_ID = os.environ.get("CUDA_DEVICE_ID", "0")
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

MODEL_PATH = os.environ.get("GEMMA_MODEL_PATH", "/dataset/gemma-4-E4B-it")

model = None
processor = None
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ContentItem(BaseModel):
    type: Literal["text", "image"]
    text: Optional[str] = Field(None, description="文本内容（当 type 为 text 时必填）")
    image: Optional[str] = Field(None, description="图片 URL 或 base64（当 type 为 image 时必填）")

    @model_validator(mode="after")
    def validate_content(self):
        if self.type == "text":
            if not self.text or not str(self.text).strip():
                raise ValueError("文本类型必须提供 text 字段")
        elif self.type == "image":
            img = self.image or ""
            if not str(img).startswith(("http://", "https://", "data:image")):
                raise ValueError("图片必须是有效的 URL 或 base64 编码字符串")
        return self


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: List[ContentItem]


class ProcessRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1, description="对话历史记录")
    max_new_tokens: int = Field(1000, ge=10, le=4096, description="生成的最大 token 数")


class ProcessResponse(BaseModel):
    response: str
    status: int
    time: int
    processing_time: float
    tokens_generated: int


def load_models():
    global model, processor
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"模型目录不存在: {MODEL_PATH}")
    try:
        logger.info("正在加载模型: %s", MODEL_PATH)
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        logger.info("正在加载处理器...")
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        logger.info("模型加载完成 device=%s", getattr(model, "device", "?"))
    except Exception as e:
        logger.error("模型加载失败: %s", e)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_models()
        yield
    except Exception as e:
        logger.error("服务初始化失败: %s", e)
        raise
    finally:
        torch_gc()


app = FastAPI(lifespan=lifespan)


def _normalize_content_items(items):
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
async def generate_response(payload: ProcessRequest = Body(...)):
    start_time = time.time()
    try:
        processed_messages = []
        system_prompt = DEFAULT_SYSTEM_PROMPT

        for msg in payload.messages:
            if msg.role == "system":
                system_prompt = " ".join(
                    [item.text or "" for item in msg.content if item.type == "text"]
                )
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
        max_token_num = min(4096, int(payload.max_new_tokens))
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_token_num,
                do_sample=False,
            )
        response_ids = generation[0][input_len:]
        raw = processor.decode(response_ids, skip_special_tokens=False)
        try:
            parsed = processor.parse_response(raw)
            decoded = parsed.get("content", raw) if isinstance(parsed, dict) else raw
        except Exception:
            decoded = processor.decode(response_ids, skip_special_tokens=True)

        ntok = int(response_ids.numel()) if hasattr(response_ids, "numel") else len(response_ids)
        return ProcessResponse(
            response=str(decoded),
            status=200,
            time=int(time.time()),
            processing_time=time.time() - start_time,
            tokens_generated=ntok,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("处理请求时出错: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)
