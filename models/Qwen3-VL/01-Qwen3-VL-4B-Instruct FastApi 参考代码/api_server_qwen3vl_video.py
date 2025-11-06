#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_num_threads(8)

# 创建FastAPI应用
app = FastAPI(title="Qwen3-VL-4B Video API", version="1.0.0")

# 模型路径
model_name_or_path = '/root/autodl-fs/Qwen/Qwen3-VL-4B-Instruct'

# 初始化模型和处理器
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)

# 请求模型
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

# 响应模型
class ChatResponse(BaseModel):
    response: str
    model: str = "Qwen3-VL-4B-Instruct"
    usage: Dict[str, int]

@app.get("/")
async def root():
    return {"message": "Qwen3-VL-4B-Instruct Video API Server is running!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "Qwen3-VL-4B-Instruct",
        "device": str(model.device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
        "supported_formats": ["image", "video"]
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    try:
        # 处理消息
        messages = request.messages
        
        # 处理视觉信息（包括图像和视频）
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 准备输入
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # 生成响应
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # 解码响应
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 计算token使用量
        input_tokens = inputs.input_ids.shape[1]
        output_tokens = len(generated_ids_trimmed[0])
        
        return ChatResponse(
            response=response_text,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )

# 兼容原有的 /generate 接口
@app.post("/generate")
async def generate_response(request: ChatRequest):
    """兼容原有接口格式"""
    result = await chat_completions(request)
    return {"response": result.response}

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )