from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils.vision_process import process_vision_info
from fastapi import FastAPI, Request
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Union

app = FastAPI()

model_name_or_path = '/root/autodl-tmp/Qwen/Qwen2-VL-2B-Instruct'
# 初始化模型和处理器（保持在全局范围内，这样只需加载一次）
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name_or_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name_or_path)

# 定义请求体模型
class MessageContent(BaseModel):
    type: str
    text: str = None
    image: str = None
    video: str = None

class ChatMessage(BaseModel):
    messages: List[Dict[str, Union[str, List[Dict[str, str]]]]]

@app.post("/generate")
async def generate_response(chat_message: ChatMessage):
    # 直接使用请求中的 messages
    text = processor.apply_chat_template(
        chat_message.messages,
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(chat_message.messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return {"response": output_text[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)