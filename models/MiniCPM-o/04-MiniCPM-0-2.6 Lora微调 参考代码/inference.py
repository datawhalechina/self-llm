from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from PIL import Image

model_path=  "/home/linzeyi/minicpm-lora/OpenBMB/MiniCPM-o-2_6"
path_to_adapter="/home/linzeyi/minicpm-lora/output/output__lora/checkpoint-1000"

model =  AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
        )

lora_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

image = Image.open('/home/linzeyi/minicpm-lora/LaTeX_OCR/0.jpg').convert('RGB')

question = "这张图对应的LaTex公式是什么？"
msgs = [{'role': 'user', 'content': [image, question]}]

answer = lora_model.chat(
    msgs=msgs,
    tokenizer=tokenizer
)
print(answer)