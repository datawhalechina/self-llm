import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig


# 加载预训练的分词器和模型
model_path = "xverse/XVERSE-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model.generation_config = GenerationConfig.from_pretrained(model_path)

# 使用 INT8、INT4 进行量化推理 
# model = model.quantize(8).cuda()
model = model.quantize(4).cuda()

model = model.eval()

print("=============Welcome to XVERSE chatbot, type 'exit' to exit.=============")


# 设置多轮对话
while True:
    user_input = input("\n帅哥美女请输入: ")
    if user_input.lower() == "exit":
        break
    # 创建消息
    history = [{"role": "user", "content": user_input}]
    response = model.chat(tokenizer, history)
    print("\nXVERSE-7B-Chat: {}".format(response))

    # 添加回答到历史
    history.append({"role": "assistant", "content": response})

