from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Atom(LLM):
    # 基于本地 Atom 自定义 LLM 类
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path :str):
        # model_path: Atom 模型路径
        # 从本地初始化模型
        super().__init__()
        print("正在从本地加载模型...")
        model_dir = '/root/autodl-tmp/FlagAlpha/Atom-7B-Chat'
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True,torch_dtype=torch.float16).eval()
        print("完成本地模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any):
        input_ids = self.tokenizer([f'<s>Human: {prompt}\n</s><s>Assistant: '], return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')
        generate_input = {
            "input_ids": input_ids,
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.3,
            "repetition_penalty": 1.3,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }
        generate_ids = self.model.generate(**generate_input)
        text = self.tokenizer.decode(generate_ids[0])
        return text

    @property
    def _llm_type(self) -> str:
        return "Atom"
