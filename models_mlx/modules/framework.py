"""
推理框架模块
提供 MLXBackend 和 TransformersBackend 两个类，封装模型加载与生成逻辑

独立运行: python -m modules.framework
"""

import time
from abc import ABC, abstractmethod

from modules.core_types import Framework


class BaseBackend(ABC):
    """推理后端基类"""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load(self, model_path):
        """加载模型"""

    @abstractmethod
    def generate(self, prompt, temperature=0.7, top_p=0.8, max_tokens=512):
        """流式生成文本，yield 累积的响应字符串"""

    @property
    def is_loaded(self):
        return self.model is not None


class MLXBackend(BaseBackend):
    """MLX 推理后端（Apple Silicon 加速）"""

    framework = Framework.MLX

    def load(self, model_path):
        import mlx.core as mx
        from mlx_lm import load

        self.model, self.tokenizer = load(model_path)
        mx.eval()

    def generate(self, prompt, temperature=0.7, top_p=0.8, max_tokens=512):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature, top_p=top_p)
        response = ""
        for chunk in stream_generate(
            self.model, self.tokenizer,
            prompt=prompt, max_tokens=max_tokens, sampler=sampler,
        ):
            response += chunk.text
            yield response


class TransformersBackend(BaseBackend):
    """HuggingFace Transformers 推理后端"""

    framework = Framework.TRANSFORMERS

    def load(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32,
            device_map="cpu", trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, prompt, temperature=0.7, top_p=0.8, max_tokens=512):
        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=20,
                do_sample=True,
                eos_token_id=[151645, 151643],
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        yield response


def create_backend(framework):
    """工厂函数：根据框架名称创建对应后端"""
    fw = Framework(framework) if not isinstance(framework, Framework) else framework
    if fw == Framework.MLX:
        return MLXBackend()
    return TransformersBackend()


# ============================================================
# 🚀 独立运行：交互式推理
# ============================================================

if __name__ == "__main__":
    from modules.download_model import scan_local_models

    print("=" * 50)
    print("💬 模型推理工具")
    print("=" * 50)

    local_models = scan_local_models()
    if not local_models:
        print("❌ 未找到本地模型，请先下载模型")
        exit(1)

    print("\n可用模型:")
    for i, m in enumerate(local_models, 1):
        print(f"  {i}. {m['label']}")

    idx = int(input(f"\n请选择模型 [1-{len(local_models)}]（默认 1）: ").strip() or "1") - 1
    selected = local_models[idx]

    # 根据路径推断框架
    parts = selected["path"].replace("\\", "/").split("/")
    framework = "mlx" if "mlx" in parts else "transformers"
    print(f"\n🔧 使用框架: {framework}")

    backend = create_backend(framework)
    print(f"⏳ 加载模型: {selected['model']}")
    s = time.time()
    backend.load(selected["path"])
    print(f"✅ 加载完成，耗时 {time.time() - s:.2f} 秒")

    print("\n开始对话（输入 quit 退出）:\n")
    messages = [{"role": "system", "content": "你是一个智能助手。"}]

    while True:
        user_input = input("用户: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})
        prompt = backend.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        print("助手: ", end="", flush=True)
        response = ""
        for partial in backend.generate(prompt):
            new_text = partial[len(response):]
            print(new_text, end="", flush=True)
            response = partial
        print()

        messages.append({"role": "assistant", "content": response})
