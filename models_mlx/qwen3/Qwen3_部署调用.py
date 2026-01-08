"""
MLX-LM 简单文本生成示例
一次性生成，适合快速测试
"""

from functools import wraps
import time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def time_logger(task_name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = task_name or func.__name__
            print(f"🚀 开始：{name}")
            s = time.time()
            out = func(*args, **kwargs)
            print(f"✅ {name} 完成，耗时 {time.time()-s:.2f} 秒")
            return out
        return wrapper
    return decorator

class Config:
    def __init__(self):
        self.local_dir = "../models" # 本地模型存放路径
        self.model_name = "Qwen3-8B-4bit" # 模型

cfg = Config()   
def get_cfg():
    return cfg
    
class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.init_model()

    @time_logger(task_name=f"加载模型 {get_cfg().model_name}")
    def init_model(self):
        self.model, self.tokenizer = load(f"{self.cfg.local_dir}/{self.cfg.model_name}")
        mx.eval()  # 确保模型加载完成

    def generate(self, prompt: str):
        # 创建采样器
        sampler = make_sampler(temp=0.7, top_p=0.8)
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt,
            max_tokens=256,
            sampler=sampler,
            verbose=False
        )
        mx.eval()  # 确保所有计算完成
        return response

    def run(self, question: str = None):
        # 构建提示
        messages = [
            {"role": "system", "content": "你是一个智能助手。"},
            {"role": "user", "content": question or "请用一句话解释什么是人工智能？"}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # 生成回答
        s_time = time.time()
        response = self.generate(prompt)
        gen_time = time.time() - s_time

        # 统计 prompt tokens
        prompt_tokens = len(self.tokenizer.encode(prompt))
        # 统计生成 tokens
        response_tokens = len(self.tokenizer.encode(response))

        print("\n🤖 模型回复\n" + "=" * 50)
        print(response)
        print("=" * 50)

        # 打印统计信息
        print("\n📊 性能统计")
        print("-" * 30)
        print(f"  Prompt tokens:   {prompt_tokens}")
        print(f"  生成 tokens:     {response_tokens}")
        print(f"  生成总耗时:      {gen_time:.2f}s")
        print(f"  生成速度:        {response_tokens / gen_time:.1f} tokens/s")
        print(f"  峰值内存:        {mx.get_peak_memory() / 1024**3:.2f} GB")
        print("-" * 30)

if __name__ == "__main__":
    app = App(get_cfg())
    app.run("请用一句话解释什么是人工智能？")