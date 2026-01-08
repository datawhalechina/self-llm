
from huggingface_hub import snapshot_download
import time

# import os
# # 设置 HF_TOKEN 可以加速下载
# # 在Hugging Face注册账号从 https://huggingface.co/settings/tokens 获取
# os.environ["HF_TOKEN"] = "hf_xxx" 
# from huggingface_hub import whoami
# print(f"✅ 当前Hugging Face用户: {whoami(token=os.environ['HF_TOKEN'])['name']}")

LOCAL_DIR = "./models/" # 本地模型存放路径
# Qwen3-0.6B-4bit
# Qwen3-8B-4bit
MODEL_NAME = "Qwen3-0.6B-4bit" # 模型名称

from functools import wraps

def time_logger(task_name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = task_name or func.__name__
            print(f"🚀 开始：{name}")
            s = time.time()
            out = func(*args, **kwargs)
            print()
            print(f"✅ {name} 完成，耗时 {time.time()-s:.2f} 秒")
            return out
        return wrapper
    return decorator

def download_model(model_name, local_dir):
    snapshot_download(
        repo_id=f"mlx-community/{model_name}",
        local_dir=f"{local_dir}/{model_name}",
    )

if __name__ == "__main__":
    time_logger(task_name=f"下载模型 {MODEL_NAME}")(download_model)(MODEL_NAME, LOCAL_DIR)
    # download_model(MODEL_NAME, LOCAL_DIR)