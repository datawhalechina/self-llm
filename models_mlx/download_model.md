开源模型参数都可以从 [Hugging Face: mlx-community](https://huggingface.co/mlx-community/models) 上下载。

**注意！！！模型下载过程通常较久，请耐心等待**

```python
from huggingface_hub import snapshot_download
import time
import os
# 设置Hugging Face访问令牌可以加速下载
# 在Hugging Face注册账号从 https://huggingface.co/settings/tokens 获取
# os.environ["HF_TOKEN"] = "hf_xxx" 
LOCAL_DIR = "./models/" # 本地模型存放路径
MODEL_NAME = "Qwen3-0.6B-4bit" # 模型名称
def download_model(model_name, local_dir):
    snapshot_download(
        repo_id=f"mlx-community/{model_name}",
        local_dir=f"{local_dir}/{model_name}",
    )
s_t = time.time()
download_model(MODEL_NAME, LOCAL_DIR)
e_t = time.time()
print(f"✅ 模型 {MODEL_NAME} 下载完成，耗时 {e_t - s_t:.2f} 秒")
```