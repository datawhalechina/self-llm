#  Phi-3-mini-4k-instruct FastApi 部署调用

## 环境准备

在 autodl 平台中租赁一个 3090 等 24G 显存的显卡机器，如下图所示镜像选择 PyTorch-->2.0.0-->3.8(ubuntu20.04)-->11.8 。

接下来打开刚刚租用服务器的 JupyterLab，并且打开其中的终端开始环境配置、模型下载和运行演示。  

![机器配置选择](../InternLM2/images/1.png)

### 创建工作目录

创建本次phi3实践的工作目录`/root/autodl-tmp/phi3`

```bash
# 创建工作目录
mkdir -p /root/autodl-tmp/phi3
```

### 安装依赖

```bash
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install fastapi==0.104.1
pip install uvicorn==0.24.0.post1
pip install requests==2.25.1
pip install modelscope==1.9.5
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
```

由于phi3要求的transformers的版本为`4.41.0.dev0版本`。

各位可以先通过下面命令查看你的Transformers包的版本

```bash
pip list |grep transformers
```

如果版本不对，可以通过下面命令升级

```bash
# phi3升级transformers为4.41.0.dev0版本
pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers
```



## 模型下载

使用 modelscope 中的`napshot_download`函数下载模型，第一个参数为模型名称，参数`cache_dir`为模型的下载路径。

在 /root/autodl-tmp 路径下新建`download.py` 文件并在其中输入以下内容，粘贴代码后记得保存文件，如下图所示。并运行`python /root/autodl-tmp/download.py`执行下载，模型大小为 8 GB，下载模型大概需要 10~15 分钟

```python
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('LLM-Research/Phi-3-mini-4k-instruct', cache_dir='/root/autodl-tmp/phi3', revision='master')
```

## 代码准备

在`/root/autodl-tmp`路径下新建`api.py`文件并在其中输入以下内容，粘贴代码后记得保存文件。下面的代码有很详细的注释，大家如有不理解的地方，欢迎提出issue。

```python
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import datetime
import torch

# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

# 创建FastAPI应用
app = FastAPI()

# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer  # 声明全局变量以便在函数内部使用模型和分词器
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    prompt = json_post_list.get('prompt')  # 获取请求中的提示
    history = json_post_list.get('history', [])  # 获取请求中的历史记录

    print(prompt)
    messages = [
            {"role": "user", "content": prompt}
    ]

    # 调用模型进行对话生成
    input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'),max_new_tokens=2048)
   
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    

    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    # 构建日志信息
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应

# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型
    model_name_or_path = '/root/autodl-tmp/phi3/model/LLM-Research/Phi-3-mini-4k-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
        ).eval()

    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  # 在指定端口和主机上启动应用
```

默认部署在 6006 端口，通过 POST 方法进行调用，可以使用curl调用，如下所示：

```bash
curl -X POST "http://127.0.0.1:6006" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你好", "history": []}'
```
响应如下：
```json
{
    "response": "你好！如果你需要帮助或者有任何问题，请随时告诉我。",
    "status": 200,
    "time": "2024-05-09 16:36:43"
}
```

SSH端口映射

```bash
ssh -CNg -L 6006:127.0.0.1:6006 -p 【你的autodl机器的ssh端口】 root@[你的autodl机器地址]
ssh -CNg -L 6006:127.0.0.1:6006 -p 36494 root@region-45.autodl.pro
```

端口映射后，用postman访问

![phi3-fastapi](./assets/01-1.png)