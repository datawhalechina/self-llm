# CharacterGLM-6B-chat

## 环境准备

在autodl平台中租一个3090等24G显存的显卡机器，如下图所示镜像选择PyTorch-->2.0.0-->3.8(ubuntu20.04)-->11.8

![image](https://github.com/suncaleb1/self-llm/assets/155936975/0dddbee9-df80-4033-9568-185ea585f261)


接下来打开刚刚租用服务器的JupyterLab，并且打开其中的终端开始环境配置、模型下载和运行demo。

pip换源和安装依赖包

```python
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope
pip install transformers
```

## 模型下载

使用 modelscope 中的snapshot_download函数下载模型，第一个参数为模型名称，参数cache_dir为模型的下载路径。

在 /root/autodl-tmp 路径下新建 download.py 文件并在其中输入以下内容，粘贴代码后记得保存文件，如下图所示。并运行 python /root/autodl-tmp/download.py执行下载，模型大小为 12 GB，下载模型大概需要 10~15 分钟

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('THUCoAI/CharacterGLM-6B', cache_dir='/root/autodl-tmp', revision='master')
```

## 代码准备

首先clone代码，打开autodl平台自带的学术镜像加速。学术镜像加速详细使用请看：
https://www.autodl.com/docs/network_turbo/

```python
source /etc/network_turbo
```

然后切换路径, clone代码.

```python
cd /root/autodl-tmp
git clone https://github.com/thu-coai/CharacterGLM-6B
```

## demo运行

修改代码路径，将 /root/autodl-tmp/CharacterGLM-6B/basic_demo/web_demo_streamlit.py中第20行的模型更换为本地的/root/autodl-tmp/THUCoAI/CharacterGLM-6B

![image](https://github.com/suncaleb1/self-llm/assets/155936975/1edc97a2-3d6e-43e3-b176-644b756b615f)


修改requirements.txt文件，将其中的torch删掉，环境中已经有了torch，不需要再安装。然后执行下面的命令：

```python
cd /root/autodl-tmp/CharacterGLM-6B
pip install -r requirements.txt
```

在终端运行以下命令即可启动推理服务,尽量cd到basic_demo文件夹下，防止找不到character.json文件

```python
cd /root/autodl-tmp/CharacterGLM-6B/basic_demo
streamlit run ./web_demo2.py --server.address 127.0.0.1 --server.port 6006
```

![alt text](../image/03-运行webdemo.png)

在将 autodl 的端口映射到本地的 http://localhost:6006 后，即可看到demo界面。具体映射步骤参考文档General-Setting文件夹下/02-AutoDL开放端口.md文档。

在浏览器打开 http://localhost:6006 界面，模型加载，即可使用，如下图所示。

![alt text](../image/03-webdemo_show.png)

## 命令行运行

修改代码路径，将 /root/autodl-tmp/CharacterGLM-6B/basic_demo/cli_demo.py中第20行的模型更换为本地的/root/autodl-tmp/THUCoAI/CharacterGLM-6B

在终端运行以下命令即可启动推理服务

```python
cd /root/autodl-tmp/CharacterGLM-6B/basic_demo
python ./cli_demo.py 
```

![alt text](../image/03-运行clidemo.png)
