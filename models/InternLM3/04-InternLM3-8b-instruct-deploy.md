<h1>InternLM3-8b-instruct部署</h1>

VLLM (Virtual-Large-Language-Model)是伯克利大学开源的大语言模型高速推理框架，使用了全新的注意力算法「PagedAttention」，
有效地管理Attention中的K和V。本小节我们将要基于VLLM，完成对InternLM3-8b-insturct的部署。

<p>具体内容可查阅官方文档：[https://github.com/vllm-project/vllm]</p>

## 环境配置依赖

环境依赖如下：
```
----------------------
 Transformer >=4.48 
 Torch == 2.3.0     
 Cuda ==  12.1      
----------------------
```

 >本文默认学习者已安装好以上 Pytorch(cuda) 环境，如未安装请自行安装。

## 准备工作

首先 `pip` 换源加速下载并安装依赖包：

```shell
# 升级pip
python -m pip install --upgrade pip
pip install vllm --download vllm
```
> 考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了 InternLM3-8b-Instruct 的环境镜像，点击下方链接并直接创建 AutoDL 示例即可。
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/InternLM3-self-llm***

## 模型下载

`modelscope` 是一个模型管理和下载工具，支持从魔搭 (Modelscope) 等平台快速下载模型。

这里使用 `modelscope` 中的 `snapshot_download` 函数下载模型，第一个参数为模型名称，第二个参数 `cache_dir` 为模型的下载路径，第三个参数 `revision` 为模型的版本号。

在 `/root/autodl-tmp` 路径下新建 `model_download.py` 文件并在其中粘贴以下代码，并保存文件。

```python
from modelscope import snapshot_download, AutoModel, AutoTokenizer

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm3-8b-instruct', cache_dir='./', revision='master')
```

> 注意：记得修改 cache_dir 为你的模型下载路径哦~
在终端运行 `python /root/autodl-tmp/model_download.py` 执行下载，模型大小为 18GB 左右，下载模型大概需要5-30分钟。

