# Qwen2-VL-2B-Instruct WebDemo 部署


# 环境准备

```
----------------
ubuntu 22.04
python 3.10
cuda 11.8
----------------
```

# 环境安装

```python
# 创建环境
# conda init bash
conda create -n qwen2vl_wb python=3.10
conda activate qwen2vl_wb

# 换源
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 需要安装的库
# 安装transformers时会依赖最新的torch版本, 目前为 torch2.5.1
pip install qwen_vl_utils==0.0.8 transformers==4.46.2 accelerate==1.1.0 gradio==5.5.0
pip install torchvision  # 会匹配对应torch版本的依赖

# 安装flash-attn
# 如显卡支持flash-attn，在确认对应python、pytorch、cuda版本后, 下载对应的release版本.
wegt https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# 镜像加速链接:
# wget https://github.moeyy.xyz/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.0.post2+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
> 如需完整的pip列表(包含依赖)请参考[02-Qwen2-VL-2B-Instruct Web Demo 参考代码/requirements.txt](./02-Qwen2-VL-2B-Instruct%20Web%20Demo%20参考代码/requirements.txt)

# 下载模型

```python
# 进入autodl-tmp/
cd autodl-tmp/

# 首先安装lfs，便于通过git直接下载模型。
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# 需要下载的模型
MODEL=Qwen2-VL-2B-Instruct
# MODEL=Qwen2-VL-7B-Instruct
# MODEL=Qwen2-VL-72B-Instruct

# # huggingface 下载
# URL="https://huggingface.co/Qwen/"
# git clone "${URL}/${MODEL}"

# 魔搭下载
URL="https://www.modelscope.cn/Qwen"
git clone "${URL}/${MODEL}.git"

# 返回根目录
cd ..
```

# 运行Demo

```python
# Ampere/Ada/Hopper架构显卡可以启用flash attn2加速推理，autodl要通过6006端口对外访问。
# python mm_qwen2vl.py --flash-attn2 --model-path ./autodl-tmp/Qwen2-VL-2B-Instruct --host 0.0.0.0 --port 6006
python mm_qwen2vl.py --model-path ./autodl-tmp/Qwen2-VL-2B-Instruct --host 0.0.0.0 --port 6006
```
> 完整代码请参考[mm_qwen2vl.py](./02-Qwen2-VL-2B-Instruct%20Web%20Demo%20参考代码/mm_qwen2vl.py)

# 测试效果

![image.png](./images/02-1.png)

> 如果觉得2B理解能力较差, 建议用7B以上模型.