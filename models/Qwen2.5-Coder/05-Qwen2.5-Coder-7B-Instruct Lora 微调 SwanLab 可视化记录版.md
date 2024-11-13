# Qwen2.5-Coder-7B-Instruct Lora 微调  SwanLab 可视化记录版

本节我们简要介绍如何基于 transformers、peft 等框架，对Qwen2.5-Coder-7B-Instruct 模型进行Lora微调。使用的数据集是由Self-LLM 开发者开源的 Chat-甄嬛 数据集。我们的目标是构建一个能够模拟甄嬛对话风格的个性化 LLM，因此我们构造的指令形如：

```json
{
  "instruction": "陵容多承姐姐怜惜，才有了安身之地。此恩此德陵容无以为报。",
  "input": "",
  "output": "妹妹这样客气，倒叫我心里不安了。一路还顺利吗？"
}
```

* `instruction`：是用户指令，用于告知模型其所需要完成的任务
* `input`： 用户提出的问题或任务输入的内容，这里置为空
* `output`：模型根据`instruction`和`input`提供的回答或响应，在这里是对`instruction`对回答

我们所使用的全部指令数据集均在根目录文件夹 [/dataset](../dataset/huanhuan.json)下。

Lora 是一种高效微调方法，深入了解其原理可参见博客：[知乎|深入浅出 Lora](https://zhuanlan.zhihu.com/p/650197598)。

本篇文章的训练过程可见 SwanLab 记录：# TODO

同时，这个教程会在同目录下给大家提供一个 [notebook](./05-Qwen2.5-Coder-7B-Instruct%20Lora%20微调%20SwanLab%20可视化记录版.ipynb)文件，方便大家快速上手。

## 目录

- [SwanLab简介](#-SwanLab简介)
- [环境配置](#-环境配置)
- [准备数据集](#-准备数据集)
- [模型下载与加载](#-模型下载与加载)
- [集成SwanLab](#-集成SwanLab)
- [开始微调（完整代码）](#-开始微调)
- [训练结果演示](#-训练结果演示)

## SwanLab简介

![05-1](./images/05-1.jpg)

[SwanLab](https://github.com/swanhubx/swanlab) 是一个开源的模型训练记录工具，面向AI研究者，提供了训练可视化、自动日志记录、超参数记录、实验对比、多人协同等功能。在SwanLab上，研究者能基于直观的可视化图表发现训练问题，对比多个实验找到研究灵感，并通过在线链接的分享与基于组织的多人协同训练，打破团队沟通的壁垒。

**为什么要记录训练**

相较于软件开发，模型训练更像一个实验科学。一个品质优秀的模型背后，往往是成千上万次实验。研究者需要不断尝试、记录、对比，积累经验，才能找到最佳的模型结构、超参数与数据配比。在这之中，如何高效进行记录与对比，对于研究效率的提升至关重要。

## 环境配置

本文的基础环境如下：

```
----------------
ubuntu 22.04
python 3.12
cuda 12.1
pytorch 2.3.0
----------------
```

> **注意**：本文默认学习者已安装好以上环境

然后我们开始环境配置

```bash
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.18.0
pip install transformers==4.44.2
pip install accelerate==0.34.2
pip install peft==0.11.1
pip install datasets==2.20.0
pip install swanlab==0.3.23
```
## 模型下载与加载

## 集成 SwanLab

## 开始微调

## 训练结果展示