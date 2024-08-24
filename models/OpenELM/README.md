
<div align="center">
<h1>
  OpenELM大模型
</h1>
</div>

## 1. 模型简介


OpenELM 是苹果开发的一款先进语言模型，通过一种新的层级缩放策略优化每个Transformer层的参数分配，从而提升模型的效率和准确性。OpenELM还提供了一个开放的训练和推理框架，包含数据集、训练日志和检查点等资源，支持研究的可重复性和透明性。这个模型旨在推动语言模型领域的开放研究与创新

模型地址：https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca

项目地址：https://github.com/apple/corenet/tree/main/projects/openelm

官方报道：https://machinelearning.apple.com/research/openelm

论文链接：https://arxiv.org/abs/2404.14619

## 2. 模型下载
OpenELM提供了多种模型格式，下载链接如下表所示：

|          模型           |                                                                           下载链接                                                                           |
|:---------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------:|
| OpenELM-270M-Instruct |      [ModelScope](https://www.modelscope.cn/models/LLM-Research/OpenELM-270M) \| [HuggingFace](https://huggingface.co/apple/OpenELM-270M-Instruct)       |                                                                                                                   
| OpenELM-450M-Instruct | [ModelScope](https://www.modelscope.cn/models/LLM-Research/OpenELM-450M-Instruct)  \| [HuggingFace](https://huggingface.co/apple/OpenELM-450M-Instruct)  |
| OpenELM-1_1B-Instruct | [ModelScope](https://www.modelscope.cn/models/LLM-Research/OpenELM-1_1B-Instruct)   \| [HuggingFace](https://huggingface.co/apple/OpenELM-1_1B-Instruct) |
|  OpenELM-3B-Instruct  |   [ModelScope](https://www.modelscope.cn/models/LLM-Research/OpenELM-3B-Instruct)   \| [HuggingFace](https://huggingface.co/apple/OpenELM-3B-Instruct)   |


##  3. 教程简介
本教程以OpenELM-3B-Instruct为基础，介绍以下内容：

- 01-OpenELM-3B-Instruct FastApi 部署调用
- 02-OpenELM-3B-Instruct Lora 微调