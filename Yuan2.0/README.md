
<div align="center">
<h1>
  源2.0 大模型
</h1>
</div>

## 1. 模型简介

源2.0 是浪潮信息发布的新一代基础语言大模型，包括源2.0-102B，源2.0-51B和源2.0-2B。源2.0是在源1.0的基础上，利用更多样的高质量预训练数据和指令微调数据集，令模型在语义、数学、推理、代码、知识等不同方面具备更强的理解能力。

算法方面，源2.0提出并采用了一种新型的注意力算法结构：局部注意力过滤增强机制(LFA：Localized Filtering-based Attention)。LFA通过先学习相邻词之间的关联性，然后再计算全局关联性的方法，能够更好地学习到自然语言的局部和全局的语言特征，对于自然语言的关联语义理解更准确、更人性，提升了模型的自然语言表达能力，进而提升了模型精度。

<div align=center>
  <img src=images/yuan2.0-0.png >
  <p>Fig.1: 源2.0 架构和LFA</p>
</div>

<div align=center>
  <img src=images/yuan2.0-1.jpg >
  <p>Fig.2: 源2.0业界主流评测任务表现</p>
</div>


项目地址：https://github.com/IEIT-Yuan/Yuan-2.0

官方报道：https://mp.weixin.qq.com/s/rjnsUS83TT7aEN3r2i0IPQ

论文链接：https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/docs/Yuan2.0_paper.pdf

## 2. 模型下载
Yuan2.0提供了多种模型格式，下载链接如下表所示：

|                                 模型                                  | 序列长度  |                                                                                                                                                                                       下载链接                                                                                                                                                                                        |
|:-------------------------------------------------------------------:| :------: |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                            源2.0-102B-hf                             |    4K    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-102B-hf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-102B-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-102B-hf)  \|  [百度网盘](https://pan.baidu.com/s/1O4GkPSTPu5nwHk4v9byt7A?pwd=pq74#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-102B-hf) |
|                             源2.0-51B-hf                             |    4K    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-51B-hf/summary)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2.0-51B-hf)  \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-51B-hf)   \| [百度网盘](https://pan.baidu.com/s/1-qw30ZuyrMfraFtkLgDg0A?pwd=v2nd#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-51B-hf) |
|                             源2.0-2B-hf                              |    8K    |  [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-2B-hf/summary)   \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-2B-hf)   \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-hf)    \| [百度网盘](https://pan.baidu.com/s/1nt-03OAnjtZwhiVywj3xGw?pwd=nqef#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-hf)   |
|                          源2.0-2B-Janus-hf                           |    8K    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-2B-Janus-hf/files)   \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-2B-Janus-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-Janus-hf)  \| [百度网盘](https://pan.baidu.com/s/1f7l-rSVlYAij33htR51TEg?pwd=hkep ) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-Janus-hf) |
|                          源2.0-2B-Februa-hf                          |    8K    |  [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-2B-Februa-hf)   \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-2B-Februa-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-Februa-hf)  \| [百度网盘](https://pan.baidu.com/s/1f7l-rSVlYAij33htR51TEg?pwd=hkep ) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-Februa-hf)  |
| 源2.0-2B-Mars-hf <sup><font color="#FFFF00">*New*</font><br /></sup> |    8K    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-2B-Mars-hf)   \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-2B-Mars-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-2B-Mars-hf)  \| [百度网盘](https://pan.baidu.com/s/1f7l-rSVlYAij33htR51TEg?pwd=hkep ) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-2B-Mars-hf)      | 

注：源2.0-2B-Mars-hf、源2.0-2B-Februa-hf、源2.0-2B-Janus-hf均为源2.0-2B-hf的迭代版本。

##  3. 教程简介

在上一节介绍的多个模型中，源2.0-2B-Mars-hf是最新版本的2B参数模型，微调和部署源2.0-2B-Mars-hf对显存和硬盘的要求都相对较低。

因此，本教程以源2.0-2B-Mars-hf为基础，介绍以下内容：

- 01-Yuan2.0-2B FastApi 部署调用
- 02-Yuan2.0-2B Langchain 接入
- 03-Yuan2.0-2B WebDemo部署
- 04-Yuan2.0-2b vLLM部署调用
- 05-Yuan2.0-2B Lora微调