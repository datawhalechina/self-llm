
<div align="center">
<h1>
  源2.0 M32大模型
</h1>
</div>

##  1. 模型简介

浪潮信息 **“源2.0 M32”大模型（简称，Yuan2.0-M32）** 采用稀疏混合专家架构（MoE），以Yuan2.0-2B模型作为基底模型，通过创新的门控网络（Attention Router）实现32个专家间（Experts*32）的协同工作与任务调度，在显著降低模型推理算力需求的情况下，带来了更强的模型精度表现与推理性能；源2.0-M32在多个业界主流的评测进行了代码生成、数学问题求解、科学问答与综合知识能力等方面的能力测评。结果显示，源2.0-M32在多项任务评测中，展示出了较为先进的能力表现，MATH（数学求解）、ARC-C（科学问答）测试精度超过LLaMA3-700亿模型。

**Yuan2.0-M32大模型** 基本信息如下：

+ **模型参数量：** 40B <br>
+ **专家数量：** 32 <br>
+ **激活专家数：** 2 <br>
+ **激活参数量：** 3.7B <br>  
+ **训练数据量：** 2000B tokens <br>
+ **支持序列长度：** 16K <br>


<div align=center>
  <img src=images/yuan2.0-m32-0.jpg >
  <p>Fig.1: 源2.0-M32 架构图</p>
</div>

<div align=center>
  <img src=images/yuan2.0-m32-1.jpg >
  <p>Fig.2: 源2.0-M32业界主流评测任务表现</p>
</div>

项目地址：https://github.com/IEIT-Yuan/Yuan2.0-M32

官方报道：https://mp.weixin.qq.com/s/WEVyYq9BkTTlO6EAfiCf6w

技术报告：https://arxiv.org/abs/2405.17976


##  2. 模型下载
Yuan2.0-M32提供了多种模型格式，下载链接如下表所示：

|    模型     | 序列长度  |   模型格式   |         下载链接         |
| :----------: | :------: | :-------: |:---------------------------: |
| Yuan2.0-M32 |    16K    |    Megatron    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32/) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32) \| [百度网盘](https://pan.baidu.com/s/1K0LVU5NxeEujtYczF_T-Rg?pwd=cupw) \| [始智AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32)
| Yuan2.0-M32-HF |    16K    | HuggingFace    |    [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-hf) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf) \| [百度网盘](https://pan.baidu.com/s/1FrbVKji7IrhpwABYSIsV-A?pwd=q6uh)\| [始智AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-hf)
| Yuan2.0-M32-GGUF |    16K    | GGUF         |    [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-gguf/summary)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-gguf) \| [百度网盘](https://pan.baidu.com/s/1BWQaz-jeZ1Fe69CqYtjS9A?pwd=f4qc) \| [始智AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-gguf)
| Yuan2.0-M32-GGUF-INT4 |    16K    | GGUF    |    [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-gguf-int4/summary)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-gguf-int4) \| [百度网盘](https://pan.baidu.com/s/1FM8xPpkhOrRcAfe7-zUgWQ?pwd=e6ag) \| [始智AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-gguf-int4)
| Yuan2.0-M32-HF-INT4 |    16K    |  HuggingFace    |    [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-HF-INT4/summary)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf-int4) \| 百度网盘 \| [始智AI](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-hf-int4/)


##  3. 教程简介

在上一节介绍的多个模型中，Yuan2-M32-HF-INT4是由原始的Yuan2-M32-HF经过auto-gptq量化而来的模型。

通过模型量化，部署Yuan2-M32-HF-INT4对显存和硬盘的要求都会显著减低。

使用3090部署Yuan2-M32-HF-INT4的显存占用如下图所示：

<div align=center>
  <img src=images/gpu.png >
  <p>Fig.2: Yuan2-M32-HF-INT4 显存占用</p>
</div>


因此，本教程以Yuan2-M32-HF-INT4为基础，介绍以下内容：

- 01-Yuan2.0-M32 FastApi 部署调用
- 02-Yuan2.0-M32 Langchain 接入 
- 03-Yuan2.0-M32 WebDemo部署
