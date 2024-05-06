# CharacterGLM-6B
## 简介
CharacterGLM-6B 是聆心智能和清华大学 CoAI 实验室联合发布的新一代对话预训练模型。
本文件夹下属文件包含 CharacterGLM-6B 的部署、微调全流程。

## 01-CharacterGLM-6B Transformer部署调用
这个项目是关于如何使用CharacterGLM-6B模型进行对话生成的部署调用。通过该部署，可以轻松地在Autodl平台中租用一台具有足够显存的显卡机器，然后配置环境、下载模型并运行demo，以实现基于该模型的对话生成功能。
### 环境准备
1. 在Autodl平台中租用一台具有24G显存的显卡机器。
2. 打开JupyterLab并在其中的终端中进行环境配置、模型下载和运行demo。
### 模型下载
通过使用modelscope中的snapshot_download函数下载模型，您可以指定模型名称以及下载路径。该模型的下载路径设置为/root/autodl-tmp，并且模型大小为12 GB。
### 代码准备
在代码准备部分，您需要加载模型和分词器，并将模型移动到GPU上（如果可用）。然后，您可以通过使用模型的评估模式来生成对话。示例代码中包括了三轮对话的示例，展示了模型的使用方法。
### 部署
通过在终端中运行trans.py文件，即可实现CharacterGLM-6B模型的Transformers部署调用。在命令行中观察loading checkpoint表示模型正在加载，等待模型加载完成后即可开始产生对话。
通过这些步骤，您可以轻松地部署并使用CharacterGLM-6B模型进行对话生成。

## 02-CharacterGLM-6B FastApi部署调用
这个项目是关于如何使用CharacterGLM-6B模型进行对话生成的FastApi部署调用。通过该部署，可以轻松地在Autodl平台中租用一台具有足够显存的显卡机器，然后配置环境、下载模型并运行demo，以实现基于该模型的对话生成功能。
### 环境准备
1. 在Autodl平台中租用一台具有24G显存的显卡机器。
2. 打开JupyterLab并在其中的终端中进行环境配置、模型下载和运行demo。
### 模型下载
通过使用modelscope中的snapshot_download函数下载模型，您可以指定模型名称以及下载路径。该模型的下载路径设置为/root/autodl-tmp，并且模型大小为12 GB。
### 代码准备
在代码准备部分，您需要加载模型和分词器，并将模型移动到GPU上（如果可用）。
### 部署
通过在终端中运行api文件，即可实现CharacterGLM-6B模型的FastApi部署调用。在命令行中观察loading checkpoint表示模型正在加载，等待模型加载完成后即可开始产生对话。
通过这些步骤，您可以轻松地部署并使用CharacterGLM-6B模型进行对话生成。

## 03-CharacterGLM-6B-chat
本文件包含了使用 CharacterGLM-6B 模型进行类似于聊天机器人的推理的代码。CharacterGLM-6B 模型是一个经过微调的大型语言模型，用于在对话场景中生成文本回复。
### 环境设置
1. 在 autodl 平台上租用具有 24GB GPU 的服务器。
2. 打开租用的服务器上的 JupyterLab，并在终端中开始配置环境、下载模型和运行演示。
3. 更换 pip 源并安装所需的软件包。
### 模型下载
使用 modelscope 中的 `snapshot_download` 函数下载模型，并将其缓存在指定的路径中。
### 代码准备
克隆代码仓库，并根据需要修改代码路径以及配置文件。
### demo 运行
修改代码路径并运行演示，通过浏览器或命令行即可与模型进行交互。

## 04-CharacterGLM-6B Lora微调
本文简要介绍如何基于 transformers 和 peft 框架，对 CharacterGLM-6B-chat 模型进行 Lora 微调。Lora 原理可参考博客：[知乎|深入浅出Lora](https://zhuanlan.zhihu.com/p/650197598)
### 环境配置
在完成基本环境配置和本地模型部署的情况下，还需要安装一些第三方库。
### 指令集构建
LLM 微调一般指指令微调过程。
### QA和Instruction的区别和联系
QA 是指一问一答的形式，通常是用户提问，模型给出回答。而 instruction 则源自于 Prompt Engineering，将问题拆分成两个部分：Instruction 用于描述任务，Input 用于描述待处理的对象。
### 数据格式化
Lora 训练的数据是需要经过格式化、编码之后再输入给模型进行训练的。
### 加载 tokenizer 和半精度模型
模型以半精度形式加载。
### 定义 LoraConfig
LoraConfig 这个类中可以设置很多参数。
### 自定义 TraininArguments 参数
TrainingArguments 这个类的源码也介绍了每个参数的具体作用。
### 使用 Trainer 训练
把 model 放进去，把上面设置的参数放进去，数据集放进去，开始训练。
### 模型推理
### 重新加载
通过 PEFT 所微调的模型，都可以使用下面的方法进行重新加载，并推理。
