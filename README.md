# 开源大模型食用指南

## 项目简介

&emsp;&emsp;本项目是一个围绕开源大模型、针对国内初学者、基于 AutoDL 平台的中国宝宝专属大模型教程，针对各类开源大模型提供包括环境配置、本地部署、高效微调等技能在内的全流程指导，简化开源大模型的部署、使用和应用流程，让更多的普通学生、研究者更好地使用开源大模型，帮助开源、自由的大模型更快融入到普通学习者的生活中。

&emsp;&emsp;本项目的主要内容包括：

  1. 基于 AutoDL 平台（可扩展，例如阿里云）的开源 LLM 环境配置指南，针对不同模型要求提供不同的详细环境配置步骤；
  2. 针对国内外主流开源 LLM 的部署使用教程，包括 LLaMA、ChatGLM、InternLM 等； 
  3. 开源 LLM 的部署应用指导，包括命令行调用、在线 Demo 部署、LangChain 框架集成等；
  4. 开源 LLM 的全量微调、高效微调方法，包括分布式全量微调、LoRA、ptuning 等。

&emsp;&emsp;项目的主要内容就是 **教程**，让更多的学生和从业者了解和熟悉开源大模型的食用方法！任何人都可以提出issue或是提交PR，共同构建维护这个项目。

&emsp;&emsp;想要深度参与的同学可以联系我们，我们会将你加入到项目的维护者中。

## 项目意义

&emsp;&emsp;什么是大模型？

>大模型（LLM）狭义上指基于深度学习算法进行训练的自然语言处理（NLP）模型，主要应用于自然语言理解和生成等领域，广义上还包括机器视觉（CV）大模型、多模态大模型和科学计算大模型等。

&emsp;&emsp;百模大战正值火热，开源 LLM 层出不穷。如今国内外已经涌现了众多优秀开源 LLM，国外如 LLaMA、Alpaca，国内如 ChatGLM、BaiChuan、InternLM（书生·蒲语）等。开源 LLM 支持用户本地部署、私域微调，每一个人都可以在开源 LLM 的基础上打造专属于自己的独特大模型。

&emsp;&emsp;然而，当前普通学生和用户想要使用这些大模型，需要具备一定的技术能力，才能完成模型的部署和使用。对于层出不穷又各有特色的开源 LLM，想要快速掌握一个开源 LLM 的应用方法，是一项比较有挑战的任务。

&emsp;&emsp;本项目旨在首先基于核心贡献者的经验，实现国内外主流开源 LLM 的部署、使用与微调教程；在实现主流 LLM 的相关部分之后，我们希望充分聚集共创者，一起丰富这个开源 LLM 的世界，打造更多、更全面特色 LLM 的教程。星火点点，汇聚成海。

&emsp;&emsp;*我们希望成为 LLM 与普罗大众的阶梯，以自由、平等的开源精神，拥抱更恢弘而辽阔的 LLM 世界。*

## 项目受众

&emsp;&emsp;本项目适合以下学习者：

* 想要使用或体验 LLM，但无条件获得或使用相关 API；
* 希望长期、低成本、大量应用 LLM；
* 对开源 LLM 感兴趣，想要亲自上手开源 LLM；
* NLP 在学，希望进一步学习 LLM；
* 希望结合开源 LLM，打造领域特色的私域 LLM；
* 以及最广大、最普通的学生群体。

## 项目规划及进展

&emsp;&emsp; 本项目拟围绕开源 LLM 应用全流程组织，包括环境配置及使用、部署应用、微调等，每个部分覆盖主流及特点开源 LLM：

### 已支持模型

- InternLM
  - [ ] InternLM-Chat-7B Transformers 部署调用
  - [ ] InternLM-Chat-7B FastApi 部署调用
  - [x] [InternLM-Chat-7B WebDemo](InternLM/01-InternLM-Chat-7B.md) @不要葱姜蒜
  - [x] [Lagent+InternLM-Chat-7B-V1.1 WebDemo](InternLM/02-Lagent+InternLM-Chat-7B-V1.1.md) @不要葱姜蒜
  - [x] [浦语灵笔图文理解&创作 WebDemo](InternLM/03-浦语灵笔图文理解&创作.md) @不要葱姜蒜
  - [ ] InternLM-Chat-7B 接入 LangChain 框架 @ Logan Zou
  - [ ] InternLM-Chat-7B Lora 微调
  - [ ] InternLM-Chat-7B ptuning 微调
  - [ ] InternLM-Chat-7B 全量微调

- ChatGLM
  - [ ] ChatGLM3-6B Transformers 部署调用
  - [ ] ChatGLM3-6B FastApi 部署调用
  - [x] [ChatGLM3-6B chat WebDemo](ChatGLM/01-ChatGLM3-6B-chat.md) @不要葱姜蒜
  - [x] [ChatGLM3-6B Code Interpreter WebDemo](ChatGLM/02-ChatGLM3-6B-Code-Interpreter.md) @不要葱姜蒜
  - [ ] ChatGLM3-6B 接入 LangChain 框架 @ Logan Zou
  - [ ] ChatGLM3-6B Lora 微调 @ Logan Zou
  - [ ] ChatGLM3-6B ptuning 微调
  - [ ] ChatGLM3-6B 全量微调
- Qwen
  - [ ] Qwen-7B-chat Transformers 部署调用
  - [ ] Qwen-7B-chat FastApi 部署调用
  - [ ] Qwen-7B-chat WebDemo
  - [ ] Qwen-7B-chat Lora 微调
  - [ ] Qwen-7B-chat ptuning 微调
  - [ ] Qwen-7B-chat 全量微调
- Yi 
  - [ ] Yi-7B-base WebDemo (Yi 暂时没有chat模型)
- llama2
  - [ ] llama2-7B-chinese-chat WebDemo
  - [ ] llama2-7B-chinese-chat Lora 微调 @ Logan Zou
  - [ ] llama2-7B-chinese-chat 全量微调 @ Logan Zou
- 欢迎大家积极提出issue和PR

### LangChain

- [ ] LangChain 基于开源大模型构建问答链 @ Logan Zou
- [ ] LangChain 使用开源大模型构建 embedding @ Logan Zou
- [ ] 基于本地大模型构建应用 @ Logan Zou
- 欢迎大家提出issue和PR

### 通用环境配置

- [x] [pip、conda 换源](#pip、conda-换源) @不要葱姜蒜
- [x] AutoDL 开放端口 @不要葱姜蒜

- 模型下载
  - [x] hugging face @不要葱姜蒜
  - [x] hugging face 镜像下载 @不要葱姜蒜
  - [x] modelscope @不要葱姜蒜
  - [x] git-lfs @不要葱姜蒜
  - [ ] Openxlab

## 通用环境配置

### pip、conda 换源

更多详细内容可移步至[MirrorZ Help](https://help.mirrors.cernet.edu.cn/)查看。

#### pip 换源

临时使用镜像源安装，如下所示：`some-package` 为你需要安装的包名

```shell
pip install -i https://mirrors.cernet.edu.cn/pypi/web/simple some-package
```

设置pip默认镜像源，升级 pip 到最新的版本 (>=10.0.0) 后进行配置，如下所示：

```shell
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple
```

如果您的 pip 默认源的网络连接较差，临时使用镜像源升级 pip：

```shell
python -m pip install -i https://mirrors.cernet.edu.cn/pypi/web/simple --upgrade pip
```

#### conda 换源

镜像站提供了 Anaconda 仓库与第三方源（conda-forge、msys2、pytorch 等，各系统都可以通过修改用户目录下的 .condarc 文件来使用镜像站。

不同系统下的.condarc目录如下：

- Linux: ${HOME}/.condarc
- macOS: ${HOME}/.condarc
- Windows: C:\Users\<YourUserName>\.condarc

注意：

- Windows 用户无法直接创建名为 .condarc 的文件，可先执行 conda config --set show_channel_urls yes 生成该文件之后再修改。

快速配置

```shell
cat <<'EOF' > ~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
EOF
```

### AutoDL 开放端口

将 `autodl `的端口映射到本地的 [http://localhost:6006](http://localhost:6006/) 仅在此处展示一次，以下两个 Demo 都是同样的方法把 `autodl `中的 `6006 `端口映射到本机的 `http://localhost:6006`的方法都是相同的，方法如图所示。

![Alt text](images/image-4.png)

### 模型下载

#### hugging face

使用`huggingface`官方提供的`huggingface-cli`命令行工具。安装依赖:

```shell
pip install -U huggingface_hub
```

然后新建python文件，填入以下代码，运行即可。

- resume-download：断点续下
- local-dir：本地存储路径。（linux环境下需要填写绝对路径）

```python
import os

# 下载模型
os.system('huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path')
```

#### hugging face 镜像下载

与使用hugginge face下载相同，只需要填入镜像地址即可。使用`huggingface`官方提供的`huggingface-cli`命令行工具。安装依赖:

```shell
pip install -U huggingface_hub
```

然后新建python文件，填入以下代码，运行即可。

- resume-download：断点续下
- local-dir：本地存储路径。（linux环境下需要填写绝对路径）

```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
os.system('huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir your_path')
```

更多关于镜像使用可以移步至 [HF Mirror](https://hf-mirror.com/) 查看。

#### modelscope

使用`modelscope`中的`snapshot_download`函数下载模型，第一个参数为模型名称，参数`cache_dir`为模型的下载路径。

注意：`cache_dir`最好为决定路径。

安装依赖：
  
```shell
pip install modelscope
pip install transformers
```

在当前目录下新建python文件，填入以下代码，运行即可。

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='your path', revision='master')
```


#### git-lfs

来到[git-lfs](https://git-lfs.com/)网站下载安装包，然后安装`git-lfs`。安装好之后在终端输入`git lfs install`，然后就可以使用`git-lfs`下载模型了。当然这种方法需要你有一点点 **Magic** 。


```shell
git clone https://huggingface.co/internlm/internlm-7b
```

## 致谢

<div align=center style="margin-top: 20px;">
  <a href="https://github.com/datawhalechina/d2l-ai-solutions-manual/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=KMnO4-zx/llm-QuicklyDeploy" />
  </a>
</div>

<div align=center style="margin-top: 20px;">
  <a href="https://datawhale.club/#/">Datawhale</a>、
  <a href="https://www.shlab.org.cn/">上海人工智能实验室</a>
</div>