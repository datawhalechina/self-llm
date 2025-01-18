<div align=center>
  <img src="./images/head-img.png" >
  <h1>开源大模型食用指南</h1>
</div>

&emsp;&emsp;本项目是一个围绕开源大模型、针对国内初学者、基于 Linux 平台的中国宝宝专属大模型教程，针对各类开源大模型提供包括环境配置、本地部署、高效微调等技能在内的全流程指导，简化开源大模型的部署、使用和应用流程，让更多的普通学生、研究者更好地使用开源大模型，帮助开源、自由的大模型更快融入到普通学习者的生活中。

&emsp;&emsp;本项目的主要内容包括：

  1. 基于 Linux 平台的开源 LLM 环境配置指南，针对不同模型要求提供不同的详细环境配置步骤；
  2. 针对国内外主流开源 LLM 的部署使用教程，包括 LLaMA、ChatGLM、InternLM 等； 
  3. 开源 LLM 的部署应用指导，包括命令行调用、在线 Demo 部署、LangChain 框架集成等；
  4. 开源 LLM 的全量微调、高效微调方法，包括分布式全量微调、LoRA、ptuning 等。

&emsp;&emsp;**项目的主要内容就是教程，让更多的学生和未来的从业者了解和熟悉开源大模型的食用方法！任何人都可以提出issue或是提交PR，共同构建维护这个项目。**

&emsp;&emsp;想要深度参与的同学可以联系我们，我们会将你加入到项目的维护者中。

> &emsp;&emsp;***学习建议：本项目的学习建议是，先学习环境配置，然后再学习模型的部署使用，最后再学习微调。因为环境配置是基础，模型的部署使用是基础，微调是进阶。初学者可以选择Qwen1.5，InternLM2，MiniCPM等模型优先学习。***

> 注：如果有同学希望了解大模型的模型构成，以及从零手写RAG、Agent和Eval等任务，可以学习Datawhale的另一个项目[Tiny-Universe](https://github.com/datawhalechina/tiny-universe)，大模型是当下深度学习领域的热点，但现有的大部分大模型教程只在于教给大家如何调用api完成大模型的应用，而很少有人能够从原理层面讲清楚模型结构、RAG、Agent 以及 Eval。所以该仓库会提供全部手写，不采用调用api的形式，完成大模型的 RAG 、 Agent 、Eval 任务。

> 注：考虑到有同学希望在学习本项目之前，希望学习大模型的理论部分，如果想要进一步深入学习 LLM 的理论基础，并在理论的基础上进一步认识、应用 LLM，可以参考 Datawhale 的 [so-large-llm](https://github.com/datawhalechina/so-large-lm.git)课程。

> 注：如果有同学在学习本课程之后，想要自己动手开发大模型应用。同学们可以参考 Datawhale 的 [动手学大模型应用开发](https://github.com/datawhalechina/llm-universe) 课程，该项目是一个面向小白开发者的大模型应用开发教程，旨在基于阿里云服务器，结合个人知识库助手项目，向同学们完整的呈现大模型应用开发流程。

## 项目意义

&emsp;&emsp;什么是大模型？

>大模型（LLM）狭义上指基于深度学习算法进行训练的自然语言处理（NLP）模型，主要应用于自然语言理解和生成等领域，广义上还包括机器视觉（CV）大模型、多模态大模型和科学计算大模型等。

&emsp;&emsp;百模大战正值火热，开源 LLM 层出不穷。如今国内外已经涌现了众多优秀开源 LLM，国外如 LLaMA、Alpaca，国内如 ChatGLM、BaiChuan、InternLM（书生·浦语）等。开源 LLM 支持用户本地部署、私域微调，每一个人都可以在开源 LLM 的基础上打造专属于自己的独特大模型。

&emsp;&emsp;然而，当前普通学生和用户想要使用这些大模型，需要具备一定的技术能力，才能完成模型的部署和使用。对于层出不穷又各有特色的开源 LLM，想要快速掌握一个开源 LLM 的应用方法，是一项比较有挑战的任务。

&emsp;&emsp;本项目旨在首先基于核心贡献者的经验，实现国内外主流开源 LLM 的部署、使用与微调教程；在实现主流 LLM 的相关部分之后，我们希望充分聚集共创者，一起丰富这个开源 LLM 的世界，打造更多、更全面特色 LLM 的教程。星火点点，汇聚成海。

&emsp;&emsp;***我们希望成为 LLM 与普罗大众的阶梯，以自由、平等的开源精神，拥抱更恢弘而辽阔的 LLM 世界。***

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

### Example 系列

- [Chat-嬛嬛](./examples/Chat-嬛嬛/readme.md)： Chat-甄嬛是利用《甄嬛传》剧本中所有关于甄嬛的台词和语句，基于LLM进行LoRA微调得到的模仿甄嬛语气的聊天语言模型。

- [Tianji-天机](./examples/Tianji-天机/readme.md)：天机是一款基于人情世故社交场景，涵盖提示词工程 、智能体制作、 数据获取与模型微调、RAG 数据清洗与使用等全流程的大语言模型系统应用教程。

### 已支持模型


- [MiniCPM-o-2_6](https://github.com/OpenBMB/MiniCPM-o)
  - [ ] minicpm-o-2.6 FastApi 部署调用 @林恒宇
  - [ ] minicpm-o-2.6 WebDemo 部署 @九月
  - [ ] minicpm-o-2.6 vLLM 部署 @邓恺俊
  - [ ] minicpm-o-2.6 可视化 Lora 微调 @林泽毅

- [InternLM3](https://github.com/InternLM/InternLM)
  - [ ] internlm3-8b-instruct FastApi 部署调用 @苏向标
  - [ ] internlm3-8b-instruct Langchian接入 @赵文恺
  - [ ] internlm3-8b-instruct WebDemo 部署 @王泽宇
  - [ ] internlm3-8b-instruct vLLM 部署 @陈睿
  - [ ] internlm3-8b-instruct Lora 微调 @九月

- [phi4](https://huggingface.co/microsoft/phi-4)
  - [ ] phi4 FastApi 部署调用 @杜森
  - [x] [phi4 langchain 接入](./models/phi4/02-Phi-4-Langchain接入.md) @小罗
  - [ ] phi4 WebDemo 部署 @杜森
  - [x] [phi4 Lora 微调](./models/phi4/04-Phi-4-Lora%20微调.md) @郑远婧

- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder)
  - [x] [Qwen2.5-Coder-7B-Instruct FastApi部署调用](./models/Qwen2.5-Coder/01-Qwen2.5-Coder-7B-Instruct%20FastApi%20部署调用.md) @赵文恺
  - [x] [Qwen2.5-Coder-7B-Instruct Langchian接入](./models/Qwen2.5-Coder/02-Qwen2.5-7B-Instruct%20Langchain%20接入.md) @杨晨旭
  - [x] [Qwen2.5-Coder-7B-Instruct WebDemo 部署](./models/Qwen2.5-Coder/03-Qwen2.5-Coder-7B-Instruct%20WebDemo部署.md) @王泽宇
  - [x] [Qwen2.5-Coder-7B-Instruct vLLM 部署](./models/Qwen2.5-Coder/04-Qwen2.5-Coder-7B-Instruct%20vLLM%20部署调用.md) @王泽宇
  - [x] [Qwen2.5-Coder-7B-Instruct Lora 微调](./models/Qwen2.5-Coder/Qwen2.5-Coder-7B-Instruct%20Lora%20微调.md) @荞麦
  - [x] [Qwen2.5-Coder-7B-Instruct Lora 微调 SwanLab 可视化记录版](./models/Qwen2.5-Coder/05-Qwen2.5-Coder-7B-Instruct%20Lora%20微调%20SwanLab%20可视化记录版.md) @杨卓

- [Qwen2-vl](https://github.com/QwenLM/Qwen2-VL)
  - [x] [Qwen2-vl-2B FastApi 部署调用](./models/Qwen2-VL/01-Qwen2-VL-2B-Instruct%20FastApi%20部署调用.md) @姜舒凡
  - [x] [Qwen2-vl-2B WebDemo 部署](./models/Qwen2-VL/02-Qwen2-VL-2B-Instruct%20Web%20Demo部署.md) @赵伟
  - [x] [Qwen2-vl-2B vLLM 部署](./models/Qwen2-VL/03-Qwen2-VL-2B-Instruct%20vLLM部署调用.md) @荞麦
  - [x] [Qwen2-vl-2B Lora 微调](./models/Qwen2-VL/04-Qwen2-VL-2B%20Lora%20微调.md) @李柯辰
  - [x] [Qwen2-vl-2B Lora 微调 SwanLab 可视化记录版](./models/Qwen2-VL/05-Qwen2-VL-2B-Instruct%20Lora%20微调%20SwanLab%20可视化记录版.md) @林泽毅
  - [x] [Qwen2-vl-2B Lora 微调案例 - LaTexOCR](./models/Qwen2-VL/06-Qwen2-VL-2B-Instruct%20Lora%20微调案例%20-%20LaTexOCR.md) @林泽毅

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)
  - [x] [Qwen2.5-7B-Instruct FastApi 部署调用](./models/Qwen2.5/01-Qwen2.5-7B-Instruct%20FastApi%20部署调用.md) @娄天奥
  - [x] [Qwen2.5-7B-Instruct langchain 接入](./models/Qwen2.5/02-Qwen2.5-7B-Instruct%20Langchain%20接入.md) @娄天奥
  - [x] [Qwen2.5-7B-Instruct vLLM 部署调用](./models/Qwen2.5/03-Qwen2.5-7B-Instruct%20vLLM%20部署调用.md) @姜舒凡
  - [x] [Qwen2.5-7B-Instruct WebDemo 部署](./models/Qwen2.5/04-Qwen2_5-7B-Instruct%20WebDemo部署.md) @高立业
  - [x] [Qwen2.5-7B-Instruct Lora 微调](./models/Qwen2.5/05-Qwen2.5-7B-Instruct%20Lora%20微调.md) @左春生
  - [x] [Qwen2.5-7B-Instruct o1-like 推理链实现](./models/Qwen2.5/06-Qwen2.5-7B-Instruct%20o1-like%20推理链实现.md) @姜舒凡
  - [x] [Qwen2.5-7B-Instruct Lora 微调 SwanLab 可视化记录版](./models/Qwen2.5/07-Qwen2.5-7B-Instruct%20Lora%20微调%20SwanLab可视化记录版.md) @林泽毅

- [Apple OpenELM](https://machinelearning.apple.com/research/openelm)
  - [x] [OpenELM-3B-Instruct FastApi 部署调用](./models/OpenELM/01-OpenELM-3B-Instruct%20FastApi部署调用.md) @王泽宇
  - [x] [OpenELM-3B-Instruct Lora 微调](./models/OpenELM/02-OpenELM-3B-Instruct%20Lora微调.md) @王泽宇

- [Llama3_1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  - [x] [Llama3_1-8B-Instruct FastApi 部署调用](./models/Llama3_1/01-Llama3_1-8B-Instruct%20FastApi%20部署调用.md) @不要葱姜蒜
  - [x] [Llama3_1-8B-Instruct langchain 接入](./models/Llama3_1/02-Llama3_1-8B-Instruct%20langchain接入.md) @张晋
  - [x] [Llama3_1-8B-Instruct WebDemo 部署](./models/Llama3_1/03-Llama3_1-8B-Instruct%20WebDemo部署.md) @张晋
  - [x] [Llama3_1-8B-Instruct Lora 微调](./models/Llama3_1/04-Llama3_1-8B--Instruct%20Lora%20微调.md) @不要葱姜蒜

- [Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
  - [x] [Gemma-2-9b-it FastApi 部署调用](./models/Gemma2/01-Gemma-2-9b-it%20FastApi%20部署调用.md) @不要葱姜蒜
  - [x] [Gemma-2-9b-it langchain 接入](./models/Gemma2/02-Gemma-2-9b-it%20langchain%20接入.md) @不要葱姜蒜
  - [x] [Gemma-2-9b-it WebDemo 部署](./models/Gemma2/03-Gemma-2-9b-it%20WebDemo%20部署.md) @不要葱姜蒜
  - [x] [Gemma-2-9b-it Peft Lora 微调](./models/Gemma2/04-Gemma-2-9b-it%20peft%20lora微调.md) @不要葱姜蒜

- [Yuan2.0](https://github.com/IEIT-Yuan/Yuan-2.0)
  - [x] [Yuan2.0-2B FastApi 部署调用](./models/Yuan2.0/01-Yuan2.0-2B%20FastApi%20部署调用.md) @张帆
  - [x] [Yuan2.0-2B Langchain 接入](./models/Yuan2.0/02-Yuan2.0-2B%20Langchain%20接入.md) @张帆
  - [x] [Yuan2.0-2B WebDemo部署](./models/Yuan2.0/03-Yuan2.0-2B%20WebDemo部署.md) @张帆
  - [x] [Yuan2.0-2B vLLM部署调用](./models/Yuan2.0/04-Yuan2.0-2B%20vLLM部署调用.md) @张帆
  - [x] [Yuan2.0-2B Lora微调](./models/Yuan2.0/05-Yuan2.0-2B%20Lora微调.md) @张帆

- [Yuan2.0-M32](https://github.com/IEIT-Yuan/Yuan2.0-M32)
  - [x] [Yuan2.0-M32 FastApi 部署调用](./models/Yuan2.0-M32/01-Yuan2.0-M32%20FastApi%20部署调用.md) @张帆
  - [x] [Yuan2.0-M32 Langchain 接入](./models/Yuan2.0-M32/02-Yuan2.0-M32%20Langchain%20接入.md) @张帆
  - [x] [Yuan2.0-M32 WebDemo部署](./models/Yuan2.0-M32/03-Yuan2.0-M32%20WebDemo部署.md) @张帆

- [DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2)
  - [x] [DeepSeek-Coder-V2-Lite-Instruct FastApi 部署调用](./models/DeepSeek-Coder-V2/01-DeepSeek-Coder-V2-Lite-Instruct%20FastApi%20部署调用.md) @姜舒凡
  - [x] [DeepSeek-Coder-V2-Lite-Instruct langchain 接入](./models/DeepSeek-Coder-V2/02-DeepSeek-Coder-V2-Lite-Instruct%20接入%20LangChain.md) @姜舒凡
  - [x] [DeepSeek-Coder-V2-Lite-Instruct WebDemo 部署](./models/DeepSeek-Coder-V2/03-DeepSeek-Coder-V2-Lite-Instruct%20WebDemo%20部署.md) @Kailigithub
  - [x] [DeepSeek-Coder-V2-Lite-Instruct Lora 微调](./models/DeepSeek-Coder-V2/04-DeepSeek-Coder-V2-Lite-Instruct%20Lora%20微调.md) @余洋

- [哔哩哔哩 Index-1.9B](https://github.com/bilibili/Index-1.9B)
  - [x] [Index-1.9B-Chat FastApi 部署调用](./models/bilibili_Index-1.9B/01-Index-1.9B-chat%20FastApi%20部署调用.md) @邓恺俊
  - [x] [Index-1.9B-Chat langchain 接入](./models/bilibili_Index-1.9B/02-Index-1.9B-Chat%20接入%20LangChain.md) @张友东
  - [x] [Index-1.9B-Chat WebDemo 部署](./models/bilibili_Index-1.9B/03-Index-1.9B-chat%20WebDemo部署.md) @九月
  - [x] [Index-1.9B-Chat Lora 微调](./models/bilibili_Index-1.9B/04-Index-1.9B-Chat%20Lora%20微调.md) @姜舒凡

- [Qwen2](https://github.com/QwenLM/Qwen2)
  - [x] [Qwen2-7B-Instruct FastApi 部署调用](./models/Qwen2/01-Qwen2-7B-Instruct%20FastApi%20部署调用.md) @康婧淇
  - [x] [Qwen2-7B-Instruct langchain 接入](./models/Qwen2/02-Qwen2-7B-Instruct%20Langchain%20接入.md) @不要葱姜蒜
  - [x] [Qwen2-7B-Instruct WebDemo 部署](./models/Qwen2/03-Qwen2-7B-Instruct%20WebDemo部署.md) @三水
  - [x] [Qwen2-7B-Instruct vLLM 部署调用](./models/Qwen2/04-Qwen2-7B-Instruct%20vLLM%20部署调用.md) @姜舒凡
  - [x] [Qwen2-7B-Instruct Lora 微调](./models/Qwen2/05-Qwen2-7B-Instruct%20Lora%20微调.md) @散步

- [GLM-4](https://github.com/THUDM/GLM-4.git)
  - [x] [GLM-4-9B-chat FastApi 部署调用](./models/GLM-4/01-GLM-4-9B-chat%20FastApi%20部署调用.md) @张友东
  - [x] [GLM-4-9B-chat langchain 接入](./models/GLM-4/02-GLM-4-9B-chat%20langchain%20接入.md) @谭逸珂
  - [x] [GLM-4-9B-chat WebDemo 部署](./models/GLM-4/03-GLM-4-9B-Chat%20WebDemo.md) @何至轩
  - [x] [GLM-4-9B-chat vLLM 部署](./models/GLM-4/04-GLM-4-9B-Chat%20vLLM%20部署调用.md) @王熠明
  - [x] [GLM-4-9B-chat Lora 微调](./models/GLM-4/05-GLM-4-9B-chat%20Lora%20微调.md) @肖鸿儒
  - [x] [GLM-4-9B-chat-hf Lora 微调](./models/GLM-4/05-GLM-4-9B-chat-hf%20Lora%20微调.md) @付志远


- [Qwen 1.5](https://github.com/QwenLM/Qwen1.5.git)
  - [x] [Qwen1.5-7B-chat FastApi 部署调用](./models/Qwen1.5/01-Qwen1.5-7B-Chat%20FastApi%20部署调用.md) @颜鑫
  - [x] [Qwen1.5-7B-chat langchain 接入](./models/Qwen1.5/02-Qwen1.5-7B-Chat%20接入langchain搭建知识库助手.md) @颜鑫
  - [x] [Qwen1.5-7B-chat WebDemo 部署](./models/Qwen1.5/03-Qwen1.5-7B-Chat%20WebDemo.md) @颜鑫
  - [x] [Qwen1.5-7B-chat Lora 微调](./models/Qwen1.5/04-Qwen1.5-7B-chat%20Lora%20微调.md) @不要葱姜蒜
  - [x] [Qwen1.5-72B-chat-GPTQ-Int4 部署环境](./models/Qwen1.5/05-Qwen1.5-7B-Chat-GPTQ-Int4%20%20WebDemo.md) @byx020119
  - [x] [Qwen1.5-MoE-chat Transformers 部署调用](./models/Qwen1.5/06-Qwen1.5-MoE-A2.7B.md) @丁悦
  - [x] [Qwen1.5-7B-chat vLLM推理部署](./models/Qwen1.5/07-Qwen1.5-7B-Chat%20vLLM%20推理部署调用.md) @高立业
  - [x] [Qwen1.5-7B-chat Lora 微调 接入SwanLab实验管理平台](./models/Qwen1.5/08-Qwen1.5-7B-chat%20LoRA微调接入实验管理.md) @黄柏特

- [谷歌-Gemma](https://huggingface.co/google/gemma-7b-it)
  - [x] [gemma-2b-it FastApi 部署调用 ](./models/Gemma/01-Gemma-2B-Instruct%20FastApi%20部署调用.md) @东东
  - [x] [gemma-2b-it langchain 接入 ](./models/Gemma/02-Gemma-2B-Instruct%20langchain%20接入.md) @东东
  - [x] [gemma-2b-it WebDemo 部署 ](./models/Gemma/03-Gemma-2B-Instruct%20WebDemo%20部署.md) @东东
  - [x] [gemma-2b-it Peft Lora 微调 ](./models/Gemma/04-Gemma-2B-Instruct%20Lora微调.md) @东东

- [phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
  - [x] [Phi-3-mini-4k-instruct FastApi 部署调用](./models/phi-3/01-Phi-3-mini-4k-instruct%20FastApi%20部署调用.md) @郑皓桦
  - [x] [Phi-3-mini-4k-instruct langchain 接入](./models/phi-3/02-Phi-3-mini-4k-instruct%20langchain%20接入.md) @郑皓桦
  - [x] [Phi-3-mini-4k-instruct WebDemo 部署](./models/phi-3/03-Phi-3-mini-4k-instruct%20WebDemo部署.md) @丁悦
  - [x] [Phi-3-mini-4k-instruct Lora 微调](./models/phi-3/04-Phi-3-mini-4k-Instruct%20Lora%20微调.md) @丁悦

- [CharacterGLM-6B](https://github.com/thu-coai/CharacterGLM-6B)
  - [x] [CharacterGLM-6B Transformers 部署调用](./models/CharacterGLM/01-CharacterGLM-6B%20Transformer部署调用.md) @孙健壮
  - [x] [CharacterGLM-6B FastApi 部署调用](./models/CharacterGLM/02-CharacterGLM-6B%20FastApi部署调用.md) @孙健壮
  - [x] [CharacterGLM-6B webdemo 部署](./models/CharacterGLM/03-CharacterGLM-6B-chat.md) @孙健壮
  - [x] [CharacterGLM-6B Lora 微调](./models/CharacterGLM/04-CharacterGLM-6B%20Lora微调.md) @孙健壮

- [LLaMA3-8B-Instruct](https://github.com/meta-llama/llama3.git)
  - [x] [LLaMA3-8B-Instruct FastApi 部署调用](./models/LLaMA3/01-LLaMA3-8B-Instruct%20FastApi%20部署调用.md) @高立业
  - [X] [LLaMA3-8B-Instruct langchain 接入](./models/LLaMA3/02-LLaMA3-8B-Instruct%20langchain%20接入.md) @不要葱姜蒜
  - [x] [LLaMA3-8B-Instruct WebDemo 部署](./models/LLaMA3/03-LLaMA3-8B-Instruct%20WebDemo%20部署.md) @不要葱姜蒜
  - [x] [LLaMA3-8B-Instruct Lora 微调](./models/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20微调.md) @高立业

- [XVERSE-7B-Chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary)
  - [x] [XVERSE-7B-Chat transformers 部署调用](./models/XVERSE/01-XVERSE-7B-chat%20Transformers推理.md) @郭志航
  - [x] [XVERSE-7B-Chat FastApi 部署调用](./models/XVERSE/02-XVERSE-7B-chat%20FastAPI部署.md) @郭志航
  - [x] [XVERSE-7B-Chat langchain 接入](./models/XVERSE/03-XVERSE-7B-chat%20langchain%20接入.md) @郭志航
  - [x] [XVERSE-7B-Chat WebDemo 部署](./models/XVERSE/04-XVERSE-7B-chat%20WebDemo%20部署.md) @郭志航
  - [x] [XVERSE-7B-Chat Lora 微调](./models/XVERSE/05-XVERSE-7B-Chat%20Lora%20微调.md) @郭志航

- [TransNormerLLM](https://github.com/OpenNLPLab/TransnormerLLM.git)
  - [X] [TransNormerLLM-7B-Chat FastApi 部署调用](./models/TransNormer/01-TransNormer-7B%20FastApi%20部署调用.md) @王茂霖
  - [X] [TransNormerLLM-7B-Chat langchain 接入](./models/TransNormer/02-TransNormer-7B%20接入langchain搭建知识库助手.md) @王茂霖
  - [X] [TransNormerLLM-7B-Chat WebDemo 部署](./models/TransNormer/03-TransNormer-7B%20WebDemo.md) @王茂霖
  - [x] [TransNormerLLM-7B-Chat Lora 微调](./models/TransNormer/04-TrasnNormer-7B%20Lora%20微调.md) @王茂霖

- [BlueLM Vivo 蓝心大模型](https://github.com/vivo-ai-lab/BlueLM.git)
  - [x] [BlueLM-7B-Chat FatApi 部署调用](./models/BlueLM/01-BlueLM-7B-Chat%20FastApi%20部署.md) @郭志航
  - [x] [BlueLM-7B-Chat langchain 接入](./models/BlueLM/02-BlueLM-7B-Chat%20langchain%20接入.md) @郭志航
  - [x] [BlueLM-7B-Chat WebDemo 部署](./models/BlueLM/03-BlueLM-7B-Chat%20WebDemo%20部署.md) @郭志航
  - [x] [BlueLM-7B-Chat Lora 微调](./models/BlueLM/04-BlueLM-7B-Chat%20Lora%20微调.md) @郭志航

- [InternLM2](https://github.com/InternLM/InternLM)
  - [x] [InternLM2-7B-chat FastApi 部署调用](./models/InternLM2/01-InternLM2-7B-chat%20FastAPI部署.md) @不要葱姜蒜
  - [x] [InternLM2-7B-chat langchain 接入](./models/InternLM2/02-InternLM2-7B-chat%20langchain%20接入.md) @不要葱姜蒜
  - [x] [InternLM2-7B-chat WebDemo 部署](./models/InternLM2/03-InternLM2-7B-chat%20WebDemo%20部署.md) @郑皓桦
  - [x] [InternLM2-7B-chat Xtuner Qlora 微调](./models/InternLM2/04-InternLM2-7B-chat%20Xtuner%20Qlora%20微调.md) @郑皓桦

- [DeepSeek 深度求索](https://github.com/deepseek-ai/DeepSeek-LLM)
  - [x] [DeepSeek-7B-chat FastApi 部署调用](./models/DeepSeek/01-DeepSeek-7B-chat%20FastApi.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat langchain 接入](./models/DeepSeek/02-DeepSeek-7B-chat%20langchain.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat WebDemo](./models/DeepSeek/03-DeepSeek-7B-chat%20WebDemo.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat Lora 微调](./models/DeepSeek/04-DeepSeek-7B-chat%20Lora%20微调.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat 4bits量化 Qlora 微调](./models/DeepSeek/05-DeepSeek-7B-chat%204bits量化%20Qlora%20微调.md) @不要葱姜蒜
  - [x] [DeepSeek-MoE-16b-chat Transformers 部署调用](./models/DeepSeek/06-DeepSeek-MoE-16b-chat%20Transformer部署调用.md) @Kailigithub
  - [x] [DeepSeek-MoE-16b-chat FastApi 部署调用](./models/DeepSeek/06-DeepSeek-MoE-16b-chat%20FastApi.md) @Kailigithub
  - [x] [DeepSeek-coder-6.7b finetune colab](./models/DeepSeek/07-deepseek_fine_tune.ipynb) @Swiftie
  - [x] [Deepseek-coder-6.7b webdemo colab](./models/DeepSeek/08-deepseek_web_demo.ipynb) @Swiftie

- [MiniCPM](https://github.com/OpenBMB/MiniCPM.git)
  - [x] [MiniCPM-2B-chat transformers 部署调用](./models/MiniCPM/MiniCPM-2B-chat%20transformers%20部署调用.md) @Kailigithub 
  - [x] [MiniCPM-2B-chat FastApi 部署调用](./models/MiniCPM/MiniCPM-2B-chat%20FastApi%20部署调用.md) @Kailigithub 
  - [x] [MiniCPM-2B-chat langchain 接入](./models/MiniCPM/MiniCPM-2B-chat%20langchain接入.md) @不要葱姜蒜 
  - [x] [MiniCPM-2B-chat webdemo 部署](./models/MiniCPM/MiniCPM-2B-chat%20WebDemo部署.md) @Kailigithub 
  - [x] [MiniCPM-2B-chat Lora && Full 微调](./models/MiniCPM/MiniCPM-2B-chat%20Lora%20&&%20Full%20微调.md) @不要葱姜蒜 
  - [x] 官方友情链接：[面壁小钢炮MiniCPM教程](https://modelbest.feishu.cn/wiki/D2tFw8Pcsi5CIzkaHNacLK64npg) @OpenBMB 
  - [x] 官方友情链接：[MiniCPM-Cookbook](https://github.com/OpenBMB/MiniCPM-CookBook) @OpenBMB

- [Qwen-Audio](https://github.com/QwenLM/Qwen-Audio.git)
  - [x] [Qwen-Audio FastApi 部署调用](./models/Qwen-Audio/01-Qwen-Audio-chat%20FastApi.md) @陈思州
  - [x] [Qwen-Audio WebDemo](./models/Qwen-Audio/02-Qwen-Audio-chat%20WebDemo.md) @陈思州

- [Qwen](https://github.com/QwenLM/Qwen.git)
  - [x] [Qwen-7B-chat Transformers 部署调用](./models/Qwen/01-Qwen-7B-Chat%20Transformers部署调用.md) @李娇娇
  - [x] [Qwen-7B-chat FastApi 部署调用](./models/Qwen/02-Qwen-7B-Chat%20FastApi%20部署调用.md) @李娇娇
  - [x] [Qwen-7B-chat WebDemo](./models/Qwen/03-Qwen-7B-Chat%20WebDemo.md) @李娇娇
  - [x] [Qwen-7B-chat Lora 微调](./models/Qwen/04-Qwen-7B-Chat%20Lora%20微调.md) @不要葱姜蒜
  - [x] [Qwen-7B-chat ptuning 微调](./models/Qwen/05-Qwen-7B-Chat%20Ptuning%20微调.md) @肖鸿儒
  - [x] [Qwen-7B-chat 全量微调](./models/Qwen/06-Qwen-7B-chat%20全量微调.md) @不要葱姜蒜
  - [x] [Qwen-7B-Chat 接入langchain搭建知识库助手](./models/Qwen/07-Qwen-7B-Chat%20接入langchain搭建知识库助手.md) @李娇娇
  - [x] [Qwen-7B-chat 低精度训练](./models/Qwen/08-Qwen-7B-Chat%20Lora%20低精度微调.md) @肖鸿儒
  - [x] [Qwen-1_8B-chat CPU 部署](./models/Qwen/09-Qwen-1_8B-chat%20CPU%20部署%20.md) @散步

- [Yi 零一万物](https://github.com/01-ai/Yi.git)
  - [x] [Yi-6B-chat FastApi 部署调用](./models/Yi/01-Yi-6B-Chat%20FastApi%20部署调用.md) @李柯辰
  - [x] [Yi-6B-chat langchain接入](./models/Yi/02-Yi-6B-Chat%20接入langchain搭建知识库助手.md) @李柯辰
  - [x] [Yi-6B-chat WebDemo](./models/Yi/03-Yi-6B-chat%20WebDemo.md) @肖鸿儒
  - [x] [Yi-6B-chat Lora 微调](./models/Yi/04-Yi-6B-Chat%20Lora%20微调.md) @李娇娇

- [Baichuan 百川智能](https://www.baichuan-ai.com/home)
  - [x] [Baichuan2-7B-chat FastApi 部署调用](./BaiChuan/01-Baichuan2-7B-chat%2BFastApi%2B%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md) @惠佳豪
  - [x] [Baichuan2-7B-chat WebDemo](./models/BaiChuan/02-Baichuan-7B-chat%2BWebDemo.md) @惠佳豪
  - [x] [Baichuan2-7B-chat 接入 LangChain 框架](./models/BaiChuan/03-Baichuan2-7B-chat%E6%8E%A5%E5%85%A5LangChain%E6%A1%86%E6%9E%B6.md) @惠佳豪
  - [x] [Baichuan2-7B-chat Lora 微调](./models/BaiChuan/04-Baichuan2-7B-chat%2Blora%2B%E5%BE%AE%E8%B0%83.md) @惠佳豪

- [InternLM](https://github.com/InternLM/InternLM.git)
  - [x] [InternLM-Chat-7B Transformers 部署调用](./models/InternLM/01-InternLM-Chat-7B%20Transformers%20部署调用.md) @小罗
  - [x] [InternLM-Chat-7B FastApi 部署调用](./models/InternLM/02-internLM-Chat-7B%20FastApi.md) @不要葱姜蒜
  - [x] [InternLM-Chat-7B WebDemo](./models/InternLM/03-InternLM-Chat-7B.md) @不要葱姜蒜
  - [x] [Lagent+InternLM-Chat-7B-V1.1 WebDemo](./models/InternLM/04-Lagent+InternLM-Chat-7B-V1.1.md) @不要葱姜蒜
  - [x] [浦语灵笔图文理解&创作 WebDemo](./models/InternLM/05-浦语灵笔图文理解&创作.md) @不要葱姜蒜
  - [x] [InternLM-Chat-7B 接入 LangChain 框架](./models/InternLM/06-InternLM接入LangChain搭建知识库助手.md) @Logan Zou

- [Atom (llama2)](https://hf-mirror.com/FlagAlpha/Atom-7B-Chat)
  - [x] [Atom-7B-chat WebDemo](./models/Atom/01-Atom-7B-chat-WebDemo.md) @Kailigithub
  - [x] [Atom-7B-chat Lora 微调](./models/Atom/02-Atom-7B-Chat%20Lora%20微调.md) @Logan Zou
  - [x] [Atom-7B-Chat 接入langchain搭建知识库助手](./models/Atom/03-Atom-7B-Chat%20接入langchain搭建知识库助手.md) @陈思州
  - [x] [Atom-7B-chat 全量微调](./models/Atom/04-Atom-7B-chat%20全量微调.md) @Logan Zou

- [ChatGLM3](https://github.com/THUDM/ChatGLM3.git)
  - [x] [ChatGLM3-6B Transformers 部署调用](./models/ChatGLM/01-ChatGLM3-6B%20Transformer部署调用.md) @丁悦
  - [x] [ChatGLM3-6B FastApi 部署调用](./models/ChatGLM/02-ChatGLM3-6B%20FastApi部署调用.md) @丁悦
  - [x] [ChatGLM3-6B chat WebDemo](./models/ChatGLM/03-ChatGLM3-6B-chat.md) @不要葱姜蒜
  - [x] [ChatGLM3-6B Code Interpreter WebDemo](./models/ChatGLM/04-ChatGLM3-6B-Code-Interpreter.md) @不要葱姜蒜
  - [x] [ChatGLM3-6B 接入 LangChain 框架](./models/ChatGLM/05-ChatGLM3-6B接入LangChain搭建知识库助手.md) @Logan Zou
  - [x] [ChatGLM3-6B Lora 微调](./models/ChatGLM/06-ChatGLM3-6B-Lora微调.md) @肖鸿儒

### 通用环境配置

- [x] [pip、conda 换源](./models/General-Setting/01-pip、conda换源.md) @不要葱姜蒜
- [x] [AutoDL 开放端口](./models/General-Setting/02-AutoDL开放端口.md) @不要葱姜蒜

- 模型下载
  - [x] [hugging face](./models/General-Setting/03-模型下载.md) @不要葱姜蒜
  - [x] [hugging face](./General-Setting/03-模型下载.md) 镜像下载 @不要葱姜蒜
  - [x] [modelscope](./models/General-Setting/03-模型下载.md) @不要葱姜蒜
  - [x] [git-lfs](./models/General-Setting/03-模型下载.md) @不要葱姜蒜
  - [x] [Openxlab](./models/General-Setting/03-模型下载.md)
- Issue && PR
  - [x] [Issue 提交](./models/General-Setting/04-Issue&PR&update.md) @肖鸿儒
  - [x] [PR 提交](./models/General-Setting/04-Issue&PR&update.md) @肖鸿儒
  - [x] [fork更新](./models/General-Setting/04-Issue&PR&update.md) @肖鸿儒

## 致谢

### 核心贡献者

- [宋志学(不要葱姜蒜)-项目负责人](https://github.com/KMnO4-zx) （Datawhale成员-中国矿业大学(北京)）
- [邹雨衡-项目负责人](https://github.com/logan-zou) （Datawhale成员-对外经济贸易大学）
- [肖鸿儒](https://github.com/Hongru0306) （Datawhale成员-同济大学）
- [郭志航](https://github.com/acwwt)（内容创作者）
- [张帆](https://github.com/zhangfanTJU)（内容创作者-Datawhale成员）
- [姜舒凡](https://github.com/Tsumugii24)（内容创作者-鲸英助教）
- [李娇娇](https://github.com/Aphasia0515) （Datawhale成员）
- [丁悦](https://github.com/dingyue772) （Datawhale-鲸英助教）
- [林泽毅](https://github.com/Zeyi-Lin)（内容创作者-SwanLab产品负责人）
- [惠佳豪](https://github.com/L4HeyXiao) （Datawhale-宣传大使）
- [王茂霖](https://github.com/mlw67)（内容创作者-Datawhale成员）
- [孙健壮](https://github.com/Caleb-Sun-jz)（内容创作者-对外经济贸易大学）
- [东东](https://github.com/LucaChen)（内容创作者-谷歌开发者机器学习技术专家）
- [高立业](https://github.com/0-yy-0)（内容创作者-DataWhale成员）
- [王泽宇](https://github.com/moyitech)（内容创作者-太原理工大学-鲸英助教）
- [Kailigithub](https://github.com/Kailigithub) （Datawhale成员）
- [郑皓桦](https://github.com/BaiYu96) （内容创作者）
- [李柯辰](https://github.com/Joe-2002) （Datawhale成员）
- [陈思州](https://github.com/jjyaoao) （Datawhale成员）
- [散步](https://github.com/sanbuphy) （Datawhale成员）
- [颜鑫](https://github.com/thomas-yanxin) （Datawhale成员）
- [荞麦](https://github.com/yeyeyeyeeeee)（内容创作者-Datawhale成员）
- [Swiftie](https://github.com/cswangxiaowei) （小米NLP算法工程师）
- [黄柏特](https://github.com/KashiwaByte)（内容创作者-西安电子科技大学）
- [张友东](https://github.com/AXYZdong)（内容创作者-Datawhale成员）
- [余洋](https://github.com/YangYu-NUAA)（内容创作者-Datawhale成员）
- [张晋](https://github.com/Jin-Zhang-Yaoguang)（内容创作者-Datawhale成员）
- [娄天奥](https://github.com/lta155)（内容创作者-中国科学院大学-鲸英助教）
- [左春生](https://github.com/LinChentang)（内容创作者-Datawhale成员）
- [杨卓](https://github.com/little1d)（内容创作者-西安电子科技大学-鲸英助教）
- [小罗](https://github.com/lyj11111111) （内容创作者-Datawhale成员）
- [谭逸珂](https://github.com/LikeGiver)（内容创作者-对外经济贸易大学）
- [王熠明](https://github.com/Bald0Wang)（内容创作者-Datawhale成员）
- [何至轩](https://github.com/pod2c)（内容创作者-鲸英助教）
- [康婧淇](https://github.com/jodie-kang)（内容创作者-Datawhale成员）
- [三水](https://github.com/sssanssss)（内容创作者-鲸英助教）
- [九月](https://github.com/chg0901)（内容创作者-Datawhale意向成员）
- [邓恺俊](https://github.com/Kedreamix)（内容创作者-Datawhale成员）
- [杨晨旭](https://github.com/langlibai66)（内容创作者-太原理工大学-鲸英助教）
- [赵文恺](https://github.com/XiLinky)（内容创作者-太原理工大学-鲸英助教）
- [赵伟](https://github.com/2710932616)（内容创作者-鲸英助教）
- [付志远](https://github.com/comfzy)（内容创作者-海南大学）

> 注：排名根据贡献程度排序

### 其他

- 特别感谢[@Sm1les](https://github.com/Sm1les)对本项目的帮助与支持
- 部分lora代码和讲解参考仓库：https://github.com/zyds/transformers-code.git
- 如果有任何想法可以联系我们 DataWhale 也欢迎大家多多提出 issue
- 特别感谢以下为教程做出贡献的同学！


<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/self-llm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/self-llm" />
  </a>
</div>

### Star History

<div align=center style="margin-top: 30px;">
  <img src="./images/star-history-2024129.png"/>
</div>

