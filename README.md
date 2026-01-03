<div align=center>
  <img src="./images/head-img.png" >
  <h1>开源大模型食用指南</h1>
</div>

<div align="center">

中文 | [English](./README_en.md)

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

> &emsp;&emsp;**进阶学习推荐** ：如果您在学习完本项目后，希望更深入地理解大语言模型的核心原理，并渴望亲手从零开始训练属于自己的大模型，我们强烈推荐关注 Datawhale 的另一个开源项目—— [Happy-LLM 从零开始的大语言模型原理与实践教程](https://github.com/datawhalechina/happy-llm) 。该项目将带您深入探索大模型的底层机制，掌握完整的训练流程。

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

- [AMChat](./examples/AMchat-高等数学/readme.md): AM (Advanced Mathematics) chat 是一个集成了数学知识和高等数学习题及其解答的大语言模型。该模型使用 Math 和高等数学习题及其解析融合的数据集，基于 InternLM2-Math-7B 模型，通过 xtuner 微调，专门设计用于解答高等数学问题。

- [数字生命](./examples/数字生命/readme.md): 本项目将以我为原型，利用特制的数据集对大语言模型进行微调，致力于创造一个能够真正反映我的个性特征的AI数字人——包括但不限于我的语气、表达方式和思维模式等等，因此无论是日常聊天还是分享心情，它都以一种既熟悉又舒适的方式交流，仿佛我在他们身边一样。整个流程是可迁移复制的，亮点是数据集的制作。 

### 已支持模型

<p align="center">
  <strong>✨ 已支持 50+ 主流大语言模型 ✨</strong><br>
  <em>每个模型都提供完整的部署、微调和使用教程</em><br>
  📖 <strong><a href="./support_model.md">查看完整模型列表和教程</a></strong> | 
  🎯 <strong><a href="./support_model.md#通用环境配置">快速开始</a></strong>
</p>


<table align="center">
  <tr>
    <td valign="top" width="25%">
      • <a href="./support_model.md#gemma3">Gemma3</a><br>
      • <a href="./support_model.md#minimax-m2">MiniMax-M2</a><br>
      • <a href="./support_model.md#qwen3">Qwen3</a><br>
      • <a href="./support_model.md#qwen3-vl-4b-instruct">Qwen3-VL</a><br>
      • <a href="./support_model.md#spatiallm">SpatialLM</a><br>
      • <a href="./support_model.md#hunyuan3d-2">Hunyuan3D-2</a><br>
      • <a href="./support_model.md#qwen2-vl">Qwen2-VL</a><br>
      • <a href="./support_model.md#minicpm-o-2_6">MiniCPM-o</a><br>
      • <a href="./support_model.md#qwen25-coder">Qwen2.5-Coder</a><br>
      • <a href="./support_model.md#deepseek-coder-v2">DeepSeek-Coder-V2</a><br>
      • <a href="./support_model.md#gpt-oss-20b">gpt-oss-20b</a><br>
      • <a href="./support_model.md#glm-41-thinking">GLM-4.1-Thinking</a>
    </td>
    <td valign="top" width="25%">
      • <a href="./support_model.md#deepseek-r1-distill">DeepSeek-R1</a><br>
      • <a href="./support_model.md#internlm3">InternLM3</a><br>
      • <a href="./support_model.md#phi4">phi4</a><br>
      • <a href="./support_model.md#glm-45-air">GLM-4.5-Air</a><br>
      • <a href="./support_model.md#hunyuan-a13b-instruct">Hunyuan-A13B</a><br>
      • <a href="./support_model.md#deepseek-深度求索">DeepSeek</a><br>
      • <a href="./support_model.md#baichuan-百川智能">Baichuan</a><br>
      • <a href="./support_model.md#internlm">InternLM</a><br>
      • <a href="./support_model.md#kimi">Kimi</a><br>
      • <a href="./support_model.md#ernie-45">ERNIE-4.5</a><br>
      • <a href="./support_model.md#llama4">Llama4</a><br>
      • <a href="./support_model.md#apple-openelm">Apple OpenELM</a>
    </td>
    <td valign="top" width="25%">
      • <a href="./support_model.md#llama31-8b-instruct">Llama3.1</a><br>
      • <a href="./support_model.md#gemma-2-9b-it">Gemma-2</a><br>
      • <a href="./support_model.md#qwen25">Qwen2.5</a><br>
      • <a href="./support_model.md#qwen2">Qwen2</a><br>
      • <a href="./support_model.md#glm-4">GLM-4</a><br>
      • <a href="./support_model.md#qwen-15">Qwen 1.5</a><br>
      • <a href="./support_model.md#phi-3">phi-3</a><br>
      • <a href="./support_model.md#minicpm">MiniCPM</a><br>
      • <a href="./support_model.md#yi-零一万物">Yi 零一万物</a><br>
      • <a href="./support_model.md#yuan20">Yuan2.0</a><br>
      • <a href="./support_model.md#yuan20-m32">Yuan2.0-M32</a><br>
      • <a href="./support_model.md#哔哩哔哩-index-19b">哔哩哔哩 Index</a>
    </td>
    <td valign="top" width="25%">
      • <a href="./support_model.md#characterglm-6b">CharacterGLM</a><br>
      • <a href="./support_model.md#bluelm-vivo-蓝心大模型">BlueLM</a><br>
      • <a href="./support_model.md#qwen-audio">Qwen-Audio</a><br>
      • <a href="./support_model.md#transnormerllm">TransNormerLLM</a><br>
      • <a href="./support_model.md#atom-llama2">Atom</a><br>
      • <a href="./support_model.md#chatglm3">ChatGLM3</a><br>
      • <a href="./support_model.md#qwen2-57b-a14b-instruct">Qwen2-57B-A14B-Instruct</a><br>
      • <a href="./support_model.md#qwen2-72b-instruct">Qwen2-72B-Instruct</a><br>
      • <a href="./support_model.md#qwen2-7b-instruct">Qwen2-7B-Instruct</a><br>
      • <a href="./support_model.md#internlm2-20b">InternLM2-20B</a><br>
      • <a href="./support_model.md#tele-chat">Tele-Chat</a><br>
      • <a href="./support_model.md#xverse2">XVERSE2</a>
    </td>
  </tr>
</table>

### AMD GPU 专区

<p align="center">
  <strong>🚀 AMD GPU 平台已支持模型</strong><br>
  <em>每个模型都提供完整的 AMD 环境配置和部署教程</em><br>
  <em>感谢 AMD University Program 对本项目的支持</em><br>
  📖 <strong><a href="./support_model_amd.md">查看完整 AMD 平台模型列表和教程</a></strong><br>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • <a href="./support_model_amd.md#谷歌-gemma3">谷歌 Gemma3</a><br>
      • AMD 环境准备与配置<br>
      • NPU 推理加速支持
    </td>
    <td valign="top" width="50%">
      • <a href="./support_model_amd.md#qwen3">Qwen3</a><br>
      • lemonade-server SDK 部署<br>
      • Ryzen AI 300 系列优化
    </td>
  </tr>
</table>

### 昇腾Ascend NPU 专区

<p align="center">
  <strong>🚀 昇腾Ascend NPU 平台已支持模型</strong><br>
  <em>每个模型都提供完整的昇腾Ascend NPU 环境配置和部署教程</em><br>
  📖 <strong><a href="./support_model_Ascend.md">查看完整昇腾 NPU 平台模型列表和教程</a></strong><br>
</p>

<table align="center">
  <tr>
    <td valign="top" width="50%">
      • <a href="./support_model_Ascend.md#qwen3">Qwen3</a><br>
      • Ascend NPU 环境配置通用指南<br>
      • Ascend NPU 大模型推理性能优化建议
    </td>
  </tr>
</table>

### 沐曦专区

<p align="center">
  <strong><em>Coming Soon!</em></strong>
</p>

### Welcome More Platforms!

- 🚀 即将支持更多平台（Apple M 系列已有设备测试），敬请期待！
- 🤝 欢迎昇腾 Ascend、摩尔线程 MUSA、沐曦等平台提供技术支持、硬件支持或参与贡献
- 🌟 欢迎各平台开发者共建共享，推动大模型技术在更多国产硬件生态中的繁荣发展！

## 致谢

### 核心贡献者

- [宋志学(不要葱姜蒜)-项目负责人](https://github.com/KMnO4-zx) （Datawhale成员）
- [邹雨衡-项目负责人](https://github.com/logan-zou) （Datawhale成员-对外经济贸易大学）
- [姜舒凡](https://github.com/Tsumugii24)（内容创作者-Datawhale成员）
- [郭宣伯](https://github.com/Twosugar666)（内容创作者-北京航空航天大学）
- [林泽毅](https://github.com/Zeyi-Lin)（内容创作者-SwanLab产品负责人）
- [林恒宇](https://github.com/LINHYYY)（内容创作者-广东东软学院-鲸英助教）
- [王泽宇](https://github.com/moyitech)（内容创作者-太原理工大学-鲸英助教）
- [郭志航](https://github.com/acwwt)（内容创作者）
- [陈榆](https://github.com/LucaChen)（内容创作者-谷歌开发者机器学习技术专家）
- [肖鸿儒](https://github.com/Hongru0306) （Datawhale成员-同济大学）
- [张帆](https://github.com/zhangfanTJU)（内容创作者-Datawhale成员）
- [李娇娇](https://github.com/Aphasia0515) （Datawhale成员）
- [高立业](https://github.com/0-yy-0)（内容创作者-DataWhale成员）
- [Kailigithub](https://github.com/Kailigithub) （Datawhale成员）
- [丁悦](https://github.com/dingyue772) （Datawhale-鲸英助教）
- [惠佳豪](https://github.com/L4HeyXiao) （Datawhale-宣传大使）
- [王茂霖](https://github.com/mlw67)（内容创作者-Datawhale成员）
- [孙健壮](https://github.com/Caleb-Sun-jz)（内容创作者-对外经济贸易大学）
- [郑皓桦](https://github.com/BaiYu96) （内容创作者）
- [荞麦](https://github.com/yeyeyeyeeeee)（内容创作者-Datawhale成员）
- [骆秀韬](https://github.com/anine09)（内容创作者-Datawhale成员-似然实验室）
- [李柯辰](https://github.com/Joe-2002) （Datawhale成员）
- [程宏](https://github.com/chg0901)（内容创作者-Datawhale意向成员）
- [谢好冉](https://github.com/ilovexsir)（内容创作者-鲸英助教）
- [李秀奇](https://github.com/li-xiu-qi)（内容创作者-DataWhale意向成员）
- [陈思州](https://github.com/jjyaoao) （Datawhale成员）
- [颜鑫](https://github.com/thomas-yanxin) （Datawhale成员）
- [杜森](https://github.com/study520ai520)（内容创作者-Datawhale成员-南阳理工学院）
- [散步](https://github.com/sanbuphy) （Datawhale成员）
- [Swiftie](https://github.com/cswangxiaowei) （小米NLP算法工程师）
- [张友东](https://github.com/AXYZdong)（内容创作者-Datawhale成员）
- [张晋](https://github.com/Jin-Zhang-Yaoguang)（内容创作者-Datawhale成员）
- [娄天奥](https://github.com/lta155)（内容创作者-中国科学院大学-鲸英助教）
- [小罗](https://github.com/lyj11111111) （内容创作者-Datawhale成员）
- [邓恺俊](https://github.com/Kedreamix)（内容创作者-Datawhale成员）
- [赵文恺](https://github.com/XiLinky)（内容创作者-太原理工大学-鲸英助教）
- [王熠明](https://github.com/Bald0Wang)（内容创作者-Datawhale成员）
- [黄柏特](https://github.com/KashiwaByte)（内容创作者-西安电子科技大学）
- [余洋](https://github.com/YangYu-NUAA)（内容创作者-Datawhale成员）
- [左春生](https://github.com/LinChentang)（内容创作者-Datawhale成员）
- [杨卓](https://github.com/little1d)（内容创作者-西安电子科技大学-鲸英助教）
- [付志远](https://github.com/comfzy)（内容创作者-海南大学）
- [郑远婧](https://github.com/isaacahahah)（内容创作者-鲸英助教-福州大学）
- [谭逸珂](https://github.com/LikeGiver)（内容创作者-对外经济贸易大学）
- [何至轩](https://github.com/pod2c)（内容创作者-鲸英助教）
- [康婧淇](https://github.com/jodie-kang)（内容创作者-Datawhale成员）
- [三水](https://github.com/sssanssss)（内容创作者-鲸英助教）
- [杨晨旭](https://github.com/langlibai66)（内容创作者-太原理工大学-鲸英助教）
- [赵伟](https://github.com/2710932616)（内容创作者-鲸英助教）
- [苏向标](https://github.com/gzhuuser)（内容创作者-广州大学-鲸英助教）
- [陈睿](https://github.com/riannyway)（内容创作者-西交利物浦大学-鲸英助教）
- [张龙斐](https://github.com/Feimike09)（内容创作者-鲸英助教）
- [孙超](https://github.com/anarchysaiko)（内容创作者-Datawhale成员）
- [樊奇](https://github.com/fanqiNO1)（内容创作者-上海交通大学）
- [卓堂越](https://github.com/nusakom)（内容创作者-鲸英助教）
- [fancy](https://github.com/fancyboi999)（内容创作者-鲸英助教）

> 注：排名根据贡献程度排序

### 其他

- 特别感谢[@Sm1les](https://github.com/Sm1les)对本项目的帮助与支持
- 感谢 AMD University Program 对本项目的支持
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
  <img src="./images/star-history-20251220.png"/>
</div>
