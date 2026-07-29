## Gemma 4 模型介绍

> 内容参考 [hello-rocm](https://github.com/datawhalechina/hello-rocm) `docs/zh/01-deploy`，适配 self-llm 目录结构。完整与最新版请以 hello-rocm 为准。

Gemma 4 是 Google DeepMind 发布的新一代多模态开源模型家族，基于与 Gemini 3 相同的研究成果和技术构建，采用 **Apache 2.0** 许可证，支持文本、图像、视频和音频输入，生成文本输出。Gemma 4 在前几代的基础上进行了全面升级——更强的推理能力、更灵活的多模态支持、更高效的架构设计——同时提供从端侧到服务器的多种尺寸，专为高级推理和智能体（Agentic）工作流而设计。

自首代 Gemma 发布以来，开发者已累计下载超过 **4 亿次**，社区构建了超过 **10 万个** Gemma 变体，形成了庞大的 Gemmaverse 生态。Gemma 4 是 Google 对社区需求的最新回应——以参数效率实现前沿能力。

> 参考来源：[Hugging Face Blog - Welcome Gemma 4](https://huggingface.co/blog/gemma4) · [Google Blog - Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)

---

### 一、模型版本总览

Gemma 4 提供四个尺寸，每个尺寸均有 **base（基础）** 和 **IT（指令微调）** 两个版本：


| 模型                  | 参数规模                     | 上下文窗口 | 模型地址                                                                                                           |
| ------------------- | ------------------------ | ----- | -------------------------------------------------------------------------------------------------------------- |
| **Gemma 4 E2B**     | 有效参数 2.3B，含嵌入共 5.1B      | 128K  | [base](https://huggingface.co/google/gemma-4-E2B) / [IT](https://huggingface.co/google/gemma-4-E2B-it)         |
| **Gemma 4 E4B**     | 有效参数 4.5B，含嵌入共 8B        | 128K  | [base](https://huggingface.co/google/gemma-4-E4B) / [IT](https://huggingface.co/google/gemma-4-E4B-it)         |
| **Gemma 4 31B**     | 31B 稠密模型                 | 256K  | [base](https://huggingface.co/google/gemma-4-31B) / [IT](https://huggingface.co/google/gemma-4-31B-it)         |
| **Gemma 4 26B A4B** | MoE 架构，激活参数 4B / 总参数 26B | 256K  | [base](https://huggingface.co/google/gemma-4-26B-A4B) / [IT](https://huggingface.co/google/gemma-4-26B-A4B-it) |


其中，**E2B** 和 **E4B** 为适合端侧部署的小型变体，支持图像、视频、文本和**音频**输入；**31B** 和 **26B A4B** 为大型变体，支持图像、视频和文本输入。

**Arena AI 排名（截至 2026 年 4 月）**：31B 稠密模型在 Arena AI 文本排行榜上排名开源模型 **第 3**，26B MoE 排名 **第 6**，能够击败参数量 20 倍于自身的模型。这种极高的"每参数智能"意味着开发者可以用显著更少的硬件实现前沿级能力。

#### 大型模型（26B / 31B）

面向研究者和开发者，未量化的 bfloat16 权重可高效运行在单张 80GB NVIDIA H100 GPU 上；量化版本可在消费级 GPU 上本地运行，驱动 IDE、编程助手和智能体工作流。26B MoE 在推理时仅激活 3.8B 参数，延迟极低；31B Dense 则追求最大的原始质量，也是微调的理想基础。

#### 端侧模型（E2B / E4B）

专为移动端和 IoT 设备从头设计，在推理时仅激活有效的 2B / 4B 参数量，以节省内存和电池寿命。Google 与 Pixel 团队以及 Qualcomm、MediaTek 等移动硬件厂商深度合作，使这些多模态模型可在手机、Raspberry Pi、NVIDIA Jetson Orin Nano 等边缘设备上**完全离线、近零延迟**运行。Android 开发者可通过 AICore Developer Preview 进行原型开发，并与未来的 Gemini Nano 4 前向兼容。

---

### 二、架构特点

Gemma 4 融合了多种经过验证的架构创新，在兼容性、效率和长上下文支持之间取得了优秀的平衡：

#### 2.1 交替注意力机制

Gemma 4 采用**局部滑动窗口注意力**与**全局全上下文注意力**交替排列的层结构。较小的稠密模型使用 512 token 的滑动窗口，较大模型使用 1024 token。这种设计在控制计算量的同时保留了长距离依赖的建模能力。

#### 2.2 双 RoPE 配置

模型使用两种 RoPE（旋转位置编码）配置：滑动窗口层使用标准 RoPE，全局层使用比例 RoPE，以更好地支持超长上下文。

#### 2.3 逐层嵌入（Per-Layer Embeddings, PLE）

PLE 是 Gemma 4 小型模型中最具特色的创新之一（最初在 Gemma-3n 中引入）。传统 Transformer 中每个 token 只在输入时获得一个嵌入向量，所有后续层都在此基础上构建。PLE 则在主残差流之外增加了一条并行的、低维度的条件化路径——为每个 token 在每一层生成一个专属的小向量，结合了 token 身份信息和上下文信息。这使得每一层都能接收到 token 特定的信息，而无需在初始嵌入中打包所有内容。由于 PLE 维度远小于主隐藏维度，增加的参数成本很低，但能带来显著的逐层特化效果。

#### 2.4 共享 KV 缓存（Shared KV Cache）

模型的最后若干层不再独立计算自己的 Key 和 Value 投影，而是**复用**同类型注意力（滑动或全局）中最后一个非共享层的 KV 张量。这一优化在几乎不影响质量的前提下，显著降低了长文本推理时的内存占用和计算量，非常适合端侧部署。

#### 2.5 视觉编码器

视觉编码器使用学习型 2D 位置编码和多维 RoPE，支持**可变宽高比**输入，并可配置不同的图像 token 数量（70、140、280、560、1120），在速度、内存和质量之间灵活取舍。

#### 2.6 音频编码器

小型变体（E2B、E4B）内置 USM 风格的 Conformer 音频编码器，与 Gemma-3n 使用相同的基础架构，支持语音理解和转录任务。

---

### 三、核心能力

Gemma 4 超越了简单的聊天场景，在多个维度上均展现出强大的实力：

#### 3.1 高级推理

具备多步规划和深度逻辑能力，在数学和指令遵循基准测试上有显著提升。

#### 3.2 智能体工作流（Agentic Workflows）

原生支持**函数调用（Function Calling）**、**结构化 JSON 输出**和**系统指令**，可构建与外部工具和 API 交互的自主智能体，可靠地执行复杂工作流。

#### 3.3 代码生成

支持高质量的离线代码生成，可将工作站变为本地优先的 AI 编程助手。

#### 3.4 视觉与音频

所有模型原生处理视频和图像，支持可变分辨率，擅长 OCR、图表理解等视觉任务。E2B 和 E4B 额外支持原生音频输入，用于语音识别和理解。

#### 3.5 长上下文

端侧模型支持 **128K** 上下文窗口，大型模型最高 **256K**，可在单次提示中传入完整代码仓库或长文档。

#### 3.6 多语言支持

原生训练覆盖 **140+ 种语言**，帮助开发者为全球用户构建高性能应用。

#### 3.7 多模态任务实测

在 Hugging Face 的实际测试中，Gemma 4 在以下多模态场景上表现出开箱即用的能力：

- **目标检测与定位**：模型能原生以 JSON 格式返回检测到的边界框坐标，无需特定指令或约束生成。
- **GUI 元素检测**：可识别并定位 UI 界面中的元素（如按钮、菜单项等），适用于 GUI 自动化场景。
- **图像描述**：所有尺寸的模型在复杂场景的细节捕捉上都表现出色。
- **视频理解**：虽未在视频上显式后训练，但模型能理解有声或无声视频内容。小型变体可同时处理视频画面与音频。
- **音频问答与转录**：E2B 和 E4B 支持对语音内容进行描述和转录，训练数据以语音为主。
- **多模态推理（Thinking）**：支持在多模态输入上启用思维链推理模式。
- **多模态函数调用**：可根据图像内容识别信息并调用外部工具，例如识别图片中的地点后调用天气查询 API。

---

### 四、基准测试结果

Gemma 4 在推理、编码、视觉、音频和长上下文等多个维度上都取得了出色成绩。31B 稠密模型的 LMArena 评分（纯文本）达到 **1452**，而 26B MoE 仅用 4B 激活参数就达到了 **1441**。

以下为指令微调版本的主要基准测试数据：


| 基准测试                 | Gemma 4 31B | Gemma 4 26B A4B | Gemma 4 E4B | Gemma 4 E2B | Gemma 3 27B |
| -------------------- | ----------- | --------------- | ----------- | ----------- | ----------- |
| **推理与知识**            |             |                 |             |             |             |
| MMLU Pro             | 85.2%       | 82.6%           | 69.4%       | 60.0%       | 67.6%       |
| AIME 2026 (no tools) | 89.2%       | 88.3%           | 42.5%       | 37.5%       | 20.8%       |
| GPQA Diamond         | 84.3%       | 82.3%           | 58.6%       | 43.4%       | 42.4%       |
| BigBench Extra Hard  | 74.4%       | 64.8%           | 33.1%       | 21.9%       | 19.3%       |
| **编码**               |             |                 |             |             |             |
| LiveCodeBench v6     | 80.0%       | 77.1%           | 52.0%       | 44.0%       | 29.1%       |
| Codeforces ELO       | 2150        | 1718            | 940         | 633         | 110         |
| **视觉**               |             |                 |             |             |             |
| MMMU Pro             | 76.9%       | 73.8%           | 52.6%       | 44.2%       | 49.7%       |
| MATH-Vision          | 85.6%       | 82.4%           | 59.5%       | 52.4%       | 46.0%       |
| **长上下文**             |             |                 |             |             |             |
| MRCR v2 128K (avg)   | 66.4%       | 44.1%           | 25.4%       | 19.1%       | 13.5%       |


**关键亮点：**

- **26B A4B（MoE）** 仅激活 4B 参数，性能接近 31B 稠密模型，效率极高。
- 相比前代 Gemma 3 27B，Gemma 4 在 AIME 2026 数学推理上提升超过 **4 倍**（20.8% → 89.2%），Codeforces 编码评分提升超过 **15 倍**（110 → 1718+）。
- 视觉和长上下文能力均大幅超越前代。

---

### 五、生态与推理框架支持

Gemma 4 拥有广泛的工具链和平台支持，开发者可根据需求灵活选择：

#### 5.1 推理框架（首日支持）


| 框架                    | 说明                                                       |
| --------------------- | -------------------------------------------------------- |
| **transformers**      | 官方集成，支持 `AutoModelForMultimodalLM`、`any-to-any` pipeline |
| **llama.cpp**         | 支持图文推理，可通过 OpenAI 兼容 API 使用，适配 LM Studio、Jan 等本地应用       |
| **vLLM**              | 高性能推理引擎                                                  |
| **SGLang**            | 高性能推理引擎，transformers 后端                                  |
| **MLX**               | Apple Silicon 原生支持，含 TurboQuant 量化优化                     |
| **Ollama**            | 一键本地部署                                                   |
| **LM Studio**         | 图形界面本地推理                                                 |
| **transformers.js**   | 浏览器端运行（WebGPU）                                           |
| **mistral.rs**        | Rust 原生推理引擎，支持全模态和内置工具调用                                 |
| **ONNX**              | 提供 ONNX 格式 checkpoint，支持边缘设备和浏览器部署                       |
| **LiteRT-LM**         | Google 端侧推理运行时                                           |
| **NVIDIA NIM / NeMo** | NVIDIA 推理与训练平台                                           |


#### 5.2 硬件平台

Gemma 4 针对多种硬件平台进行了优化：

- **NVIDIA GPU**：从 Jetson Orin Nano 到 Blackwell 系列全覆盖
- **AMD GPU**：通过开源 **ROCm** 软件栈集成
- **Google TPU**：支持 Trillium 和 Ironwood TPU 大规模部署
- **Apple Silicon**：通过 MLX 原生支持
- **移动端芯片**：与 Qualcomm、MediaTek 合作优化

#### 5.3 开发平台与快速体验


| 平台                                                                | 用途                               |
| ----------------------------------------------------------------- | -------------------------------- |
| [Google AI Studio](https://aistudio.google.com/)                  | 在线体验 31B 和 26B MoE               |
| [Google AI Edge Gallery](https://ai.google.dev/edge)              | 体验端侧模型 E4B 和 E2B                 |
| [Android Studio Agent Mode](https://developer.android.com/studio) | Android 应用开发集成                   |
| [Hugging Face](https://huggingface.co/collections/google/gemma-4) | 模型下载与社区                          |
| [Kaggle](https://www.kaggle.com/)                                 | 模型下载与 Gemma 4 Good Challenge 挑战赛 |
| [Google Colab](https://colab.research.google.com/)                | 免费微调与实验                          |
| [Vertex AI](https://cloud.google.com/vertex-ai)                   | 企业级部署（Cloud Run / GKE / TPU 加速）  |


---

### 六、微调支持

Gemma 4 的高度优化架构使其在各种硬件上都能高效微调——从数十亿 Android 设备到笔记本 GPU、开发者工作站再到加速器。社区已有成功的微调案例：INSAIT 基于 Gemma 创建了保加利亚语优先的语言模型 BgGPT，Yale 大学利用 Cell2Sentence-Scale 发现了癌症治疗的新途径。

Gemma 4 支持使用多种工具进行微调：

- **TRL（Transformer Reinforcement Learning）**：支持多模态工具响应训练，可从环境中接收图像反馈（如在 CARLA 模拟器中训练 Gemma 4 学习驾驶）。
- **TRL on Vertex AI**：提供在 Google Cloud Vertex AI 上进行 SFT 微调的完整示例，可冻结视觉和音频塔进行函数调用能力扩展。
- **Unsloth Studio**：支持通过图形界面进行本地或 Colab 微调。
- **Keras**：Google 高级深度学习 API 支持。

---

### 七、许可证与安全

Gemma 4 采用 **Apache 2.0** 开源许可证，赋予开发者完全的灵活性和数字主权——对数据、基础设施和模型拥有完全控制权，可自由地在本地或云端的任何环境中构建和部署。

在安全性方面，Gemma 4 经历了与 Google 专有模型相同的严格基础设施安全协议，为企业和主权机构提供可信赖、透明的基础。

---

### 八、小结

Gemma 4 是当前开源多模态模型中的佼佼者，其核心优势包括：

1. **真正开源**：Apache 2.0 许可证，可自由商用，完全掌控。
2. **极致的参数效率**：击败参数量 20 倍于自身的模型，Arena AI 开源排名 Top 3/6。
3. **全模态支持**：文本、图像、视频、音频统一处理，覆盖 140+ 语言。
4. **覆盖全尺寸**：从 2.3B 端侧模型（手机/IoT）到 31B 服务器级模型，适配各类场景。
5. **架构高效**：PLE、共享 KV 缓存、MoE 等设计使模型在保持高性能的同时显著降低资源需求。
6. **智能体就绪**：原生函数调用、结构化输出和系统指令，适合构建复杂的 Agentic 工作流。
7. **生态完善**：主流推理框架首日支持，多硬件平台优化（NVIDIA / AMD ROCm / Google TPU / Apple Silicon），量化与微调工具齐备。

> 更多信息请参阅：
>
> - [Hugging Face Blog - Welcome Gemma 4](https://huggingface.co/blog/gemma4)
> - [Google Blog - Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
> - [Hugging Face Gemma 4 模型合集](https://huggingface.co/collections/google/gemma-4)

---

## 延伸学习

更完整的 ROCm 部署、微调与基础设施教程，请前往 Datawhale 项目 [hello-rocm](https://github.com/datawhalechina/hello-rocm)：

- 仓库：[https://github.com/datawhalechina/hello-rocm](https://github.com/datawhalechina/hello-rocm)
- 在线文档：[https://datawhalechina.github.io/hello-rocm/](https://datawhalechina.github.io/hello-rocm/)
- 部署专区原文：[https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy](https://github.com/datawhalechina/hello-rocm/tree/main/docs/zh/01-deploy)

> 本文内容参考 hello-rocm `01-deploy` 教程，并按 self-llm `models_amd` 目录结构做了路径适配。