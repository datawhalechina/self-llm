# 已支持模型列表

> 本仓库致力于知错和整理各大语言模型的部署、微调和使用教程。我们为每个支持的模型提供了详细的部署指南、API调用示例、LangChain集成方案、WebDemo实现以及微调方法。所有教程均经过实践验证，确保能够在实际环境中顺利运行。欢迎贡献新的模型教程或改进现有文档！

## 目录

- [GLM-4.7-Flash](#glm-47-flash)
- [谷歌-Gemma3](#谷歌-gemma3)
- [MiniMax-M2](#minimax-m2)
- [Qwen3-VL-4B-Instruct](#qwen3-vl-4b-instruct)
- [BGE-M3](#bge-m3)
- [gpt-oss-20b](#gpt-oss-20b)
- [GLM-4.1-Thinking](#glm-41-thinking)
- [GLM-4.5-Air](#glm-45-air)
- [ERNIE-4.5](#ernie-45)
- [Hunyuan-A13B-Instruct](#hunyuan-a13b-instruct)
- [Qwen3](#qwen3)
- [Kimi](#kimi)
- [Llama4](#llama4)
- [SpatialLM](#spatiallm)
- [Hunyuan3D-2](#hunyuan3d-2)
- [Gemma3](#gemma3)
- [DeepSeek-R1-Distill](#deepseek-r1-distill)
- [MiniCPM-o-2_6](#minicpm-o-2_6)
- [InternLM3](#internlm3)
- [phi4](#phi4)
- [Qwen2.5-Coder](#qwen25-coder)
- [Qwen2-vl](#qwen2-vl)
- [Qwen2.5](#qwen25)
- [Apple OpenELM](#apple-openelm)
- [Llama3_1-8B-Instruct](#llama31-8b-instruct)
- [Gemma-2-9b-it](#gemma-2-9b-it)
- [Yuan2.0](#yuan20)
- [Yuan2.0-M32](#yuan20-m32)
- [DeepSeek-Coder-V2](#deepseek-coder-v2)
- [哔哩哔哩 Index-1.9B](#哔哩哔哩-index-19b)
- [Qwen2](#qwen2)
- [GLM-4](#glm-4)
- [Qwen 1.5](#qwen-15)
- [phi-3](#phi-3)
- [CharacterGLM-6B](#characterglm-6b)
- [LLaMA3-8B-Instruct](#llama3-8b-instruct)
- [XVERSE-7B-Chat](#xverse-7b-chat)
- [TransNormerLLM](#transnormerllm)
- [BlueLM Vivo 蓝心大模型](#bluelm-vivo-蓝心大模型)
- [InternLM2](#internlm2)
- [DeepSeek 深度求索](#deepseek-深度求索)
- [MiniCPM](#minicpm)
- [Qwen-Audio](#qwen-audio)
- [Qwen](#qwen)
- [Yi 零一万物](#yi-零一万物)
- [Baichuan 百川智能](#baichuan-百川智能)
- [InternLM](#internlm)
- [Atom (llama2)](#atom-llama2)
- [ChatGLM3](#chatglm3)
- [通用环境配置](#通用环境配置)


## 已支持模型列表

### Kimi-K2.5

[Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5)
- [x] [Kimi-K2.5 论文解读](./models/Kimi-K2.5/01-Kimi-K2.5-论文解读.md) @樊奇
- [ ] Kimi-K2.5 vLLM 部署调用及 Docker 镜像
- [ ] Kimi-K2.5 SGLang 部署调用及 Docker 镜像

### Step-3.5-Flash

[Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash)
- [ ] Step-3.5-Flash vLLM 部署调用及 Docker 镜像
- [ ] Step-3.5-Flash SGLang 部署调用及 Docker 镜像
- [ ] Step-3.5-Flash Lora 微调及 Docker 镜像

### GLM-4.7-Flash

[GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash)
- [ ] GLM-4.7-Flash vLLM 部署调用及 Docker 镜像
- [ ] GLM-4.7-Flash SGLang 部署调用及 Docker 镜像
- [ ] GLM-4.7-Flash Lora 微调及 Docker 镜像

### 谷歌-Gemma3

[谷歌-Gemma3](https://huggingface.co/google/gemma-7b-it)
- [x] [gemma-2b-it FastApi 部署调用 ](./models/Gemma/01-Gemma-2B-Instruct%20FastApi%20部署调用.md) @陈榆
- [x] [gemma-2b-it langchain 接入 ](./models/Gemma/02-Gemma-2B-Instruct%20langchain%20接入.md) @陈榆
- [x] [gemma-2b-it WebDemo 部署 ](./models/Gemma/03-Gemma-2B-Instruct%20WebDemo%20部署.md) @陈榆
- [x] [gemma-2b-it Peft Lora 微调 ](./models/Gemma/04-Gemma-2B-Instruct%20Lora微调.md) @陈榆
- [X] [gemma3-4b-it AMD 环境准备](./models/Gemma3/7-gemma3-4b-it%20AMD环境准备.md) @陈榆
- [X] [gemma3-4b-it AMD 模型服务部署](./models/Gemma3/8-gemma3-4b-it%20模型服务部署.md) @陈榆

### MiniMax-M2

[MiniMax-M2](https://github.com/MiniMax-AI/MiniMax-M2)
- [x] [MiniMax-M2 在线体验地址](https://agent.minimax.io/)
- [x] [MiniMax-M2 Hugging Face 地址](https://huggingface.co/MiniMaxAI/MiniMax-M2)
- [x] [MiniMax-M2 Text Generation Guide](https://platform.minimax.io/docs/guides/text-generation)
- [x] [MiniMax-M2 模型结构解析 Blog](./models/MiniMax-M2/1-MiniMax-M2-Blog.md) @王泽宇
- [x] [MiniMax-M2 vllm 部署调用](./models/MiniMax-M2/2-MiniMax-M2-vLLM.md) @姜舒凡
- [x] [MiniMax-M2 SGLang 部署调用](./models/MiniMax-M2/3-MiniMax-M2-SGLang.md) @姜舒凡
- [x] [MiniMax-M2 evalscope 智商情商评测及并发评测](./models/MiniMax-M2/4-MiniMax-M2-EvalScope.md) @姜舒凡
- [x] [AutoDL MiniMax-M2 vllm部署及evalscope镜像](https://www.codewithgpu.com/i/datawhalechina/self-llm/mimimax-m2) @姜舒凡

### Qwen3-VL-4B-Instruct

[Qwen3-VL-4B-Instruct](https://github.com/QwenLM/Qwen3-VL)
  - [x] [Qwen3-VL 模型结构解析（DeepStack解析）](./models/Qwen3-VL/01-Qwen3-VL-MoE-模型结构解析-Blog.md) @王泽宇
  - [x] [Qwen3-VL-4B-Instruct FastApi 部署调用](./models/Qwen3-VL/02-Qwen3-VL-4B-Instruct%20FastApi%20部署调用.md) @王嘉鹏
  - [x] [Qwen3-VL-4B-Instruct vLLM 部署](./models/Qwen3-VL/04-Qwen3-VL-4B-Instruct-vLLM.md) @姜舒凡
  - [x] [Qwen3-VL-4B-Instruct Lora 可视化微调案例-LaTexOCR](./models/Qwen3-VL/05-Qwen3-VL-4B-Instruct%20%20Lora%20可视化微调案例%20-%20LaTexOCR.md) @李秀奇

### BGE-M3

[BGE-M3](https://huggingface.co/BAAI/bge-m3)
  - [x] [代码检索场景微调实战 微调BGE-M3 embedding模型](./models/BGE-M3-finetune-embedding-with-valid/README.md) @李秀奇

### gpt-oss-20b

[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
  - [x] [gpt-oss-20b vllm 部署调用](./models/gpt-oss/1-gpt-oss-20b%20vllm%20部署调用.md)@郭宣伯
  - [x] [gpt-oss-20b EvalScope 并发评测](./models/gpt-oss/2-gpt-oss-20b%20Evalscope并发测试.md) @郭宣伯
  - [x] [gpt-oss-20b lmstudio 本地部署调用](./models/gpt-oss/3-gpt-oss-20b%20lmstudio%20本地部署调用.md) @郭
  - [x] [gpt-oss-20b Lora 微调及 SwanLab 可视化记录](./models/gpt-oss/4-gpt-oss-20b%20Lora%20微调及%20SwanLab%20可视化记录.md) @郭宣伯
  - [x] [gpt-oss-20b DPO 微调及 SwanLab 可视化记录](./models/gpt-oss/5-gpt-oss-20b%20DPO%20微调及%20SwanLab%20可视化记录.md) @郭宣伯

### GLM-4.1-Thinking

[GLM-4.1-Thinking](https://github.com/zai-org/GLM-4.1V-Thinking)
  - [x] [GLM-4.1V-Thinking vLLM 部署调用](./models/GLM-4.1V-Thinking/01-GLM-4%201V-Thinking%20vLLM部署调用.md) @林恒宇
  - [x] [GLM-4.1V-Thinking Gradio部署](./models/GLM-4.1V-Thinking/02-GLM-4%201V-Thinking%20Gradio部署.md) @林恒宇
  - [x] [GLM-4.1V-Thinking Lora 微调及 SwanLab 可视化记录](./models/GLM-4.1V-Thinking/03-GLM-4%201V-Thinking%20LoRA%20及%20SwanLab%20可视化记录.md) @林恒宇
  - [x] [GLM-4.1V-Thinking Docker 镜像](https://www.codewithgpu.com/i/datawhalechina/self-llm/GLM4.1V-Thinking-lora) @林恒宇

### GLM-4.5-Air

[GLM-4.5-Air](https://github.com/zai-org/GLM-4.5)
  - [x] [GLM-4.5-Air vLLM 部署调用](./models/GLM-4.5-Air/01-GLM-4.5-Air-vLLM%20部署调用.md) @不要葱姜蒜
  - [x] [GLM-4.5-Air EvalScope 智商情商 && 并发评测](./models/GLM-4.5-Air/02-GLM-4.5-Air%20EvalScope%20并发测试.md) @不要葱姜蒜
  - [x] [GLM-4.5-Air Lora 微调](./models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20及%20Swanlab%20可视化微调.md) @不要葱姜蒜
  - [x] [GLM-4.5-Air Ucloud Docker 镜像](https://www.compshare.cn/images/lUQhKDCeCdZW?referral_code=ELukJdQS3vvCwYIfgsQf2C&ytag=GPU_yy_github_selfllm) @不要葱姜蒜

### ERNIE-4.5

[ERNIE-4.5](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT)
  - [x] [ERNIE-4.5-0.3B-PT Lora 微调及 SwanLab 可视化记录](./models/ERNIE-4.5/01-ERNIE-4.5-0.3B-PT%20Lora%20微调及%20SwanLab%20可视化记录.md) @不要葱姜蒜
  - [x] [ERNIE-4.5-0.3B-PT Lora Docker 镜像](https://www.codewithgpu.com/i/datawhalechina/self-llm/ERNIE-4.5-lora) @不要葱姜蒜

### Hunyuan-A13B-Instruct

[Hunyuan-A13B-Instruct](https://github.com/Tencent-Hunyuan/Hunyuan-A13B)
  - [x] [Hunyuan-A13B-Instruct 模型架构解析 Blog](./models/Hunyuan-A13B-Instruct/01-Hunyuan-A13B-Instruct%20模型架构解析%20Blog.md) @卓堂越
  - [x] [Hunyuan-A13B-Instruct SGLang 部署调用](./models/Hunyuan-A13B-Instruct/03-Hunyuan-A13B-Instruct-SGLang部署调用.md) @fancy
  - [x] [Hunyuan-A13B-Instruct Lora SwanLab 可视化微调](./models/Hunyuan-A13B-Instruct/05-Hunyuan-A13B-Instruct-LoRA及SwanLab可视化记录.md) @谢好冉
  - [x] [Hunyuan-A13B-Instruct Lora Docker 镜像](https://www.codewithgpu.com/i/datawhalechina/self-llm/Hunyuan-A13B-Instruct-lora) @谢好冉

### Qwen3

[Qwen3](https://github.com/QwenLM/Qwen3)
  - [x] [Qwen3 模型结构解析 Blog](./models/Qwen3/01-Qwen3-模型结构解析-Blog.md) @王泽宇
  - [x] [Qwen3-8B vllm 部署调用](./models/Qwen3/02-Qwen3-8B-vLLM%20部署调用.md) @李娇娇
  - [x] [Qwen3-8B Windows LMStudio 部署调用](./models/Qwen3/03-Qwen3-7B-Instruct%20Windows%20LMStudio%20部署.md) @王熠明
  - [x] [Qwen3-8B Evalscope 智商情商评测](./models/Qwen3/04-Qwen3-8B%20EvalScope智商情商评测.md) @李娇娇
  - [x] [Qwen3-8B Lora 微调及SwanLab 可视化记录](./models/Qwen3/05-Qwen3-8B-LoRA及SwanLab可视化记录.md) @姜舒凡
  - [x] [Qwen3-30B-A3B 微调及SwanLab 可视化记录](./models/Qwen3/06-Qwen3-30B-A3B%20微调及%20SwanLab%20可视化记录.md) @高立业
  - [x] [Qwen3 Think 解密 Blog](./models/Qwen3/07-Qwen3-Think-解密-Blog.md) @樊奇
  - [x] [Qwen3-8B Docker 镜像](https://www.codewithgpu.com/i/datawhalechina/self-llm/Qwen3) @高立业
  - [x] [Qwen3-0.6B 的小模型有什么用](./models//Qwen3/08-Qwen3_0_6B的小模型有什么用.md) @不要葱姜蒜
  - [x] [Qwen3-1.7B 医学推理式对话微调 及 SwanLab 可视化记录](./models/Qwen3/09-Qwen3-1.7B-医学推理式对话微调%20及%20SwanLab%20可视化记录.md) @林泽毅
  - [x] [Qwen3-8B GRPO微调及通过swanlab可视化](./models/Qwen3/10-Qwen3-8B%20GRPO微调及通过swanlab可视化.md) @郭宣伯
  - [X] [Qwen3-8B-AMD部署调用](./models/Qwen3/11-Qwen3-8B-AMD部署调用.md) @陈榆

### Kimi

[Kimi](https://github.com/MoonshotAI/Kimi-VL)
  - [x] [Kimi-VL-A3B 技术报告解读](./models/Kimi-VL/02-Kimi-VL-技术报告解读.md) @王泽宇
  - [x] [Kimi-VL-A3B-Thinking WebDemo 部署（网页对话助手）](./models/Kimi-VL/01-Kimi-VL-对话助手.md) @姜舒凡

### Llama4

[Llama4](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
  - [x] [Llama4 对话助手](./models/Llama4/01-Llama4-对话助手/01-Llama4-对话助手.md) @姜舒凡

### SpatialLM

[SpatialLM](https://github.com/manycore-research/SpatialLM)
  - [x] [SpatialLM 3D点云理解与目标检测模型部署](./models/SpatialLM/readme.md) @王泽宇

### Hunyuan3D-2

[Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2)
  - [x] [Hunyuan3D-2 系列模型部署](./models/Hunyuan3D-2/01-Hunyuan3D-2%20系列模型部署.md) @林恒宇
  - [x] [Hunyuan3D-2 系列模型代码调用](./models/Hunyuan3D-2/02-Hunyuan3D-2%20系列模型代码调用.md) @林恒宇
  - [x] [Hunyuan3D-2 系列模型Gradio部署](./models/Hunyuan3D-2/03-Hunyuan3D-2%20系列模型Gradio部署.md) @林恒宇
  - [x] [Hunyuan3D-2 系列模型API Server](./models/Hunyuan3D-2/04-Hunyuan3D-2%20系列模型API%20Server.md) @林恒宇
  - [x] [Hunyuan3D-2 Docker 镜像](https://www.codewithgpu.com/i/datawhalechina/self-llm/Hunyuan3D-2) @林恒宇

### Gemma3

[Gemma3](https://huggingface.co/google/gemma-3-4b-it)
  - [x] [gemma-3-4b-it FastApi 部署调用](./models/Gemma3/01-gemma-3-4b-it%20FastApi%20部署调用.md) @杜森
  - [x] [gemma-3-4b-it ollama + open-webui部署](./models/Gemma3/03-gemma-3-4b-it-ollama%20+%20open-webui部署.md) @孙超
  - [x] [gemma-3-4b-it evalscope 智商情商评测](./models/Gemma3/04-Gemma3-4b%20%20evalscope智商情商评测.md) @张龙斐
  - [x] [gemma-3-4b-it Lora 微调](./models/Gemma3/05-gemma-3-4b-it%20LoRA.md) @荞麦
  - [x] [gemma-3-4b-it Docker 镜像](https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-gemma3) @姜舒凡
  - [x] [gemma-3-4b-it GRPO微调及通过swanlab可视化](./models/Gemma3/6-gemma3-4B-itGRPO微调及通过swanlab可视化.md) @郭宣伯

### DeepSeek-R1-Distill

[DeepSeek-R1-Distill](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
  - [x] [DeepSeek-R1-Distill-Qwen-7B FastApi 部署调用](./models/DeepSeek-R1-Distill-Qwen/01-DeepSeek-R1-Distill-Qwen-7B%20FastApi%20部署调用.md) @骆秀韬
  - [x] [DeepSeek-R1-Distill-Qwen-7B Langchain 接入](./models/DeepSeek-R1-Distill-Qwen/02-DeepSeek-R1-Distill-Qwen-7B%20Langchain%20接入.md) @骆秀韬
  - [x] [DeepSeek-R1-Distill-Qwen-7B WebDemo 部署](./models/DeepSeek-R1-Distill-Qwen/03-DeepSeek-R1-Distill-Qwen-7B%20WebDemo%20部署.md) @骆秀韬
  - [x] [DeepSeek-R1-Distill-Qwen-7B vLLM 部署调用](./models/DeepSeek-R1-Distill-Qwen/04-DeepSeek-R1-Distill-Qwen-7B%20vLLM%20部署调用.md) @骆秀韬
  - [x] [DeepSeek-R1-0528-Qwen3-8B-GRPO及swanlab可视化](./models/DeepSeek-R1-Distill-Qwen/05-DeepSeek-R1-0528-Qwen3-8B-GRPO及swanlab可视化.md) @郭宣伯

### MiniCPM-o-2_6

[MiniCPM-o-2_6](https://github.com/OpenBMB/MiniCPM-o)
  - [x] [minicpm-o-2.6 FastApi 部署调用](./models/MiniCPM-o/01MiniCPM-o%202%206%20FastApi部署调用%20.md) @林恒宇
  - [x] [minicpm-o-2.6 WebDemo 部署](./models/MiniCPM-o/02minicpm-o-2.6WebDemo_streamlit.py) @程宏
  - [x] [minicpm-o-2.6 多模态语音能力](./models/MiniCPM-o/03-MiniCPM-o-2.6%20多模态语音能力.md) @邓恺俊
  - [x] [minicpm-o-2.6 可视化 LaTeX_OCR Lora 微调](./models/MiniCPM-o/04-MiniCPM-0-2.6%20Lora微调.md) @林泽毅

### InternLM3

[InternLM3](https://github.com/InternLM/InternLM)
  - [x] [internlm3-8b-instruct FastApi 部署调用](./models/InternLM3/01-InternLM3-8B-Instruct%20FastAPI.md) @苏向标
  - [x] [internlm3-8b-instruct Langchian接入](./models/InternLM3/02-internlm3-8b-Instruct%20Langchain%20接入.md) @赵文恺
  - [x] [internlm3-8b-instruct WebDemo 部署](./models/InternLM3/03-InternLM3-8B-Instruct%20WebDemo部署.md) @王泽宇
  - [x] [internlm3-8b-instruct Lora 微调](./models/InternLM3/04-InternLM3-8B-Instruct%20LoRA.md) @程宏
  - [x] [internlm3-8b-instruct o1-like推理链实现](./models/InternLM3/05-internlm3-8b-instruct%20与o1%20.md) @陈睿

### phi4

[phi4](https://huggingface.co/microsoft/phi-4)
  - [x] [phi4 FastApi 部署调用](./models/phi4/01-Phi-4%20FastApi%20部署调用.md) @杜森
  - [x] [phi4 langchain 接入](./models/phi4/02-Phi-4-Langchain接入.md) @小罗
  - [x] [phi4 WebDemo 部署](./models/phi4/03-Phi-4%20WebDemo部署.md) @杜森
  - [x] [phi4 Lora 微调](./models/phi4/04-Phi-4-Lora%20微调.md) @郑远婧
  - [x] [phi4 Lora 微调 NER任务 SwanLab 可视化记录版](./models/phi4/05-Phi-4-Lora%20微调%20命名实体识别.md) @林泽毅
  - [x] [phi4 GRPO微调及通过swanlab可视化](./models/phi4/06-Phi-4-GRPO及swanlab可视化.md) @郭宣伯

### Qwen2.5-Coder

[Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder)
  - [x] [Qwen2.5-Coder-7B-Instruct FastApi部署调用](./models/Qwen2.5-Coder/01-Qwen2.5-Coder-7B-Instruct%20FastApi%20部署调用.md) @赵文恺
  - [x] [Qwen2.5-Coder-7B-Instruct Langchian接入](./models/Qwen2.5-Coder/02-Qwen2.5-7B-Instruct%20Langchain%20接入.md) @杨晨旭
  - [x] [Qwen2.5-Coder-7B-Instruct WebDemo 部署](./models/Qwen2.5-Coder/03-Qwen2.5-Coder-7B-Instruct%20WebDemo部署.md) @王泽宇
  - [x] [Qwen2.5-Coder-7B-Instruct vLLM 部署](./models/Qwen2.5-Coder/04-Qwen2.5-Coder-7B-Instruct%20vLLM%20部署调用.md) @王泽宇
  - [x] [Qwen2.5-Coder-7B-Instruct Lora 微调](./models/Qwen2.5-Coder/Qwen2.5-Coder-7B-Instruct%20Lora%20微调.md) @荞麦
  - [x] [Qwen2.5-Coder-7B-Instruct Lora 微调 SwanLab 可视化记录版](./models/Qwen2.5-Coder/05-Qwen2.5-Coder-7B-Instruct%20Lora%20微调%20SwanLab%20可视化记录版.md) @杨卓

### Qwen2-vl

[Qwen2-vl](https://github.com/QwenLM/Qwen2-VL)
  - [x] [Qwen2-vl-2B FastApi 部署调用](./models/Qwen2-VL/01-Qwen2-VL-2B-Instruct%20FastApi%20部署调用.md) @姜舒凡
  - [x] [Qwen2-vl-2B WebDemo 部署](./models/Qwen2-VL/02-Qwen2-VL-2B-Instruct%20Web%20Demo部署.md) @赵伟
  - [x] [Qwen2-vl-2B vLLM 部署](./models/Qwen2-VL/03-Qwen2-VL-2B-Instruct%20vLLM部署调用.md) @荞麦
  - [x] [Qwen2-vl-2B Lora 微调](./models/Qwen2-VL/04-Qwen2-VL-2B%20Lora%20微调.md) @李柯辰
  - [x] [Qwen2-vl-2B Lora 微调 SwanLab 可视化记录版](./models/Qwen2-VL/05-Qwen2-VL-2B-Instruct%20Lora%20微调%20SwanLab%20可视化记录版.md) @林泽毅
  - [x] [Qwen2-vl-2B Lora 微调案例 - LaTexOCR](./models/Qwen2-VL/06-Qwen2-VL-2B-Instruct%20Lora%20微调案例%20-%20LaTexOCR.md) @林泽毅

### Qwen2.5

[Qwen2.5](https://github.com/QwenLM/Qwen2.5)
  - [x] [Qwen2.5-7B-Instruct FastApi 部署调用](./models/Qwen2.5/01-Qwen2.5-7B-Instruct%20FastApi%20部署调用.md) @娄天奥
  - [x] [Qwen2.5-7B-Instruct langchain 接入](./models/Qwen2.5/02-Qwen2.5-7B-Instruct%20Langchain%20接入.md) @娄天奥
  - [x] [Qwen2.5-7B-Instruct vLLM 部署调用](./models/Qwen2.5/03-Qwen2.5-7B-Instruct%20vLLM%20部署调用.md) @姜舒凡
  - [x] [Qwen2.5-7B-Instruct WebDemo 部署](./models/Qwen2.5/04-Qwen2_5-7B-Instruct%20WebDemo部署.md) @高立业
  - [x] [Qwen2.5-7B-Instruct Lora 微调](./models/Qwen2.5/05-Qwen2.5-7B-Instruct%20Lora%20微调.md) @左春生
  - [x] [Qwen2.5-7B-Instruct o1-like 推理链实现](./models/Qwen2.5/06-Qwen2.5-7B-Instruct%20o1-like%20推理链实现.md) @姜舒凡
  - [x] [Qwen2.5-7B-Instruct Lora 微调 SwanLab 可视化记录版](./models/Qwen2.5/07-Qwen2.5-7B-Instruct%20Lora%20微调%20SwanLab可视化记录版.md) @林泽毅

### Apple OpenELM

[Apple OpenELM](https://machinelearning.apple.com/research/openelm)
  - [x] [OpenELM-3B-Instruct FastApi 部署调用](./models/OpenELM/01-OpenELM-3B-Instruct%20FastApi部署调用.md) @王泽宇
  - [x] [OpenELM-3B-Instruct Lora 微调](./models/OpenELM/02-OpenELM-3B-Instruct%20Lora微调.md) @王泽宇

### Llama3_1-8B-Instruct

[Llama3_1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  - [x] [Llama3_1-8B-Instruct FastApi 部署调用](./models/Llama3_1/01-Llama3_1-8B-Instruct%20FastApi%20部署调用.md) @不要葱姜蒜
  - [x] [Llama3_1-8B-Instruct langchain 接入](./models/Llama3_1/02-Llama3_1-8B-Instruct%20langchain接入.md) @张晋
  - [x] [Llama3_1-8B-Instruct WebDemo 部署](./models/Llama3_1/03-Llama3_1-8B-Instruct%20WebDemo部署.md) @张晋
  - [x] [Llama3_1-8B-Instruct Lora 微调](./models/Llama3_1/04-Llama3_1-8B--Instruct%20Lora%20微调.md) @不要葱姜蒜
  - [x] [动手转换GGUF模型并使用Ollama本地部署](./models/Llama3_1/动手转换GGUF模型并使用Ollama本地部署.md) @Gaoboy

### Gemma-2-9b-it

[Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
  - [x] [Gemma-2-9b-it FastApi 部署调用](./models/Gemma2/01-Gemma-2-9b-it%20FastApi%20部署调用.md) @不要葱姜蒜
  - [x] [Gemma-2-9b-it langchain 接入](./models/Gemma2/02-Gemma-2-9b-it%20langchain%20接入.md) @不要葱姜蒜
  - [x] [Gemma-2-9b-it WebDemo 部署](./models/Gemma2/03-Gemma-2-9b-it%20WebDemo%20部署.md) @不要葱姜蒜
  - [x] [Gemma-2-9b-it Peft Lora 微调](./models/Gemma2/04-Gemma-2-9b-it%20peft%20lora微调.md) @不要葱姜蒜

### Yuan2.0

[Yuan2.0](https://github.com/IEIT-Yuan/Yuan-2.0)
  - [x] [Yuan2.0-2B FastApi 部署调用](./models/Yuan2.0/01-Yuan2.0-2B%20FastApi%20部署调用.md) @张帆
  - [x] [Yuan2.0-2B Langchain 接入](./models/Yuan2.0/02-Yuan2.0-2B%20Langchain%20接入.md) @张帆
  - [x] [Yuan2.0-2B WebDemo部署](./models/Yuan2.0/03-Yuan2.0-2B%20WebDemo部署.md) @张帆
  - [x] [Yuan2.0-2B vLLM部署调用](./models/Yuan2.0/04-Yuan2.0-2B%20vLLM部署调用.md) @张帆
  - [x] [Yuan2.0-2B Lora微调](./models/Yuan2.0/05-Yuan2.0-2B%20Lora微调.md) @张帆

### Yuan2.0-M32

[Yuan2.0-M32](https://github.com/IEIT-Yuan/Yuan2.0-M32)
  - [x] [Yuan2.0-M32 FastApi 部署调用](./models/Yuan2.0-M32/01-Yuan2.0-M32%20FastApi%20部署调用.md) @张帆
  - [x] [Yuan2.0-M32 Langchain 接入](./models/Yuan2.0-M32/02-Yuan2.0-M32%20Langchain%20接入.md) @张帆
  - [x] [Yuan2.0-M32 WebDemo部署](./models/Yuan2.0-M32/03-Yuan2.0-M32%20WebDemo部署.md) @张帆

### DeepSeek-Coder-V2

[DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2)
  - [x] [DeepSeek-Coder-V2-Lite-Instruct FastApi 部署调用](./models/DeepSeek-Coder-V2/01-DeepSeek-Coder-V2-Lite-Instruct%20FastApi%20部署调用.md) @姜舒凡
  - [x] [DeepSeek-Coder-V2-Lite-Instruct langchain 接入](./models/DeepSeek-Coder-V2/02-DeepSeek-Coder-V2-Lite-Instruct%20接入%20LangChain.md) @姜舒凡
  - [x] [DeepSeek-Coder-V2-Lite-Instruct WebDemo 部署](./models/DeepSeek-Coder-V2/03-DeepSeek-Coder-V2-Lite-Instruct%20WebDemo%20部署.md) @Kailigithub
  - [x] [DeepSeek-Coder-V2-Lite-Instruct Lora 微调](./models/DeepSeek-Coder-V2/04-DeepSeek-Coder-V2-Lite-Instruct%20Lora%20微调.md) @余洋

### 哔哩哔哩 Index-1.9B

[哔哩哔哩 Index-1.9B](https://github.com/bilibili/Index-1.9B)
  - [x] [Index-1.9B-Chat FastApi 部署调用](./models/bilibili_Index-1.9B/01-Index-1.9B-chat%20FastApi%20部署调用.md) @邓恺俊
  - [x] [Index-1.9B-Chat langchain 接入](./models/bilibili_Index-1.9B/02-Index-1.9B-Chat%20接入%20LangChain.md) @张友东
  - [x] [Index-1.9B-Chat WebDemo 部署](./models/bilibili_Index-1.9B/03-Index-1.9B-chat%20WebDemo部署.md) @程宏
  - [x] [Index-1.9B-Chat Lora 微调](./models/bilibili_Index-1.9B/04-Index-1.9B-Chat%20Lora%20微调.md) @姜舒凡

### Qwen2

[Qwen2](https://github.com/QwenLM/Qwen2)
  - [x] [Qwen2-7B-Instruct FastApi 部署调用](./models/Qwen2/01-Qwen2-7B-Instruct%20FastApi%20部署调用.md) @康婧淇
  - [x] [Qwen2-7B-Instruct langchain 接入](./models/Qwen2/02-Qwen2-7B-Instruct%20Langchain%20接入.md) @不要葱姜蒜
  - [x] [Qwen2-7B-Instruct WebDemo 部署](./models/Qwen2/03-Qwen2-7B-Instruct%20WebDemo部署.md) @三水
  - [x] [Qwen2-7B-Instruct vLLM 部署调用](./models/Qwen2/04-Qwen2-7B-Instruct%20vLLM%20部署调用.md) @姜舒凡
  - [x] [Qwen2-7B-Instruct Lora 微调](./models/Qwen2/05-Qwen2-7B-Instruct%20Lora%20微调.md) @散步

### GLM-4

[GLM-4](https://github.com/THUDM/GLM-4.git)
  - [x] [GLM-4-9B-chat FastApi 部署调用](./models/GLM-4/01-GLM-4-9B-chat%20FastApi%20部署调用.md) @张友东
  - [x] [GLM-4-9B-chat langchain 接入](./models/GLM-4/02-GLM-4-9B-chat%20langchain%20接入.md) @谭逸珂
  - [x] [GLM-4-9B-chat WebDemo 部署](./models/GLM-4/03-GLM-4-9B-Chat%20WebDemo.md) @何至轩
  - [x] [GLM-4-9B-chat vLLM 部署](./models/GLM-4/04-GLM-4-9B-Chat%20vLLM%20部署调用.md) @王熠明
  - [x] [GLM-4-9B-chat Lora 微调](./models/GLM-4/05-GLM-4-9B-chat%20Lora%20微调.md) @肖鸿儒
  - [x] [GLM-4-9B-chat-hf Lora 微调](./models/GLM-4/05-GLM-4-9B-chat-hf%20Lora%20微调.md) @付志远


### Qwen 1.5

[Qwen 1.5](https://github.com/QwenLM/Qwen1.5.git)
  - [x] [Qwen1.5-7B-chat FastApi 部署调用](./models/Qwen1.5/01-Qwen1.5-7B-Chat%20FastApi%20部署调用.md) @颜鑫
  - [x] [Qwen1.5-7B-chat langchain 接入](./models/Qwen1.5/02-Qwen1.5-7B-Chat%20接入langchain搭建知识库助手.md) @颜鑫
  - [x] [Qwen1.5-7B-chat WebDemo 部署](./models/Qwen1.5/03-Qwen1.5-7B-Chat%20WebDemo.md) @颜鑫
  - [x] [Qwen1.5-7B-chat Lora 微调](./models/Qwen1.5/04-Qwen1.5-7B-chat%20Lora%20微调.md) @不要葱姜蒜
  - [x] [Qwen1.5-72B-chat-GPTQ-Int4 部署环境](./models/Qwen1.5/05-Qwen1.5-7B-Chat-GPTQ-Int4%20%20WebDemo.md) @byx020119
  - [x] [Qwen1.5-MoE-chat Transformers 部署调用](./models/Qwen1.5/06-Qwen1.5-MoE-A2.7B.md) @丁悦
  - [x] [Qwen1.5-7B-chat vLLM推理部署](./models/Qwen1.5/07-Qwen1.5-7B-Chat%20vLLM%20推理部署调用.md) @高立业
  - [x] [Qwen1.5-7B-chat Lora 微调 接入SwanLab实验管理平台](./models/Qwen1.5/08-Qwen1.5-7B-chat%20LoRA微调接入实验管理.md) @黄柏特

### phi-3

[phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
  - [x] [Phi-3-mini-4k-instruct FastApi 部署调用](./models/phi-3/01-Phi-3-mini-4k-instruct%20FastApi%20部署调用.md) @郑皓桦
  - [x] [Phi-3-mini-4k-instruct langchain 接入](./models/phi-3/02-Phi-3-mini-4k-instruct%20langchain%20接入.md) @郑皓桦
  - [x] [Phi-3-mini-4k-instruct WebDemo 部署](./models/phi-3/03-Phi-3-mini-4k-instruct%20WebDemo部署.md) @丁悦
  - [x] [Phi-3-mini-4k-instruct Lora 微调](./models/phi-3/04-Phi-3-mini-4k-Instruct%20Lora%20微调.md) @丁悦

### CharacterGLM-6B

[CharacterGLM-6B](https://github.com/thu-coai/CharacterGLM-6B)
  - [x] [CharacterGLM-6B Transformers 部署调用](./models/CharacterGLM/01-CharacterGLM-6B%20Transformer部署调用.md) @孙健壮
  - [x] [CharacterGLM-6B FastApi 部署调用](./models/CharacterGLM/02-CharacterGLM-6B%20FastApi部署调用.md) @孙健壮
  - [x] [CharacterGLM-6B webdemo 部署](./models/CharacterGLM/03-CharacterGLM-6B-chat.md) @孙健壮
  - [x] [CharacterGLM-6B Lora 微调](./models/CharacterGLM/04-CharacterGLM-6B%20Lora微调.md) @孙健壮

### LLaMA3-8B-Instruct

[LLaMA3-8B-Instruct](https://github.com/meta-llama/llama3.git)
  - [x] [LLaMA3-8B-Instruct FastApi 部署调用](./models/LLaMA3/01-LLaMA3-8B-Instruct%20FastApi%20部署调用.md) @高立业
  - [X] [LLaMA3-8B-Instruct langchain 接入](./models/LLaMA3/02-LLaMA3-8B-Instruct%20langchain%20接入.md) @不要葱姜蒜
  - [x] [LLaMA3-8B-Instruct WebDemo 部署](./models/LLaMA3/03-LLaMA3-8B-Instruct%20WebDemo%20部署.md) @不要葱姜蒜
  - [x] [LLaMA3-8B-Instruct Lora 微调](./models/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20微调.md) @高立业

### XVERSE-7B-Chat

[XVERSE-7B-Chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary)
  - [x] [XVERSE-7B-Chat transformers 部署调用](./models/XVERSE/01-XVERSE-7B-chat%20Transformers推理.md) @郭志航
  - [x] [XVERSE-7B-Chat FastApi 部署调用](./models/XVERSE/02-XVERSE-7B-chat%20FastAPI部署.md) @郭志航
  - [x] [XVERSE-7B-Chat langchain 接入](./models/XVERSE/03-XVERSE-7B-chat%20langchain%20接入.md) @郭志航
  - [x] [XVERSE-7B-Chat WebDemo 部署](./models/XVERSE/04-XVERSE-7B-chat%20WebDemo%20部署.md) @郭志航
  - [x] [XVERSE-7B-Chat Lora 微调](./models/XVERSE/05-XVERSE-7B-Chat%20Lora%20微调.md) @郭志航

### TransNormerLLM

[TransNormerLLM](https://github.com/OpenNLPLab/TransnormerLLM.git)
  - [X] [TransNormerLLM-7B-Chat FastApi 部署调用](./models/TransNormer/01-TransNormer-7B%20FastApi%20部署调用.md) @王茂霖
  - [X] [TransNormerLLM-7B-Chat langchain 接入](./models/TransNormer/02-TransNormer-7B%20接入langchain搭建知识库助手.md) @王茂霖
  - [X] [TransNormerLLM-7B-Chat WebDemo 部署](./models/TransNormer/03-TransNormer-7B%20WebDemo.md) @王茂霖
  - [x] [TransNormerLLM-7B-Chat Lora 微调](./models/TransNormer/04-TrasnNormer-7B%20Lora%20微调.md) @王茂霖

### BlueLM Vivo 蓝心大模型

[BlueLM Vivo 蓝心大模型](https://github.com/vivo-ai-lab/BlueLM.git)
  - [x] [BlueLM-7B-Chat FatApi 部署调用](./models/BlueLM/01-BlueLM-7B-Chat%20FastApi%20部署.md) @郭志航
  - [x] [BlueLM-7B-Chat langchain 接入](./models/BlueLM/02-BlueLM-7B-Chat%20langchain%20接入.md) @郭志航
  - [x] [BlueLM-7B-Chat WebDemo 部署](./models/BlueLM/03-BlueLM-7B-Chat%20WebDemo%20部署.md) @郭志航
  - [x] [BlueLM-7B-Chat Lora 微调](./models/BlueLM/04-BlueLM-7B-Chat%20Lora%20微调.md) @郭志航

### InternLM2

[InternLM2](https://github.com/InternLM/InternLM)
  - [x] [InternLM2-7B-chat FastApi 部署调用](./models/InternLM2/01-InternLM2-7B-chat%20FastAPI部署.md) @不要葱姜蒜
  - [x] [InternLM2-7B-chat langchain 接入](./models/InternLM2/02-InternLM2-7B-chat%20langchain%20接入.md) @不要葱姜蒜
  - [x] [InternLM2-7B-chat WebDemo 部署](./models/InternLM2/03-InternLM2-7B-chat%20WebDemo%20部署.md) @郑皓桦
  - [x] [InternLM2-7B-chat Xtuner Qlora 微调](./models/InternLM2/04-InternLM2-7B-chat%20Xtuner%20Qlora%20微调.md) @郑皓桦

### DeepSeek 深度求索

[DeepSeek 深度求索](https://github.com/deepseek-ai/DeepSeek-LLM)
  - [x] [DeepSeek-7B-chat FastApi 部署调用](./models/DeepSeek/01-DeepSeek-7B-chat%20FastApi.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat langchain 接入](./models/DeepSeek/02-DeepSeek-7B-chat%20langchain.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat WebDemo](./models/DeepSeek/03-DeepSeek-7B-chat%20WebDemo.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat Lora 微调](./models/DeepSeek/04-DeepSeek-7B-chat%20Lora%20微调.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat 4bits量化 Qlora 微调](./models/DeepSeek/05-DeepSeek-7B-chat%204bits量化%20Qlora%20微调.md) @不要葱姜蒜
  - [x] [DeepSeek-MoE-16b-chat Transformers 部署调用](./models/DeepSeek/06-DeepSeek-MoE-16b-chat%20Transformer部署调用.md) @Kailigithub
  - [x] [DeepSeek-MoE-16b-chat FastApi 部署调用](./models/DeepSeek/06-DeepSeek-MoE-16b-chat%20FastApi.md) @Kailigithub
  - [x] [DeepSeek-coder-6.7b finetune colab](./models/DeepSeek/07-deepseek_fine_tune.ipynb) @Swiftie
  - [x] [Deepseek-coder-6.7b webdemo colab](./models/DeepSeek/08-deepseek_web_demo.ipynb) @Swiftie

### MiniCPM

[MiniCPM](https://github.com/OpenBMB/MiniCPM.git)
  - [x] [MiniCPM-2B-chat transformers 部署调用](./models/MiniCPM/MiniCPM-2B-chat%20transformers%20部署调用.md) @Kailigithub
  - [x] [MiniCPM-2B-chat FastApi 部署调用](./models/MiniCPM/MiniCPM-2B-chat%20FastApi%20部署调用.md) @Kailigithub
  - [x] [MiniCPM-2B-chat langchain 接入](./models/MiniCPM/MiniCPM-2B-chat%20langchain接入.md) @不要葱姜蒜
  - [x] [MiniCPM-2B-chat webdemo 部署](./models/MiniCPM/MiniCPM-2B-chat%20WebDemo部署.md) @Kailigithub
  - [x] [MiniCPM-2B-chat Lora && Full 微调](./models/MiniCPM/MiniCPM-2B-chat%20Lora%20&&%20Full%20微调.md) @不要葱姜蒜
  - [x] 官方友情链接：[面壁小钢炮MiniCPM教程](https://modelbest.feishu.cn/wiki/D2tFw8Pcsi5CIzkaHNacLK64npg) @OpenBMB
  - [x] 官方友情链接：[MiniCPM-Cookbook](https://github.com/OpenBMB/MiniCPM-CookBook) @OpenBMB

### Qwen-Audio

[Qwen-Audio](https://github.com/QwenLM/Qwen-Audio.git)
  - [x] [Qwen-Audio FastApi 部署调用](./models/Qwen-Audio/01-Qwen-Audio-chat%20FastApi.md) @陈思州
  - [x] [Qwen-Audio WebDemo](./models/Qwen-Audio/02-Qwen-Audio-chat%20WebDemo.md) @陈思州

### Qwen

[Qwen](https://github.com/QwenLM/Qwen.git)
  - [x] [Qwen-7B-chat Transformers 部署调用](./models/Qwen/01-Qwen-7B-Chat%20Transformers部署调用.md) @李娇娇
  - [x] [Qwen-7B-chat FastApi 部署调用](./models/Qwen/02-Qwen-7B-Chat%20FastApi%20部署调用.md) @李娇娇
  - [x] [Qwen-7B-chat WebDemo](./models/Qwen/03-Qwen-7B-Chat%20WebDemo.md) @李娇娇
  - [x] [Qwen-7B-chat Lora 微调](./models/Qwen/04-Qwen-7B-Chat%20Lora%20微调.md) @不要葱姜蒜
  - [x] [Qwen-7B-chat ptuning 微调](./models/Qwen/05-Qwen-7B-Chat%20Ptuning%20微调.md) @肖鸿儒
  - [x] [Qwen-7B-chat 全量微调](./models/Qwen/06-Qwen-7B-chat%20全量微调.md) @不要葱姜蒜
  - [x] [Qwen-7B-Chat 接入langchain搭建知识库助手](./models/Qwen/07-Qwen-7B-Chat%20接入langchain搭建知识库助手.md) @李娇娇
  - [x] [Qwen-7B-chat 低精度训练](./models/Qwen/08-Qwen-7B-Chat%20Lora%20低精度微调.md) @肖鸿儒
  - [x] [Qwen-1_8B-chat CPU 部署](./models/Qwen/09-Qwen-1_8B-chat%20CPU%20部署%20.md) @散步

### Yi 零一万物

[Yi 零一万物](https://github.com/01-ai/Yi.git)
  - [x] [Yi-6B-chat FastApi 部署调用](./models/Yi/01-Yi-6B-Chat%20FastApi%20部署调用.md) @李柯辰
  - [x] [Yi-6B-chat langchain接入](./models/Yi/02-Yi-6B-Chat%20接入langchain搭建知识库助手.md) @李柯辰
  - [x] [Yi-6B-chat WebDemo](./models/Yi/03-Yi-6B-chat%20WebDemo.md) @肖鸿儒
  - [x] [Yi-6B-chat Lora 微调](./models/Yi/04-Yi-6B-Chat%20Lora%20微调.md) @李娇娇

### Baichuan 百川智能

[Baichuan 百川智能](https://www.baichuan-ai.com/home)
  - [x] [Baichuan2-7B-chat FastApi 部署调用](./BaiChuan/01-Baichuan2-7B-chat%2BFastApi%2B%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md) @惠佳豪
  - [x] [Baichuan2-7B-chat WebDemo](./models/BaiChuan/02-Baichuan-7B-chat%2BWebDemo.md) @惠佳豪
  - [x] [Baichuan2-7B-chat 接入 LangChain 框架](./models/BaiChuan/03-Baichuan2-7B-chat%E6%8E%A5%E5%85%A5LangChain%E6%A1%86%E6%9E%B6.md) @惠佳豪
  - [x] [Baichuan2-7B-chat Lora 微调](./models/BaiChuan/04-Baichuan2-7B-chat%2Blora%2B%E5%BE%AE%E8%B0%83.md) @惠佳豪

### InternLM

[InternLM](https://github.com/InternLM/InternLM.git)
  - [x] [InternLM-Chat-7B Transformers 部署调用](./models/InternLM/01-InternLM-Chat-7B%20Transformers%20部署调用.md) @小罗
  - [x] [InternLM-Chat-7B FastApi 部署调用](./models/InternLM/02-internLM-Chat-7B%20FastApi.md) @不要葱姜蒜
  - [x] [InternLM-Chat-7B WebDemo](./models/InternLM/03-InternLM-Chat-7B.md) @不要葱姜蒜
  - [x] [Lagent+InternLM-Chat-7B-V1.1 WebDemo](./models/InternLM/04-Lagent+InternLM-Chat-7B-V1.1.md) @不要葱姜蒜
  - [x] [浦语灵笔图文理解&创作 WebDemo](./models/InternLM/05-浦语灵笔图文理解&创作.md) @不要葱姜蒜
  - [x] [InternLM-Chat-7B 接入 LangChain 框架](./models/InternLM/06-InternLM接入LangChain搭建知识库助手.md) @Logan Zou

### Atom (llama2)

[Atom (llama2)](https://hf-mirror.com/FlagAlpha/Atom-7B-Chat)
  - [x] [Atom-7B-chat WebDemo](./models/Atom/01-Atom-7B-chat-WebDemo.md) @Kailigithub
  - [x] [Atom-7B-chat Lora 微调](./models/Atom/02-Atom-7B-Chat%20Lora%20微调.md) @Logan Zou
  - [x] [Atom-7B-Chat 接入langchain搭建知识库助手](./models/Atom/03-Atom-7B-Chat%20接入langchain搭建知识库助手.md) @陈思州
  - [x] [Atom-7B-chat 全量微调](./models/Atom/04-Atom-7B-chat%20全量微调.md) @Logan Zou

### ChatGLM3

[ChatGLM3](https://github.com/THUDM/ChatGLM3.git)
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