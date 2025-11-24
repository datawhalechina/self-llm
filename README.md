<div align=center>
  <img src="./images/head-img.png" >
  <h1>Open Source Large Model User Guide</h1>
</div>

<div align="center">

[中文](./README.md) | English

</div>

&emsp;&emsp;This project is a large model tutorial tailored for domestic beginners, focusing on open-source large models and based on the Linux platform. It provides comprehensive guidance on environment configuration, local deployment, and efficient fine-tuning for various open-source large models. The goal is to simplify the deployment, usage, and application processes of open-source large models, enabling more students and researchers to effectively utilize these models and integrate open-source, freely available large models into their daily lives.

&emsp;&emsp;The main content of this project includes:

  1. A guide to configuring the environment for open-source LLMs on the Linux platform, offering detailed steps tailored to different model requirements;
  2. Deployment and usage tutorials for mainstream open-source LLMs, both domestic and international, including LLaMA, ChatGLM, InternLM, MiniCPM, and more;
  3. Guidance on deploying and applying open-source LLMs, covering command-line invocation, online demo deployment, and integration with the LangChain framework;
  4. Methods for full fine-tuning and efficient fine-tuning of open-source LLMs, including distributed full fine-tuning, LoRA, and ptuning.

&emsp;&emsp;**The main content of this project is tutorials, aimed at helping more students and future practitioners understand and master the usage of open-source large models! Anyone can submit issues or pull requests to contribute to the project.**

&emsp;&emsp;Students who wish to deeply participate can contact us, and we will add them as project maintainers.

> &emsp;&emsp;***Learning Suggestion: The recommended learning path for this project is to start with environment configuration, then move on to model deployment and usage, and finally tackle fine-tuning. Environment configuration is the foundation, model deployment and usage are the basics, and fine-tuning is the advanced step. Beginners are advised to start with models like Qwen1.5, InternLM2, and MiniCPM.***

> Note: For students interested in understanding the architecture of large models and learning to hand-write RAG, Agent, and Eval tasks from scratch, they can refer to another Datawhale project, [Tiny-Universe](https://github.com/datawhalechina/tiny-universe). Large models are a hot topic in the field of deep learning, but most existing tutorials focus on teaching how to call APIs for large model applications, with few explaining the model structure, RAG, Agent, and Eval from a theoretical perspective. This repository provides a completely hand-written approach, without using APIs, to complete RAG, Agent, and Eval tasks for large models.

> Note: For students who wish to learn the theoretical aspects of large models before diving into this project, they can refer to Datawhale's [so-large-llm](https://github.com/datawhalechina/so-large-lm.git) course to gain a deeper understanding of LLM theory and its applications.

> Note: For students who want to develop large model applications after completing this course, they can refer to Datawhale's [Hands-On Large Model Application Development](https://github.com/datawhalechina/llm-universe) course. This project is a tutorial for beginner developers, aiming to present the complete large model application development process based on Alibaba Cloud servers and a personal knowledge base assistant project.

## Project Significance

&emsp;&emsp;What is a large model?

> A large model (LLM) narrowly refers to a natural language processing (NLP) model trained based on deep learning algorithms, primarily used in natural language understanding and generation. Broadly, it also includes computer vision (CV) large models, multimodal large models, and scientific computing large models.

&emsp;&emsp;The battle of a hundred models is in full swing, with open-source LLMs emerging one after another. Numerous excellent open-source LLMs have appeared both domestically and internationally, such as LLaMA and Alpaca abroad, and ChatGLM, BaiChuan, and InternLM (Scholar·Puyu) in China. Open-source LLMs support local deployment and private domain fine-tuning, allowing everyone to create their own unique large models based on open-source LLMs.

&emsp;&emsp;However, for ordinary students and users, using these large models requires a certain level of technical expertise to complete the deployment and usage. With the continuous emergence of diverse open-source LLMs, quickly mastering the application methods of an open-source LLM is a challenging task.

&emsp;&emsp;This project aims to first provide deployment, usage, and fine-tuning tutorials for mainstream open-source LLMs based on the core contributors' experience. After completing the relevant sections for mainstream LLMs, we hope to gather more collaborators to enrich this open-source LLM world, creating tutorials for more and more unique LLMs. Sparks will gather into a sea.

&emsp;&emsp;***We hope to become the bridge between LLMs and the general public, embracing a broader and more expansive LLM world with the spirit of freedom and equality in open source.***

## Target Audience

&emsp;&emsp;This project is suitable for the following learners:

* Those who want to use or experience LLMs but lack access to or cannot use related APIs;
* Those who wish to apply LLMs in large quantities over the long term at a low cost;
* Those interested in open-source LLMs and want to get hands-on experience;
* NLP students who wish to further their understanding of LLMs;
* Those who want to combine open-source LLMs to create domain-specific private LLMs;
* And the broadest, most ordinary student population.

## Project Plan and Progress

&emsp;&emsp;This project is organized around the entire application process of open-source LLMs, including environment configuration and usage, deployment applications, and fine-tuning. Each section covers mainstream and unique open-source LLMs:

### Example Series

- [Chat-Huanhuan](./examples/Chat-嬛嬛/readme.md): Chat-Huanhuan is a chat language model that mimics the tone of Zhen Huan, fine-tuned using LoRA based on all the lines and dialogues related to Zhen Huan from the script of "Empresses in the Palace."

- [Tianji-Sky Machine](./examples/Tianji-天机/readme.md): Tianji is a large language model system application tutorial based on social scenarios of human relationships, covering prompt engineering, agent creation, data acquisition and model fine-tuning, RAG data cleaning and usage, and more.

### Supported Models

- [MiniMax-M2](https://github.com/MiniMax-AI/MiniMax-M2)
  - [x] [MiniMax-M2 Online Experience](https://agent.minimax.io/)
  - [x] [MiniMax-M2 Hugging Face](https://huggingface.co/MiniMaxAI/MiniMax-M2)
  - [x] [MiniMax-M2 Text Generation Guide](https://platform.minimax.io/docs/guides/text-generation)
  - [x] [MiniMax-M2 Model Architecture Analysis Blog](./models/MiniMax-M2/1-MiniMax-M2-Blog.md) @Wang Zeyu
  - [x] [MiniMax-M2 vLLM Deployment](./models/MiniMax-M2/2-MiniMax-M2-vLLM.md) @Jiang Shufan
  - [x] [MiniMax-M2 SGLang Deployment](./models/MiniMax-M2/2-MiniMax-M2-vLLM.md) @Jiang Shufan
  - [x] [MiniMax-M2 EvalScope IQ/EQ & Concurrent Evaluation](./models/MiniMax-M2/4-MiniMax-M2-EvalScope.md) @Jiang Shufan
  - [x] [AutoDL MiniMax-M2 vLLM Deployment & EvalScope Image](https://www.codewithgpu.com/i/datawhalechina/self-llm/mimimax-m2) @Jiang Shufan

- [Qwen3-VL-4B-Instruct](https://github.com/QwenLM/Qwen3-VL)
  - [x] [Qwen3-VL Model Architecture Analysis (DeepStack)](./models/Qwen3-VL/01-Qwen3-VL-MoE-模型结构解析-Blog.md) @Wang Zeyu
  - [x] [Qwen3-VL-4B-Instruct FastApi Deployment](./models/Qwen3-VL/02-Qwen3-VL-4B-Instruct%20FastApi%20部署调用.md) @Wang Jiapeng
  - [x] [Qwen3-VL-4B-Instruct vLLM Deployment](./models/Qwen3-VL/04-Qwen3-VL-4B-Instruct-vLLM.md) @Jiang Shufan
  - [x] [Qwen3-VL-4B-Instruct Lora Visual Fine-Tuning Case - LaTexOCR](./models/Qwen3-VL/05-Qwen3-VL-4B-Instruct%20%20Lora%20可视化微调案例%20-%20LaTexOCR.md) @Li Xiuqi

- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
  - [x] [Code Retrieval Fine-Tuning: BGE-M3 Embedding Model](./models/BGE-M3-finetune-embedding-with-valid/README.md) @Li Xiuqi 

- [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
  - [x] [gpt-oss-20b vLLM Deployment](./models/gpt-oss/1-gpt-oss-20b%20vllm%20部署调用.md) @Guo Xuanbo
  - [x] [gpt-oss-20b EvalScope Concurrent Evaluation](./models/gpt-oss/2-gpt-oss-20b%20Evalscope并发测试.md) @Guo Xuanbo
  - [x] [gpt-oss-20b LMStudio Local Deployment](./models/gpt-oss/3-gpt-oss-20b%20lmstudio%20本地部署调用.md) @Guo Xuanbo
  - [x] [gpt-oss-20b Lora Fine-Tuning & SwanLab Visualization](./models/gpt-oss/4-gpt-oss-20b%20Lora%20微调及%20SwanLab%20可视化记录.md) @Guo Xuanbo
  - [x] [gpt-oss-20b DPO Fine-Tuning & SwanLab Visualization](./models/gpt-oss/5-gpt-oss-20b%20DPO%20微调及%20SwanLab%20可视化记录.md) @Guo Xuanbo

- [GLM-4.1-Thinking](https://github.com/zai-org/GLM-4.1V-Thinking)
  - [x] [GLM-4.1V-Thinking vLLM Deployment](./models/GLM-4.1V-Thinking/01-GLM-4%201V-Thinking%20vLLM部署调用.md) @Lin Hengyu
  - [x] [GLM-4.1V-Thinking Gradio Deployment](./models/GLM-4.1V-Thinking/02-GLM-4%201V-Thinking%20Gradio部署.md) @Lin Hengyu
  - [x] [GLM-4.1V-Thinking Lora Fine-Tuning & SwanLab Visualization](./models/GLM-4.1V-Thinking/03-GLM-4%201V-Thinking%20LoRA%20及%20SwanLab%20可视化记录.md) @Lin Hengyu
  - [x] [GLM-4.1V-Thinking Docker Image](https://www.codewithgpu.com/i/datawhalechina/self-llm/GLM4.1V-Thinking-lora) @Lin Hengyu

- [GLM-4.5-Air](https://github.com/zai-org/GLM-4.5)
  - [x] [GLM-4.5-Air vLLM Deployment](./models/GLM-4.5-Air/01-GLM-4.5-Air-vLLM%20部署调用.md) @Buyao Congjiangsuan
  - [x] [GLM-4.5-Air EvalScope IQ/EQ & Concurrent Evaluation](./models/GLM-4.5-Air/02-GLM-4.5-Air%20EvalScope%20并发测试.md) @Buyao Congjiangsuan
  - [x] [GLM-4.5-Air Lora Fine-Tuning](./models/GLM-4.5-Air/03-GLM-4.5-Air-Lora%20及%20Swanlab%20可视化微调.md) @Buyao Congjiangsuan
  - [x] [GLM-4.5-Air Ucloud Docker Image](https://www.compshare.cn/images/lUQhKDCeCdZW?referral_code=ELukJdQS3vvCwYIfgsQf2C&ytag=GPU_yy_github_selfllm) @Buyao Congjiangsuan

- [ERNIE-4.5](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT)
  - [x] [ERNIE-4.5-0.3B-PT Lora Fine-Tuning & SwanLab Visualization](./models/ERNIE-4.5/01-ERNIE-4.5-0.3B-PT%20Lora%20微调及%20SwanLab%20可视化记录.md) @Buyao Congjiangsuan
  - [x] [ERNIE-4.5-0.3B-PT Lora Docker Image](https://www.codewithgpu.com/i/datawhalechina/self-llm/ERNIE-4.5-lora) @Buyao Congjiangsuan

- [Hunyuan-A13B-Instruct](https://github.com/Tencent-Hunyuan/Hunyuan-A13B)
  - [x] [Hunyuan-A13B-Instruct Model Architecture Analysis Blog](./models/Hunyuan-A13B-Instruct/01-Hunyuan-A13B-Instruct%20模型架构解析%20Blog.md) @Zhuo Tangyue
  - [x] [Hunyuan-A13B-Instruct SGLang Deployment](./models/Hunyuan-A13B-Instruct/03-Hunyuan-A13B-Instruct-SGLang部署调用.md) @fancy
  - [x] [Hunyuan-A13B-Instruct Lora SwanLab Visual Fine-Tuning](./models/Hunyuan-A13B-Instruct/05-Hunyuan-A13B-Instruct-LoRA及SwanLab可视化记录.md) @Xie Haoran
  - [x] [Hunyuan-A13B-Instruct Lora Docker Image](https://www.codewithgpu.com/i/datawhalechina/self-llm/Hunyuan-A13B-Instruct-lora) @Xie Haoran

- [Qwen3](https://github.com/QwenLM/Qwen3)
  - [x] [Qwen3 Model Architecture Analysis Blog](./models/Qwen3/01-Qwen3-模型结构解析-Blog.md) @Wang Zeyu
  - [x] [Qwen3-8B vLLM Deployment](./models/Qwen3/02-Qwen3-8B-vLLM%20部署调用.md) @Li Jiaojiao
  - [x] [Qwen3-8B Windows LMStudio Deployment](./models/Qwen3/03-Qwen3-7B-Instruct%20Windows%20LMStudio%20部署.md) @Wang Yiming
  - [x] [Qwen3-8B Evalscope IQ/EQ Evaluation](./models/Qwen3/04-Qwen3-8B%20EvalScope智商情商评测.md) @Li Jiaojiao
  - [x] [Qwen3-8B Lora Fine-Tuning & SwanLab Visualization](./models/Qwen3/05-Qwen3-8B-LoRA及SwanLab可视化记录.md) @Jiang Shufan
  - [x] [Qwen3-30B-A3B Fine-Tuning & SwanLab Visualization](./models/Qwen3/06-Qwen3-30B-A3B%20微调及%20SwanLab%20可视化记录.md) @Gao Liye
  - [x] [Qwen3 Think Decoded Blog](./models/Qwen3/07-Qwen3-Think-解密-Blog.md) @Fan Qi
  - [x] [Qwen3-8B Docker Image](https://www.codewithgpu.com/i/datawhalechina/self-llm/Qwen3) @Gao Liye
  - [x] [Utility of Qwen3-0.6B Small Model](./models/Qwen3/08-Qwen3_0_6B的小模型有什么用.md) @Buyao Congjiangsuan
  - [x] [Qwen3-1.7B Medical Reasoning Dialogue Fine-Tuning & SwanLab Visualization](./models/Qwen3/09-Qwen3-1.7B-医学推理式对话微调%20及%20SwanLab%20可视化记录.md) @Lin Zeyi
  - [x] [Qwen3-8B GRPO Fine-Tuning & SwanLab Visualization](./models/Qwen3/10-Qwen3-8B%20GRPO微调及通过swanlab可视化.md) @Guo Xuanbo

- [Kimi](https://github.com/MoonshotAI/Kimi-VL)
  - [x] [Kimi-VL-A3B Technical Report Interpretation](./models/Kimi-VL/02-Kimi-VL-技术报告解读.md) @Wang Zeyu
  - [x] [Kimi-VL-A3B-Thinking WebDemo Deployment](./models/Kimi-VL/01-Kimi-VL-对话助手.md) @Jiang Shufan

- [Llama4](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
  - [x] [Llama4 Chat Assistant](./models/Llama4/01-Llama4-对话助手/01-Llama4-对话助手.md) @Jiang Shufan

- [SpatialLM](https://github.com/manycore-research/SpatialLM)
  - [x] [SpatialLM 3D Point Cloud Understanding & Object Detection Model Deployment](./models/SpatialLM/readme.md) @Wang Zeyu

- [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2)
  - [x] [Hunyuan3D-2 Series Model Deployment](./models/Hunyuan3D-2/01-Hunyuan3D-2%20系列模型部署.md) @Lin Hengyu
  - [x] [Hunyuan3D-2 Series Model Code Invocation](./models/Hunyuan3D-2/02-Hunyuan3D-2%20系列模型代码调用.md) @Lin Hengyu
  - [x] [Hunyuan3D-2 Series Model Gradio Deployment](./models/Hunyuan3D-2/03-Hunyuan3D-2%20系列模型Gradio部署.md) @Lin Hengyu
  - [x] [Hunyuan3D-2 Series Model API Server](./models/Hunyuan3D-2/04-Hunyuan3D-2%20系列模型API%20Server.md) @Lin Hengyu
  - [x] [Hunyuan3D-2 Docker Image](https://www.codewithgpu.com/i/datawhalechina/self-llm/Hunyuan3D-2) @Lin Hengyu

- [Gemma3](https://huggingface.co/google/gemma-3-4b-it)
  - [x] [gemma-3-4b-it FastApi Deployment](./models/Gemma3/01-gemma-3-4b-it%20FastApi%20部署调用.md) @Du Sen
  - [x] [gemma-3-4b-it ollama + open-webui Deployment](./models/Gemma3/03-gemma-3-4b-it-ollama%20+%20open-webui部署.md) @Sun Chao
  - [x] [gemma-3-4b-it Evalscope IQ/EQ Evaluation](./models/Gemma3/04-Gemma3-4b%20%20evalscope智商情商评测.md) @Zhang Longfei
  - [x] [gemma-3-4b-it Lora Fine-Tuning](./models/Gemma3/05-gemma-3-4b-it%20LoRA.md) @Qiaomai
  - [x] [gemma-3-4b-it Docker Image](https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-gemma3) @Jiang Shufan
  - [x] [gemma-3-4b-it GRPO Fine-Tuning & SwanLab Visualization](./models/Gemma3/6-gemma3-4B-itGRPO微调及通过swanlab可视化.md) @Guo Xuanbo

- [DeepSeek-R1-Distill](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
  - [x] [DeepSeek-R1-Distill-Qwen-7B FastApi Deployment](./models/DeepSeek-R1-Distill-Qwen/01-DeepSeek-R1-Distill-Qwen-7B%20FastApi%20部署调用.md) @Luo Xiutao
  - [x] [DeepSeek-R1-Distill-Qwen-7B Langchain Integration](./models/DeepSeek-R1-Distill-Qwen/02-DeepSeek-R1-Distill-Qwen-7B%20Langchain%20接入.md) @Luo Xiutao
  - [x] [DeepSeek-R1-Distill-Qwen-7B WebDemo Deployment](./models/DeepSeek-R1-Distill-Qwen/03-DeepSeek-R1-Distill-Qwen-7B%20WebDemo%20部署.md) @Luo Xiutao
  - [x] [DeepSeek-R1-Distill-Qwen-7B vLLM Deployment](./models/DeepSeek-R1-Distill-Qwen/04-DeepSeek-R1-Distill-Qwen-7B%20vLLM%20部署调用.md) @Luo Xiutao
  - [x] [DeepSeek-R1-0528-Qwen3-8B-GRPO & SwanLab Visualization](./models/DeepSeek-R1-Distill-Qwen/05-DeepSeek-R1-0528-Qwen3-8B-GRPO及swanlab可视化.md) @Guo Xuanbo

- [MiniCPM-o-2_6](https://github.com/OpenBMB/MiniCPM-o)
  - [x] [minicpm-o-2.6 FastApi Deployment and Invocation](./models/MiniCPM-o/01MiniCPM-o%202%206%20FastApi部署调用%20.md) @林恒宇
  - [x] [minicpm-o-2.6 WebDemo Deployment](./models/MiniCPM-o/02minicpm-o-2.6WebDemo_streamlit.py) @程宏
  - [x] [minicpm-o-2.6 Multimodal Speech Capabilities](./models/MiniCPM-o/03-MiniCPM-o-2.6%20多模态语音能力.md) @邓恺俊
  - [x] [minicpm-o-2.6 Visualized Lora Fine-Tuning](./models/MiniCPM-o/04-MiniCPM-0-2.6%20Lora微调.md) @林泽毅

- [InternLM3](https://github.com/InternLM/InternLM)
  - [x] [internlm3-8b-instruct FastApi Deployment and Invocation](./models/InternLM3/01-InternLM3-8B-Instruct%20FastAPI.md) @苏向标
  - [x] [internlm3-8b-instruct Langchian Integration](./models/InternLM3/02-internlm3-8b-Instruct%20Langchain%20接入.md) @赵文恺
  - [x] [internlm3-8b-instruct WebDemo Deployment](./models/InternLM3/03-InternLM3-8B-Instruct%20WebDemo部署.md) @王泽宇
  - [x] [internlm3-8b-instruct Lora Fine-Tuning](./models/InternLM3/04-InternLM3-8B-Instruct%20LoRA.md) @程宏
  - [x] [internlm3-8b-instruct o1-like Reasoning Chain Implementation](./models/InternLM3/05-internlm3-8b-instruct%20与o1%20.md) @陈睿

- [phi4](https://huggingface.co/microsoft/phi-4)
  - [x] [phi4 FastApi Deployment and Invocation](./models/phi4/01-Phi-4%20FastApi%20部署调用.md) @杜森
  - [x] [phi4 Langchain Integration](./models/phi4/02-Phi-4-Langchain接入.md) @小罗
  - [x] [phi4 WebDemo Deployment](./models/phi4/03-Phi-4%20WebDemo部署.md) @杜森
  - [x] [phi4 Lora Fine-Tuning](./models/phi4/04-Phi-4-Lora%20微调.md) @郑远婧
  - [x] [phi4 Lora Fine-Tuning for NER Task with SwanLab Visualization](./models/phi4/05-Phi-4-Lora%20微调%20命名实体识别.md) @林泽毅

- [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder)
  - [x] [Qwen2.5-Coder-7B-Instruct FastApi Deployment and Invocation](./models/Qwen2.5-Coder/01-Qwen2.5-Coder-7B-Instruct%20FastApi%20部署调用.md) @赵文恺
  - [x] [Qwen2.5-Coder-7B-Instruct Langchian Integration](./models/Qwen2.5-Coder/02-Qwen2.5-7B-Instruct%20Langchain%20接入.md) @杨晨旭
  - [x] [Qwen2.5-Coder-7B-Instruct WebDemo Deployment](./models/Qwen2.5-Coder/03-Qwen2.5-Coder-7B-Instruct%20WebDemo部署.md) @王泽宇
  - [x] [Qwen2.5-Coder-7B-Instruct vLLM Deployment](./models/Qwen2.5-Coder/04-Qwen2.5-Coder-7B-Instruct%20vLLM%20部署调用.md) @王泽宇
  - [x] [Qwen2.5-Coder-7B-Instruct Lora Fine-Tuning](./models/Qwen2.5-Coder/Qwen2.5-Coder-7B-Instruct%20Lora%20微调.md) @荞麦
  - [x] [Qwen2.5-Coder-7B-Instruct Lora Fine-Tuning with SwanLab Visualization](./models/Qwen2.5-Coder/05-Qwen2.5-Coder-7B-Instruct%20Lora%20微调%20SwanLab%20可视化记录版.md) @杨卓

- [Qwen2-vl](https://github.com/QwenLM/Qwen2-VL)
  - [x] [Qwen2-vl-2B FastApi Deployment and Invocation](./models/Qwen2-VL/01-Qwen2-VL-2B-Instruct%20FastApi%20部署调用.md) @姜舒凡
  - [x] [Qwen2-vl-2B WebDemo Deployment](./models/Qwen2-VL/02-Qwen2-VL-2B-Instruct%20Web%20Demo部署.md) @赵伟
  - [x] [Qwen2-vl-2B vLLM Deployment](./models/Qwen2-VL/03-Qwen2-VL-2B-Instruct%20vLLM部署调用.md) @荞麦
  - [x] [Qwen2-vl-2B Lora Fine-Tuning](./models/Qwen2-VL/04-Qwen2-VL-2B%20Lora%20微调.md) @李柯辰
  - [x] [Qwen2-vl-2B Lora Fine-Tuning with SwanLab Visualization](./models/Qwen2-VL/05-Qwen2-VL-2B-Instruct%20Lora%20微调%20SwanLab%20可视化记录版.md) @林泽毅
  - [x] [Qwen2-vl-2B Lora Fine-Tuning Case - LaTexOCR](./models/Qwen2-VL/06-Qwen2-VL-2B-Instruct%20Lora%20微调案例%20-%20LaTexOCR.md) @林泽毅

- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)
  - [x] [Qwen2.5-7B-Instruct FastApi Deployment and Invocation](./models/Qwen2.5/01-Qwen2.5-7B-Instruct%20FastApi%20部署调用.md) @娄天奥
  - [x] [Qwen2.5-7B-Instruct Langchain Integration](./models/Qwen2.5/02-Qwen2.5-7B-Instruct%20Langchain%20接入.md) @娄天奥
  - [x] [Qwen2.5-7B-Instruct vLLM Deployment and Invocation](./models/Qwen2.5/03-Qwen2.5-7B-Instruct%20vLLM%20部署调用.md) @姜舒凡
  - [x] [Qwen2.5-7B-Instruct WebDemo Deployment](./models/Qwen2.5/04-Qwen2_5-7B-Instruct%20WebDemo部署.md) @高立业
  - [x] [Qwen2.5-7B-Instruct Lora Fine-Tuning](./models/Qwen2.5/05-Qwen2.5-7B-Instruct%20Lora%20微调.md) @左春生
  - [x] [Qwen2.5-7B-Instruct o1-like Reasoning Chain Implementation](./models/Qwen2.5/06-Qwen2.5-7B-Instruct%20o1-like%20推理链实现.md) @姜舒凡
  - [x] [Qwen2.5-7B-Instruct Lora Fine-Tuning with SwanLab Visualization](./models/Qwen2.5/07-Qwen2.5-7B-Instruct%20Lora%20微调%20SwanLab可视化记录版.md) @林泽毅

- [Apple OpenELM](https://machinelearning.apple.com/research/openelm)
  - [x] [OpenELM-3B-Instruct FastApi Deployment and Invocation](./models/OpenELM/01-OpenELM-3B-Instruct%20FastApi部署调用.md) @王泽宇
  - [x] [OpenELM-3B-Instruct Lora Fine-Tuning](./models/OpenELM/02-OpenELM-3B-Instruct%20Lora微调.md) @王泽宇

- [Llama3_1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  - [x] [Llama3_1-8B-Instruct FastApi Deployment and Invocation](./models/Llama3_1/01-Llama3_1-8B-Instruct%20FastApi%20部署调用.md) @不要葱姜蒜
  - [x] [Llama3_1-8B-Instruct Langchain Integration](./models/Llama3_1/02-Llama3_1-8B-Instruct%20langchain接入.md) @张晋
  - [x] [Llama3_1-8B-Instruct WebDemo Deployment](./models/Llama3_1/03-Llama3_1-8B-Instruct%20WebDemo部署.md) @张晋
  - [x] [Llama3_1-8B-Instruct Lora Fine-Tuning](./models/Llama3_1/04-Llama3_1-8B--Instruct%20Lora%20微调.md) @不要葱姜蒜

- [Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it)
  - [x] [Gemma-2-9b-it FastApi Deployment and Invocation](./models/Gemma2/01-Gemma-2-9b-it%20FastApi%20部署调用.md) @不要葱姜蒜
  - [x] [Gemma-2-9b-it Langchain Integration](./models/Gemma2/02-Gemma-2-9b-it%20langchain%20接入.md) @不要葱姜蒜
  - [x] [Gemma-2-9b-it WebDemo Deployment](./models/Gemma2/03-Gemma-2-9b-it%20WebDemo%20部署.md) @不要葱姜蒜
  - [x] [Gemma-2-9b-it Peft Lora Fine-Tuning](./models/Gemma2/04-Gemma-2-9b-it%20peft%20lora微调.md) @不要葱姜蒜

- [Yuan2.0](https://github.com/IEIT-Yuan/Yuan-2.0)
  - [x] [Yuan2.0-2B FastApi Deployment and Invocation](./models/Yuan2.0/01-Yuan2.0-2B%20FastApi%20部署调用.md) @张帆
  - [x] [Yuan2.0-2B Langchain Integration](./models/Yuan2.0/02-Yuan2.0-2B%20Langchain%20接入.md) @张帆
  - [x] [Yuan2.0-2B WebDemo Deployment](./models/Yuan2.0/03-Yuan2.0-2B%20WebDemo部署.md) @张帆
  - [x] [Yuan2.0-2B vLLM Deployment and Invocation](./models/Yuan2.0/04-Yuan2.0-2B%20vLLM部署调用.md) @张帆
  - [x] [Yuan2.0-2B Lora Fine-Tuning](./models/Yuan2.0/05-Yuan2.0-2B%20Lora微调.md) @张帆

- [Yuan2.0-M32](https://github.com/IEIT-Yuan/Yuan2.0-M32)
  - [x] [Yuan2.0-M32 FastApi Deployment and Invocation](./models/Yuan2.0-M32/01-Yuan2.0-M32%20FastApi%20部署调用.md) @张帆
  - [x] [Yuan2.0-M32 Langchain Integration](./models/Yuan2.0-M32/02-Yuan2.0-M32%20Langchain%20接入.md) @张帆
  - [x] [Yuan2.0-M32 WebDemo Deployment](./models/Yuan2.0-M32/03-Yuan2.0-M32%20WebDemo部署.md) @张帆

- [DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2)
  - [x] [DeepSeek-Coder-V2-Lite-Instruct FastApi Deployment and Invocation](./models/DeepSeek-Coder-V2/01-DeepSeek-Coder-V2-Lite-Instruct%20FastApi%20部署调用.md) @姜舒凡
  - [x] [DeepSeek-Coder-V2-Lite-Instruct Langchain Integration](./models/DeepSeek-Coder-V2/02-DeepSeek-Coder-V2-Lite-Instruct%20接入%20LangChain.md) @姜舒凡
  - [x] [DeepSeek-Coder-V2-Lite-Instruct WebDemo Deployment](./models/DeepSeek-Coder-V2/03-DeepSeek-Coder-V2-Lite-Instruct%20WebDemo%20部署.md) @Kailigithub
  - [x] [DeepSeek-Coder-V2-Lite-Instruct Lora Fine-Tuning](./models/DeepSeek-Coder-V2/04-DeepSeek-Coder-V2-Lite-Instruct%20Lora%20微调.md) @余洋

- [Bilibili Index-1.9B](https://github.com/bilibili/Index-1.9B)
  - [x] [Index-1.9B-Chat FastApi Deployment and Invocation](./models/bilibili_Index-1.9B/01-Index-1.9B-chat%20FastApi%20部署调用.md) @邓恺俊
  - [x] [Index-1.9B-Chat Langchain Integration](./models/bilibili_Index-1.9B/02-Index-1.9B-Chat%20接入%20LangChain.md) @张友东
  - [x] [Index-1.9B-Chat WebDemo Deployment](./models/bilibili_Index-1.9B/03-Index-1.9B-chat%20WebDemo部署.md) @程宏
  - [x] [Index-1.9B-Chat Lora Fine-Tuning](./models/bilibili_Index-1.9B/04-Index-1.9B-Chat%20Lora%20微调.md) @姜舒凡

- [Qwen2](https://github.com/QwenLM/Qwen2)
  - [x] [Qwen2-7B-Instruct FastApi Deployment and Invocation](./models/Qwen2/01-Qwen2-7B-Instruct%20FastApi%20部署调用.md) @康婧淇
  - [x] [Qwen2-7B-Instruct Langchain Integration](./models/Qwen2/02-Qwen2-7B-Instruct%20Langchain%20接入.md) @不要葱姜蒜
  - [x] [Qwen2-7B-Instruct WebDemo Deployment](./models/Qwen2/03-Qwen2-7B-Instruct%20WebDemo部署.md) @三水
  - [x] [Qwen2-7B-Instruct vLLM Deployment and Invocation](./models/Qwen2/04-Qwen2-7B-Instruct%20vLLM%20部署调用.md) @姜舒凡
  - [x] [Qwen2-7B-Instruct Lora Fine-Tuning](./models/Qwen2/05-Qwen2-7B-Instruct%20Lora%20微调.md) @散步

- [GLM-4](https://github.com/THUDM/GLM-4.git)
  - [x] [GLM-4-9B-chat FastApi Deployment and Invocation](./models/GLM-4/01-GLM-4-9B-chat%20FastApi%20部署调用.md) @张友东
  - [x] [GLM-4-9B-chat Langchain Integration](./models/GLM-4/02-GLM-4-9B-chat%20langchain%20接入.md) @谭逸珂
  - [x] [GLM-4-9B-chat WebDemo Deployment](./models/GLM-4/03-GLM-4-9B-Chat%20WebDemo.md) @何至轩
  - [x] [GLM-4-9B-chat vLLM Deployment](./models/GLM-4/04-GLM-4-9B-Chat%20vLLM%20部署调用.md) @王熠明
  - [x] [GLM-4-9B-chat Lora Fine-Tuning](./models/GLM-4/05-GLM-4-9B-chat%20Lora%20微调.md) @肖鸿儒
  - [x] [GLM-4-9B-chat-hf Lora Fine-Tuning](./models/GLM-4/05-GLM-4-9B-chat-hf%20Lora%20微调.md) @付志远

- [Qwen 1.5](https://github.com/QwenLM/Qwen1.5.git)
  - [x] [Qwen1.5-7B-chat FastApi Deployment and Invocation](./models/Qwen1.5/01-Qwen1.5-7B-Chat%20FastApi%20部署调用.md) @颜鑫
  - [x] [Qwen1.5-7B-chat Langchain Integration](./models/Qwen1.5/02-Qwen1.5-7B-Chat%20接入langchain搭建知识库助手.md) @颜鑫
  - [x] [Qwen1.5-7B-chat WebDemo Deployment](./models/Qwen1.5/03-Qwen1.5-7B-Chat%20WebDemo.md) @颜鑫
  - [x] [Qwen1.5-7B-chat Lora Fine-Tuning](./models/Qwen1.5/04-Qwen1.5-7B-chat%20Lora%20微调.md) @不要葱姜蒜
  - [x] [Qwen1.5-72B-chat-GPTQ-Int4 Deployment Environment](./models/Qwen1.5/05-Qwen1.5-7B-Chat-GPTQ-Int4%20%20WebDemo.md) @byx020119
  - [x] [Qwen1.5-MoE-chat Transformers Deployment and Invocation](./models/Qwen1.5/06-Qwen1.5-MoE-A2.7B.md) @丁悦
  - [x] [Qwen1.5-7B-chat vLLM Inference Deployment](./models/Qwen1.5/07-Qwen1.5-7B-Chat%20vLLM%20推理部署调用.md) @高立业
  - [x] [Qwen1.5-7B-chat Lora Fine-Tuning with SwanLab Experiment Management Platform](./models/Qwen1.5/08-Qwen1.5-7B-chat%20LoRA微调接入实验管理.md) @黄柏特

- [Google-Gemma](https://huggingface.co/google/gemma-7b-it)
  - [x] [gemma-2b-it FastApi Deployment and Invocation](./models/Gemma/01-Gemma-2B-Instruct%20FastApi%20部署调用.md) @东东
  - [x] [gemma-2b-it Langchain Integration](./models/Gemma/02-Gemma-2B-Instruct%20langchain%20接入.md) @东东
  - [x] [gemma-2b-it WebDemo Deployment](./models/Gemma/03-Gemma-2B-Instruct%20WebDemo%20部署.md) @东东
  - [x] [gemma-2b-it Peft Lora Fine-Tuning](./models/Gemma/04-Gemma-2B-Instruct%20Lora微调.md) @东东

- [phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
  - [x] [Phi-3-mini-4k-instruct FastApi Deployment and Invocation](./models/phi-3/01-Phi-3-mini-4k-instruct%20FastApi%20部署调用.md) @郑皓桦
  - [x] [Phi-3-mini-4k-instruct Langchain Integration](./models/phi-3/02-Phi-3-mini-4k-instruct%20langchain%20接入.md) @郑皓桦
  - [x] [Phi-3-mini-4k-instruct WebDemo Deployment](./models/phi-3/03-Phi-3-mini-4k-instruct%20WebDemo部署.md) @丁悦
  - [x] [Phi-3-mini-4k-instruct Lora Fine-Tuning](./models/phi-3/04-Phi-3-mini-4k-Instruct%20Lora%20微调.md) @丁悦

- [CharacterGLM-6B](https://github.com/thu-coai/CharacterGLM-6B)
  - [x] [CharacterGLM-6B Transformers Deployment and Invocation](./models/CharacterGLM/01-CharacterGLM-6B%20Transformer部署调用.md) @孙健壮
  - [x] [CharacterGLM-6B FastApi Deployment and Invocation](./models/CharacterGLM/02-CharacterGLM-6B%20FastApi部署调用.md) @孙健壮
  - [x] [CharacterGLM-6B WebDemo Deployment](./models/CharacterGLM/03-CharacterGLM-6B-chat.md) @孙健壮
  - [x] [CharacterGLM-6B Lora Fine-Tuning](./models/CharacterGLM/04-CharacterGLM-6B%20Lora微调.md) @孙健壮

- [LLaMA3-8B-Instruct](https://github.com/meta-llama/llama3.git)
  - [x] [LLaMA3-8B-Instruct FastApi Deployment and Invocation](./models/LLaMA3/01-LLaMA3-8B-Instruct%20FastApi%20部署调用.md) @高立业
  - [x] [LLaMA3-8B-Instruct Langchain Integration](./models/LLaMA3/02-LLaMA3-8B-Instruct%20langchain%20接入.md) @不要葱姜蒜
  - [x] [LLaMA3-8B-Instruct WebDemo Deployment](./models/LLaMA3/03-LLaMA3-8B-Instruct%20WebDemo%20部署.md) @不要葱姜蒜
  - [x] [LLaMA3-8B-Instruct Lora Fine-Tuning](./models/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20微调.md) @高立业

- [XVERSE-7B-Chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary)
  - [x] [XVERSE-7B-Chat Transformers Deployment and Invocation](./models/XVERSE/01-XVERSE-7B-chat%20Transformers推理.md) @郭志航
  - [x] [XVERSE-7B-Chat FastApi Deployment and Invocation](./models/XVERSE/02-XVERSE-7B-chat%20FastAPI部署.md) @郭志航
  - [x] [XVERSE-7B-Chat Langchain Integration](./models/XVERSE/03-XVERSE-7B-chat%20langchain%20接入.md) @郭志航
  - [x] [XVERSE-7B-Chat WebDemo Deployment](./models/XVERSE/04-XVERSE-7B-chat%20WebDemo%20部署.md) @郭志航
  - [x] [XVERSE-7B-Chat Lora Fine-Tuning](./models/XVERSE/05-XVERSE-7B-Chat%20Lora%20微调.md) @郭志航

- [TransNormerLLM](https://github.com/OpenNLPLab/TransnormerLLM.git)
  - [x] [TransNormerLLM-7B-Chat FastApi Deployment and Invocation](./models/TransNormer/01-TransNormer-7B%20FastApi%20部署调用.md) @王茂霖
  - [x] [TransNormerLLM-7B-Chat Langchain Integration](./models/TransNormer/02-TransNormer-7B%20接入langchain搭建知识库助手.md) @王茂霖
  - [x] [TransNormerLLM-7B-Chat WebDemo Deployment](./models/TransNormer/03-TransNormer-7B%20WebDemo.md) @王茂霖
  - [x] [TransNormerLLM-7B-Chat Lora Fine-Tuning](./models/TransNormer/04-TrasnNormer-7B%20Lora%20微调.md) @王茂霖

- [BlueLM Vivo Blue Heart Model](https://github.com/vivo-ai-lab/BlueLM.git)
  - [x] [BlueLM-7B-Chat FastApi Deployment and Invocation](./models/BlueLM/01-BlueLM-7B-Chat%20FastApi%20部署.md) @郭志航
  - [x] [BlueLM-7B-Chat Langchain Integration](./models/BlueLM/02-BlueLM-7B-Chat%20langchain%20接入.md) @郭志航
  - [x] [BlueLM-7B-Chat WebDemo Deployment](./models/BlueLM/03-BlueLM-7B-Chat%20WebDemo%20部署.md) @郭志航
  - [x] [BlueLM-7B-Chat Lora Fine-Tuning](./models/BlueLM/04-BlueLM-7B-Chat%20Lora%20微调.md) @郭志航

- [InternLM2](https://github.com/InternLM/InternLM)
  - [x] [InternLM2-7B-chat FastApi Deployment and Invocation](./models/InternLM2/01-InternLM2-7B-chat%20FastAPI部署.md) @不要葱姜蒜
  - [x] [InternLM2-7B-chat Langchain Integration](./models/InternLM2/02-InternLM2-7B-chat%20langchain%20接入.md) @不要葱姜蒜
  - [x] [InternLM2-7B-chat WebDemo Deployment](./models/InternLM2/03-InternLM2-7B-chat%20WebDemo%20部署.md) @郑皓桦
  - [x] [InternLM2-7B-chat Xtuner Qlora Fine-Tuning](./models/InternLM2/04-InternLM2-7B-chat%20Xtuner%20Qlora%20微调.md) @郑皓桦

- [DeepSeek Deep Exploration](https://github.com/deepseek-ai/DeepSeek-LLM)
  - [x] [DeepSeek-7B-chat FastApi Deployment and Invocation](./models/DeepSeek/01-DeepSeek-7B-chat%20FastApi.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat Langchain Integration](./models/DeepSeek/02-DeepSeek-7B-chat%20langchain.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat WebDemo](./models/DeepSeek/03-DeepSeek-7B-chat%20WebDemo.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat Lora Fine-Tuning](./models/DeepSeek/04-DeepSeek-7B-chat%20Lora%20微调.md) @不要葱姜蒜
  - [x] [DeepSeek-7B-chat 4bits Quantization Qlora Fine-Tuning](./models/DeepSeek/05-DeepSeek-7B-chat%204bits量化%20Qlora%20微调.md) @不要葱姜蒜
  - [x] [DeepSeek-MoE-16b-chat Transformers Deployment and Invocation](./models/DeepSeek/06-DeepSeek-MoE-16b-chat%20Transformer部署调用.md) @Kailigithub
  - [x] [DeepSeek-MoE-16b-chat FastApi Deployment and Invocation](./models/DeepSeek/06-DeepSeek-MoE-16b-chat%20FastApi.md) @Kailigithub
  - [x] [DeepSeek-coder-6.7b Fine-Tuning Colab](./models/DeepSeek/07-deepseek_fine_tune.ipynb) @Swiftie
  - [x] [Deepseek-coder-6.7b WebDemo Colab](./models/DeepSeek/08-deepseek_web_demo.ipynb) @Swiftie

- [MiniCPM](https://github.com/OpenBMB/MiniCPM.git)
  - [x] [MiniCPM-2B-chat Transformers Deployment and Invocation](./models/MiniCPM/MiniCPM-2B-chat%20transformers%20部署调用.md) @Kailigithub 
  - [x] [MiniCPM-2B-chat FastApi Deployment and Invocation](./models/MiniCPM/MiniCPM-2B-chat%20FastApi%20部署调用.md) @Kailigithub 
  - [x] [MiniCPM-2B-chat Langchain Integration](./models/MiniCPM/MiniCPM-2B-chat%20langchain接入.md) @不要葱姜蒜 
  - [x] [MiniCPM-2B-chat WebDemo Deployment](./models/MiniCPM/MiniCPM-2B-chat%20WebDemo部署.md) @Kailigithub 
  - [x] [MiniCPM-2B-chat Lora && Full Fine-Tuning](./models/MiniCPM/MiniCPM-2B-chat%20Lora%20&&%20Full%20微调.md) @不要葱姜蒜 
  - [x] Official Link: [MiniCPM Tutorial](https://modelbest.feishu.cn/wiki/D2tFw8Pcsi5CIzkaHNacLK64npg) @OpenBMB 
  - [x] Official Link: [MiniCPM-Cookbook](https://github.com/OpenBMB/MiniCPM-CookBook) @OpenBMB

- [Qwen-Audio](https://github.com/QwenLM/Qwen-Audio.git)
  - [x] [Qwen-Audio FastApi Deployment and Invocation](./models/Qwen-Audio/01-Qwen-Audio-chat%20FastApi.md) @陈思州
  - [x] [Qwen-Audio WebDemo](./models/Qwen-Audio/02-Qwen-Audio-chat%20WebDemo.md) @陈思州

- [Qwen](https://github.com/QwenLM/Qwen.git)
  - [x] [Qwen-7B-chat Transformers Deployment and Invocation](./models/Qwen/01-Qwen-7B-Chat%20Transformers部署调用.md) @李娇娇
  - [x] [Qwen-7B-chat FastApi Deployment and Invocation](./models/Qwen/02-Qwen-7B-Chat%20FastApi%20部署调用.md) @李娇娇
  - [x] [Qwen-7B-chat WebDemo](./models/Qwen/03-Qwen-7B-Chat%20WebDemo.md) @李娇娇
  - [x] [Qwen-7B-chat Lora Fine-Tuning](./models/Qwen/04-Qwen-7B-Chat%20Lora%20微调.md) @不要葱姜蒜
  - [x] [Qwen-7B-chat Ptuning Fine-Tuning](./models/Qwen/05-Qwen-7B-Chat%20Ptuning%20微调.md) @肖鸿儒
  - [x] [Qwen-7B-chat Full Fine-Tuning](./models/Qwen/06-Qwen-7B-chat%20全量微调.md) @不要葱姜蒜
  - [x] [Qwen-7B-Chat Langchain Integration for Knowledge Base Assistant](./models/Qwen/07-Qwen-7B-Chat%20接入langchain搭建知识库助手.md) @李娇娇
  - [x] [Qwen-7B-chat Low-Precision Training](./models/Qwen/08-Qwen-7B-Chat%20Lora%20低精度微调.md) @肖鸿儒
  - [x] [Qwen-1_8B-chat CPU Deployment](./models/Qwen/09-Qwen-1_8B-chat%20CPU%20部署%20.md) @散步

- [Yi 01.AI](https://github.com/01-ai/Yi.git)
  - [x] [Yi-6B-chat FastApi Deployment and Invocation](./models/Yi/01-Yi-6B-Chat%20FastApi%20部署调用.md) @李柯辰
  - [x] [Yi-6B-chat Langchain Integration](./models/Yi/02-Yi-6B-Chat%20接入langchain搭建知识库助手.md) @李柯辰
  - [x] [Yi-6B-chat WebDemo](./models/Yi/03-Yi-6B-chat%20WebDemo.md) @肖鸿儒
  - [x] [Yi-6B-chat Lora Fine-Tuning](./models/Yi/04-Yi-6B-Chat%20Lora%20微调.md) @李娇娇

- [Baichuan Baichuan Intelligence](https://www.baichuan-ai.com/home)
  - [x] [Baichuan2-7B-chat FastApi Deployment and Invocation](./BaiChuan/01-Baichuan2-7B-chat%2BFastApi%2B%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md) @惠佳豪
  - [x] [Baichuan2-7B-chat WebDemo](./models/BaiChuan/02-Baichuan-7B-chat%2BWebDemo.md) @惠佳豪
  - [x] [Baichuan2-7B-chat LangChain Framework Integration](./models/BaiChuan/03-Baichuan2-7B-chat%E6%8E%A5%E5%85%A5LangChain%E6%A1%86%E6%9E%B6.md) @惠佳豪
  - [x] [Baichuan2-7B-chat Lora Fine-Tuning](./models/BaiChuan/04-Baichuan2-7B-chat%2Blora%2B%E5%BE%AE%E8%B0%83.md) @惠佳豪

- [InternLM](https://github.com/InternLM/InternLM.git)
  - [x] [InternLM-Chat-7B Transformers Deployment and Invocation](./models/InternLM/01-InternLM-Chat-7B%20Transformers%20部署调用.md) @小罗
  - [x] [InternLM-Chat-7B FastApi Deployment and Invocation](./models/InternLM/02-internLM-Chat-7B%20FastApi.md) @不要葱姜蒜
  - [x] [InternLM-Chat-7B WebDemo](./models/InternLM/03-InternLM-Chat-7B.md) @不要葱姜蒜
  - [x] [Lagent+InternLM-Chat-7B-V1.1 WebDemo](./models/InternLM/04-Lagent+InternLM-Chat-7B-V1.1.md) @不要葱姜蒜
  - [x] [Puyu Lingbi Image Understanding & Creation WebDemo](./models/InternLM/05-浦语灵笔图文理解&创作.md) @不要葱姜蒜
  - [x] [InternLM-Chat-7B LangChain Framework Integration](./models/InternLM/06-InternLM接入LangChain搭建知识库助手.md) @Logan Zou

- [Atom (llama2)](https://hf-mirror.com/FlagAlpha/Atom-7B-Chat)
  - [x] [Atom-7B-chat WebDemo](./models/Atom/01-Atom-7B-chat-WebDemo.md) @Kailigithub
  - [x] [Atom-7B-chat Lora Fine-Tuning](./models/Atom/02-Atom-7B-Chat%20Lora%20微调.md) @Logan Zou
  - [x] [Atom-7B-Chat Langchain Integration for Knowledge Base Assistant](./models/Atom/03-Atom-7B-Chat%20接入langchain搭建知识库助手.md) @陈思州
  - [x] [Atom-7B-chat Full Fine-Tuning](./models/Atom/04-Atom-7B-chat%20全量微调.md) @Logan Zou

- [ChatGLM3](https://github.com/THUDM/ChatGLM3.git)
  - [x] [ChatGLM3-6B Transformers Deployment and Invocation](./models/ChatGLM/01-ChatGLM3-6B%20Transformer部署调用.md) @丁悦
  - [x] [ChatGLM3-6B FastApi Deployment and Invocation](./models/ChatGLM/02-ChatGLM3-6B%20FastApi部署调用.md) @丁悦
  - [x] [ChatGLM3-6B Chat WebDemo](./models/ChatGLM/03-ChatGLM3-6B-chat.md) @不要葱姜蒜
  - [x] [ChatGLM3-6B Code Interpreter WebDemo](./models/ChatGLM/04
### 通用环境配置

### General Environment Configuration

- [x] [pip, conda Source Change](./models/General-Setting/01-pip、conda换源.md) @不要葱姜蒜
- [x] [AutoDL Open Port](./models/General-Setting/02-AutoDL开放端口.md) @不要葱姜蒜

- Model Download
  - [x] [hugging face](./models/General-Setting/03-模型下载.md) @不要葱姜蒜
  - [x] [hugging face](./General-Setting/03-模型下载.md) Mirror Download @不要葱姜蒜
  - [x] [modelscope](./models/General-Setting/03-模型下载.md) @不要葱姜蒜
  - [x] [git-lfs](./models/General-Setting/03-模型下载.md) @不要葱姜蒜
  - [x] [Openxlab](./models/General-Setting/03-模型下载.md)

- Issue && PR
  - [x] [Submit Issue](./models/General-Setting/04-Issue&PR&update.md) @肖鸿儒
  - [x] [Submit PR](./models/General-Setting/04-Issue&PR&update.md) @肖鸿儒
  - [x] [Fork Update](./models/General-Setting/04-Issue&PR&update.md) @肖鸿儒


## Acknowledgments

### Core Contributors

- [宋志学(不要葱姜蒜)-项目负责人](https://github.com/KMnO4-zx) （Datawhale成员-中国矿业大学(北京)）
- [邹雨衡-项目负责人](https://github.com/logan-zou) （Datawhale成员-对外经济贸易大学）
- [肖鸿儒](https://github.com/Hongru0306) （Datawhale成员-同济大学）
- [郭志航](https://github.com/acwwt)（内容创作者）
- [林泽毅](https://github.com/Zeyi-Lin)（内容创作者-SwanLab产品负责人）
- [张帆](https://github.com/zhangfanTJU)（内容创作者-Datawhale成员）
- [姜舒凡](https://github.com/Tsumugii24)（内容创作者-Datawhale成员）
- [李娇娇](https://github.com/Aphasia0515) （Datawhale成员）
- [丁悦](https://github.com/dingyue772) （Datawhale-鲸英助教）
- [王泽宇](https://github.com/moyitech)（内容创作者-太原理工大学-鲸英助教）
- [惠佳豪](https://github.com/L4HeyXiao) （Datawhale-宣传大使）
- [王茂霖](https://github.com/mlw67)（内容创作者-Datawhale成员）
- [孙健壮](https://github.com/Caleb-Sun-jz)（内容创作者-对外经济贸易大学）
- [东东](https://github.com/LucaChen)（内容创作者-谷歌开发者机器学习技术专家）
- [高立业](https://github.com/0-yy-0)（内容创作者-DataWhale成员）
- [Kailigithub](https://github.com/Kailigithub) （Datawhale成员）
- [郑皓桦](https://github.com/BaiYu96) （内容创作者）
- [李柯辰](https://github.com/Joe-2002) （Datawhale成员）
- [程宏](https://github.com/chg0901)（内容创作者-Datawhale意向成员）
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
- [邓恺俊](https://github.com/Kedreamix)（内容创作者-Datawhale成员）
- [赵文恺](https://github.com/XiLinky)（内容创作者-太原理工大学-鲸英助教）
- [付志远](https://github.com/comfzy)（内容创作者-海南大学）
- [杜森](https://github.com/study520ai520)（内容创作者-Datawhale成员-南阳理工学院）
- [郑远婧](https://github.com/isaacahahah)（内容创作者-鲸英助教-福州大学）
- [谭逸珂](https://github.com/LikeGiver)（内容创作者-对外经济贸易大学）
- [王熠明](https://github.com/Bald0Wang)（内容创作者-Datawhale成员）
- [何至轩](https://github.com/pod2c)（内容创作者-鲸英助教）
- [康婧淇](https://github.com/jodie-kang)（内容创作者-Datawhale成员）
- [三水](https://github.com/sssanssss)（内容创作者-鲸英助教）
- [杨晨旭](https://github.com/langlibai66)（内容创作者-太原理工大学-鲸英助教）
- [赵伟](https://github.com/2710932616)（内容创作者-鲸英助教）
- [苏向标](https://github.com/gzhuuser)（内容创作者-广州大学-鲸英助教）
- [陈睿](https://github.com/riannyway)（内容创作者-西交利物浦大学-鲸英助教）
- [林恒宇](https://github.com/LINHYYY)（内容创作者-广东东软学院-鲸英助教）

> Note: Ranking is based on the level of contribution.

### Others

- Special thanks to [@Sm1les](https://github.com/Sm1les) for their help and support for this project.
- Some LoRA code and explanations are referenced from the repository: https://github.com/zyds/transformers-code.git
- If you have any ideas, feel free to contact us at DataWhale. We also welcome everyone to raise issues!
- Special thanks to the following students who contributed to the tutorials!


<div align=center style="margin-top: 30px;">
  <a href="https://github.com/datawhalechina/self-llm/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=datawhalechina/self-llm" />
  </a>
</div>

### Star History

<div align=center style="margin-top: 30px;">
  <img src="./images/star-history-2024129.png"/>
</div>
