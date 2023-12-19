# 09-Qwen-1_8B-chat CPU 部署

## 概述

本文介绍了在 Intel 设备上部署 Qwen 1.8B 模型的过程，你需要至少16GB内存的机器来完成这项任务，我们将使用英特尔的大模型推理库 [BigDL](https://github.com/intel-analytics/BigDL) 来实现完整过程。

Bigdl-llm 是一个在英特尔设备上运行 LLM（大语言模型）的加速库，通过 INT4/FP4/INT8/FP8 精度量化和架构针对性优化以实现大模型在 英特尔 CPU、GPU上的低资源占用与高速推理能力（适用于任何 PyTorch 模型）。

本文演示为了通用性，只涉及 CPU 相关的代码，如果你想学习如何在 Intel GPU 上部署大模型，可以参考[官网文档](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html)。

## 环境配置

在开始之前，我们需要准备好 bigdl-llm 以及之后部署的相关运行环境，我们推荐你在 python 3.9 的环境中进行之后的操作。

如果你发现下载速度过慢，可以尝试更换默认镜像源：`pip config set global.index-url https://pypi.doubanio.com/simple`


```python
%pip install --pre --upgrade bigdl-llm[all] 
%pip install gradio 
%pip install hf-transfer
%pip install transformers_stream_generator einops
%pip install tiktoken
```

## 模型下载

首先，我们通过 huggingface-cli 获取 qwen-1.8B 模型，耗时较长需要稍作等待；这里考虑到国内的下载限制增加了环境变量加速下载。


```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 下载模型
os.system('huggingface-cli download --resume-download qwen/Qwen-1_8B-Chat --local-dir qwen18chat_src')
```

## 保存量化模型

为了实现大语言模型的低资源消耗推理，我们首先需要把模型量化到 int4 精度，随后序列化保存在本地的相应文件夹方便重复加载推理；利用 `save_low_bit` api 我们可以很容易实现这一步。


```python
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import  AutoTokenizer
import os
if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(),"qwen18chat_src")
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.save_low_bit('qwen18chat_int4')
    tokenizer.save_pretrained('qwen18chat_int4')
```

## 加载量化模型

保存 int4 模型文件后，我们便可以把他加载到内存进行进一步推理；如果你在本机上无法导出量化模型，也可以在更大内存的机器中保存模型再转移到小内存的端侧设备中运行，大部分常用家用PC即可满足 int4 模型实际运行的资源需求。



```python
import torch
import time
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

QWEN_PROMPT_FORMAT = "<human>{prompt} <bot>"
load_path = "qwen18chat_int4"
model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)

input_str = "给我讲一个年轻人奋斗创业最终取得成功的故事"
with torch.inference_mode():
    prompt = QWEN_PROMPT_FORMAT.format(prompt=input_str)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    st = time.time()
    output = model.generate(input_ids,
                            max_new_tokens=512)
    end = time.time()
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f'Inference time: {end-st} s')
    print('-'*20, 'Prompt', '-'*20)
    print(prompt)
    print('-'*20, 'Output', '-'*20)
    print(output_str)
```

## gradio-demo 体验

为了得到更好的多轮对话体验，这里还提供了一个简单的 `gradio` demo界面方便调试使用，你可以修改内置 `system` 信息甚至微调模型让本地模型更接近你设想中的大模型需求。



```python
import gradio as gr
import time
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

QWEN_PROMPT_FORMAT = "<human>{prompt} <bot>"

load_path = "qwen18chat_int4"
model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(load_path,trust_remote_code=True)

def add_text(history, text):
    _, history = model.chat(tokenizer, text, history=history)
    return history, gr.Textbox(value="", interactive=False)

def bot(history):
    response =  history[-1][1]
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [], 
        elem_id="chatbot",
        bubble_full_width=False,
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)

demo.queue()
demo.launch()
```

利用 Intel 的大语言模型推理框架，我们可以实现大模型在 Intel 端侧设备的高性能推理。 只需要 2G 内存占用就可以实现与本地大模型的流畅对话，一起来体验下吧。
