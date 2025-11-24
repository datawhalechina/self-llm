# 04-Qwen3-VL-4B-Instruct-vLLM



## **Introduction to vLLM**

`vLLM` is a high-performance large language model inference and serving framework with the following features:

- Efficient KV Cache and Memory Management: Based on `PagedAttention`, it significantly reduces video memory waste and improves throughput in long text and high concurrency scenarios.
- Compatible with OpenAI Interface: It can directly provide `completions` and `chat completions` capabilities in the form of `OpenAI API`, facilitating integration with the existing ecosystem.
- Multi-GPU Parallelism and Easy Expansion: Supports strategies such as Tensor Parallel, with simple parameters, making it easy to horizontally expand throughput and context length limits.
- Good Ecosystem: Seamlessly connects with `HuggingFace`/`ModelScope` model repositories, supporting various inference optimizations and features (such as inference/thinking content parsing, tool calling).



## Environment Preparation

The recommended basic environment is as follows:

> Video Memory Budget Suggestion: 1 * NVIDIA RTX 5090

```
----------------
ubuntu 22.04
python 3.12
cuda 12.8
pytorch 2.8.0 
----------------
```

Virtual Environment Configuration

```bash
pip install vllm==0.11.0
pip install openai==2.3.0
pip install modelscope==1.30.0
pip install qwen_vl_utils==0.0.14
```

> Tip: Please ensure that the NVIDIA driver, CUDA, and PyTorch CUDA compilation versions in the environment match. You can use `nvidia-smi` and `python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"` for a quick self-check.



## Model Download

```python
# model_download.py
# Note: Modify cache_dir to the path where you want to save the model
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-VL-4B-Instruct', cache_dir='Please modify me!!!', revision='master')

print(f"Model download completed, saved path: {model_dir}")
```



## **Model Introduction**

![fig-4-2](./images/fig-4-5.jpg)

`Qwen3-VL-4B-Instruct` is a vision-language model in the [Qwen3-VL](https://www.modelscope.cn/collections/Qwen3-VL-5c7a94c8cb144b) series, possessing excellent text understanding and generation capabilities, visual perception and reasoning capabilities, as well as spatial and video dynamic understanding capabilities.
Evaluation results show that this model is comparable to `GPT-5-Mini` and `Cluade-4-Sonnet` in multiple tasks such as STEM, VQA, OCR, video understanding, and agents.
In addition, this model is a non-reasoning model. If you want to experience a stronger model with thinking mode, you can download the [Qwen3-VL-30B-A3B-Thinking](https://www.modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Thinking) of the same size or the larger flagship model [Qwen3-VL-235B-A22B-Thinking](https://www.modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8).

<p align="center">   
    <img src="./images/fig-4-1.jpg" width="500" /> 
    <img src="./images/fig-4-4.jpg" width="500" /> 
</p>


## **vLLM Serving**

### **Python Command Line Start Service**

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Please modify me!!!/Qwen/Qwen3-VL-4B-Instruct \
  --served-model-name Qwen3-VL-4B-Instruct \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --port 8085 \
  --trust_remote_code \
  --gpu_memory_utilization 0.9
```

#### Parameter Explanation

- `CUDA_VISIBLE_DEVICES`: Specify visible GPUs.

- `--tensor-parallel-size`: Tensor parallel division number. Used when using multiple GPUs; multi-card can improve throughput and available context length limit.
- `--max-model-len`: Maximum context length for a single request (input + output). Larger values occupy more video memory and are prone to triggering OOM. Can be adjusted downwards according to video memory conditions, such as 8192.
- `--gpu_memory_utilization`: vLLM target available video memory ratio (0~1). If OOM occurs, try increasing it.
- `--served-model-name`: The model name exposed externally. The client needs to use the same name `model` to call.
- `--port`/`--host`: Service listening port/address, default is 8080.
- `--trust_remote_code`: Allow loading custom code from the repository (required, otherwise some models cannot be initialized correctly).

After successful startup, you will see the output `Application startup complete` as shown in the figure:

![fig-4-2](./images/fig-4-2.png)

The service started by vLLM above is compatible with the OpenAI interface, so it can be conveniently called through Python's OpenAI library. Below we test the capabilities of `Qwen3-VL-4B-Instruct` through practical cases of daily Q&A, image description, and video reasoning.



### **Daily and Image Description Test**

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8085/v1" # Use the correct port
daily_chat_message = "Write the sentence 'I love Qwen3-VL-4B-Instruct' in reverse."

# Instantiate OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Daily Q&A
daily_chat_response = client.chat.completions.create(
    model="Qwen3-VL-4B-Instruct",
    messages = [
        {
            "role": "user",
            "content": daily_chat_message
        }
    ]
)
print(f"Qwen3-VL-4B-Instruct Daily Q&A: {daily_chat_response.choices[0].message.content}")
print("-"*100)

# Image Description
image_des_response = client.chat.completions.create(
    model="Qwen3-VL-4B-Instruct",
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    }
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
)
print(f"Qwen3-VL-4B-Instruct Image Description: {image_des_response.choices[0].message.content}")

```


Test results are as follows:
````bash
Qwen3-VL-4B-Instruct Daily Q&A: Let's write the sentence "I love Qwen3-VL-4B-Instruct" in reverse step by step.

Writing all the content of the sentence "I love Qwen3-VL-4B-Instruct" in reverse, i.e., character by character reversal, the result is as follows:

tcurtsnI-B4-LV-3newQ evol I

----------------------------------------------------------------------------------------------------
Qwen3-VL-4B-Instruct Image Description: Of course. Here is a detailed description of the image.

This is a heartwarming and serene photograph capturing a tender moment between a woman and her dog on a sandy beach during what appears to be sunrise or sunset.

- **Main Subjects and Interaction:** The central focus is a woman and a large, light-colored dog, likely a yellow Labrador Retriever, sitting on the sand. They are engaged in a playful and affectionate interaction. The dog is sitting upright, extending its right front paw to meet the woman's hand in a "high-five" gesture. The woman, sitting cross-legged, is smiling warmly and looking at the dog, reciprocating the high-five. Her expression conveys joy and affection.

- **Setting and Environment:** The scene is set on a wide, open beach. The sand is light-colored and appears soft, with gentle ripples and footprints visible. In the background, the ocean stretches out to the horizon, with a small, gentle wave breaking near the shore. The overall atmosphere is peaceful and tranquil.

- **Lighting and Atmosphere:** The image is bathed in the warm, golden light of the sun, which is low in the sky, likely just above the horizon. This creates a beautiful lens flare and a soft, hazy glow, particularly on the right side of the image where the sun is positioned. The light illuminates the woman's hair and the dog's fur, creating a warm and inviting ambiance. The sky is a bright, pale white, indicating the intensity of the sunlight.

- **Details and Attire:** The woman has long, dark brown hair and is wearing a black and white plaid flannel shirt over dark pants. She is barefoot, which adds to the relaxed and natural feel of the scene. The dog is wearing a blue harness adorned with a pattern of small, colorful paw prints. A red leash lies on the sand near the dog.

- **Composition:** The subjects are positioned slightly off-center, creating a balanced and dynamic composition. The shallow depth of field keeps the woman and dog in sharp focus while softly blurring the background, which draws the viewer's attention to their interaction. The overall mood of the image is one of happiness, companionship, and the simple joy of a shared moment with a beloved pet.
````

### **Analysis**

#### Q1

<p align="center">   
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" width="800" /> 
</p>

#### Q2

`I love Qwen3-VL-4B-Instruct` → `tcurtsnI-B4-LV-3newQ evol I`

After manual verification, the responses of `Qwen3-VL-4B-Instruct` to these two questions are quite good. The reversed sentence is completely correct, and the image description is very detailed and clear, consistent with the original image. For comparison of the same image description with the test results of `Qwen2-VL`, please refer to [self-llm Qwen2-VL-2B-Instruct FastApi Deployment and Invocation](https://github.com/datawhalechina/self-llm/blob/8f77c691403bc2a5be12dad994f1d60b67c15618/models/Qwen2-VL/01-Qwen2-VL-2B-Instruct%20FastApi%20%E9%83%A8%E7%BD%B2%E8%B0%83%E7%94%A8.md?plain=1#L180).

| Evaluation Dimension | Qwen3-VL-4B-Instruct | Qwen2-VL-2B-Instruct |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Detail Richness and Image Restoration | Provided extremely detailed visual details: including the dog's breed (Yellow Labrador), the woman's hair color (dark brown), clothing (black and white plaid flannel shirt + dark pants), the dog's accessories (blue harness with colorful paw prints), the color of the leash (red), and environmental details such as footprints and ripples on the beach. This information not only enhances the realism of the picture but also helps the reader accurately construct the image in their mind. | Although basic elements were mentioned (such as plaid shirt, raised dog paw, ocean background), the description was relatively general, lacking specific features (such as dog breed, color, accessories, etc.), and the visual sense was weak. |
| Emotion and Atmosphere Creation | Not only described "smiling", but also deeply portrayed the character's emotions ("warm smile", "joy and affection") and the overall atmosphere ("peaceful", "warm", "simple happy companionship"), and reinforced the emotional tone through lighting ("golden morning/evening light", "soft glow", "lens flare"). | Although "calm and joyful" was mentioned, the emotional expression was relatively superficial, lacking delicate emotional layers and atmospheric rendering. |
| Structure and Logic | Adopted a clear segmented structure (subject interaction, environment, lighting, attire, composition), with rigorous logic and clear hierarchy, facilitating readers to systematically understand the image content. | It was a one-paragraph narrative, with piled-up information and lack of organization, making the reading experience less smooth than the former. |
| Language Expressiveness | Used more literary and visual vocabulary, such as "bathed in golden sunlight", "shallow depth of field highlights the subject", "gentle waves breaking on the shore", etc., making the language vivid and infectious. | The language was relatively plain, leaning towards functional description, lacking beauty and infectiousness. |

`Qwen3-VL-4B-Instruct` is significantly better than `Qwen2-VL-2B-Instruct` in four aspects: **detail restoration, emotional expression, structural organization, and language expressiveness**. It not only accurately conveyed the image content but also successfully evoked the reader's emotional resonance, and is obviously of higher quality as an image description.



### Video Reasoning Test

Input video preview is as follows:

![fig-4-3](./images/fig-4-3.gif)

```python
import torch
from qwen_vl_utils import process_vision_info
from modelscope import AutoProcessor
from vllm import LLM, SamplingParams

import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


if __name__ == '__main__':
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "./space_woaudio.mp4",
                },
                {"type": "text", "text": "Describe this video. And guess what is the man going to do?"},
            ],
        }
    ]

    checkpoint_path = "Please modify me!!!/Qwen/Qwen3-VL-4B-Instruct"
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]

    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.99,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=128,
        top_k=-1,
        stop_token_ids=[],
    )

    for i, input_ in enumerate(inputs):
        print()
        print('=' * 40)
        print(f"Inputs[{i}]: {input_['prompt']=!r}")
    print('\n' + '>' * 40)

    outputs = llm.generate(inputs, sampling_params=sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print()
        print('=' * 40)
        print(f"Generated Response: {generated_text!r}")
```

Test Results:

```bash
========================================
Inputs[0]: input_['prompt']='<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Describe this video. And guess what is the man going to do?<|im_end|>\n<|im_start|>assistant\n'

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Adding requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.58s/it]
Processed prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.64s/it, est. speed input: 1270.83 toks/s, output: 77.94 toks/s]

========================================
Generated Response: "Based on the provided video frames, here is a detailed description and a logical guess about the man's next action.\n\n### Video Description\n\nThe video is set inside a high-tech **Mission Control Center**, as indicated by the prominent sign above the main display. The environment is filled with advanced technology, suggesting a setting for monitoring and managing a space mission.\n\n- **The Man:** A middle-aged man with short, graying hair is the central figure. He is dressed in a dark blue polo shirt with a small NASA logo on the left chest and khaki pants. He is actively speaking and gesturing with both hands, indicating he is giving..." (Note: Truncated due to exceeding `max_tokens`)
```

> Note: Video reasoning requires a large amount of video memory, and may require setting **relatively extreme inference parameters**, such as max_tokens=128, gpu_memory_utilization=0.99 in the above code, which is also the reason why the test was truncated. However, it can still be seen from the generated content that the model has good video understanding and reasoning capabilities.

