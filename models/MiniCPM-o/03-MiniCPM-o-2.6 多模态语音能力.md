# MiniCPM-o-2.6 多模态语音能力

## 环境准备  

本文基础环境如下：

```
----------------
ubuntu 22.04
python 3.12
cuda 12.1
pytorch 2.5.1
----------------
```

> 本文默认学习者已配置好以上 `Pytorch (cuda)` 环境，如未配置请先自行安装。

首先 `pip` 换源加速下载并安装依赖包

```bash
python -m pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.20.0
pip install transformers==4.44.2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install accelerate==1.2.1 timm==0.9.10 soundfile==0.12.1 librosa==0.9.0 vector-quantize-pytorch==1.18.5 vocos==0.1.0
```

## 模型下载  

使用 `modelscope` 中的 `snapshot_download` 函数下载模型，第一个参数为模型名称，参数 `cache_dir`为模型的下载路径。

先切换到 `autodl-tmp` 目录，`cd /root/autodl-tmp`

然后新建名为 `model_download.py` 的 `python` 脚本，并在其中输入以下内容并保存

```python  
from modelscope import snapshot_download

snapshot_download('OpenBMB/MiniCPM-o-2_6', local_dir='/root/autodl-tmp/MiniCPM-o-2_6')
```

然后在终端中输入 `python model_download.py` 执行下载，这里需要耐心等待一段时间直到模型下载完成。

> 注意：记得修改 `local_dir` 为你的模型本地下载路径哦~

## **多模态语音能力**

MiniCPM-o-2.6具有一定的多模态语音理解能力，在新的语音模式中，MiniCPM-o 2.6**支持可配置声音的中英双语语音对话，还具备情感/语速/风格控制、端到端声音克隆、角色扮演等进阶能力**。所以接下来一起来探索一下MiniCPM多模态的语音能力。可以准备一个jupyter或者python脚本运行以下代码。

### **模型初始化**

MiniCPM-o 默认加载是 omni 模型，初始化视觉，音频和TTS模块，也可以根据需求选择初始化

```python
import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
import librosa

# 加载 omni 模型，默认会初始化视觉、音频和 TTS 模块
# 如果只加载视觉模型，设置 init_audio=False 和 init_tts=False
# 如果只加载音频模型，设置 init_vision=False
model = AutoModel.from_pretrained(
    '/root/autodl-tmp/MiniCPM-o-2_6/',
    trust_remote_code=True,
    attn_implementation='sdpa', # 使用 sdpa 或 flash_attention_2
    torch_dtype=torch.bfloat16,
    init_vision=True,
    init_audio=True,
    init_tts=True
)

model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/MiniCPM-o-2_6/', trust_remote_code=True)

# 除了视觉模式，TTS 处理器和 vocos 也需要初始化
model.init_tts()
```

### **语音模仿**

语音模仿任务反映了模型的端到端语音建模能力。模型接收音频输入，输出 ASR 转录，然后重建原始音频，高度相似。重建的音频与原始音频之间的相似度越高，模型在端到端语音建模方面的基础能力就越强。

```python
mimick_prompt = "请重复每个用户的讲话内容，包括语音风格和内容。"
audio_input, _ = librosa.load('/root/autodl-tmp/MiniCPM-o-2_6/assets/mimick.wav', sr=16000, mono=True)
msgs = [{'role': 'user', 'content': [mimick_prompt,audio_input]}]

res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    temperature=0.3,
    generate_audio=True,
    output_audio_path='output.wav', # 将 TTS 结果保存到 output_audio_path
)
print(res)
```

```bash
OmniOutput(text='全民制作人们大家好，我是练习时长两年半的个人练习生蔡徐坤，喜欢唱跳rap、篮球music.', spk_embeds=None, audio_wav=tensor([-5.5982e-03, -4.1898e-03, -4.9984e-03,  ..., -3.4108e-05,
         2.6977e-05,  1.9374e-05]), sampling_rate=24000)
```

[mimick.webm](https://github.com/user-attachments/assets/9c9cc796-6910-48f3-94f7-c83a40c672f6)

### **音频角色扮演/助手**

同时还可以设置`mode`参数为`audio_roleplay`或者`audio_assistant`来进行音频角色扮演或者助手，一般对话推荐`audio_assistant`。

```python
ref_audio, _ = librosa.load('/root/autodl-tmp/MiniCPM-o-2_6/assets/demo.wav', sr=16000, mono=True) # 加载参考音频

# 音频角色扮演:  # 在此模式下，模型会根据音频提示进行角色扮演。 (更自然的对话，但不稳定)
# sys_prompt = model.get_sys_prompt(ref_audio=ref_audio, mode='audio_roleplay', language='en')
# user_question = {'role': 'user', 'content': [librosa.load('xxx.wav', sr=16000, mono=True)[0]]}

# 音频助手: # 在此模式下，模型会以参考音频中的语音作为 AI 助手进行对话。 (稳定，更适合一般对话)
sys_prompt = model.get_sys_prompt(ref_audio=ref_audio, mode='audio_assistant', language='zh') 
user_question = {'role': 'user', 'content': [librosa.load('/root/autodl-tmp/MiniCPM-o-2_6/assets/qa.wav', sr=16000, mono=True)[0]]}

msgs = [sys_prompt, user_question]
# 第一轮
res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result.wav',
)

# # 第二轮
# history = msgs.append({'role': 'assistant', 'content': res})
# user_question = {'role': 'user', 'content': [librosa.load('xxx.wav', sr=16000, mono=True)[0]]}
# msgs = history.append(user_question)
# res = model.chat(
#     msgs=msgs,
#     tokenizer=tokenizer,
#     sampling=True,
#     max_new_tokens=128,
#     use_tts_template=True,
#     generate_audio=True,
#     temperature=0.3,
#     output_audio_path='result_round_2.wav',
# )
print(res)
```

```bash
OmniOutput(text='大家好，我是面壁智能小钢炮。祝大家新年快乐！', spk_embeds=None, audio_wav=tensor([-4.9051e-05, -4.5966e-05, -7.2269e-05,  ...,  1.0666e-03,
         1.5180e-03,  1.1057e-03]), sampling_rate=24000)
```

[audio_assistant.webm](https://github.com/user-attachments/assets/a15d719e-d909-4f05-9a24-7061bbd8ade8)

### **多种音频任务**

除此之外，还能做以下的音频任务，包括音频理解，语音生成，语音克隆等

#### 音频理解

```python
# 音频理解任务Prompt
# 语音:
#     ASR with ZH(same as AST en2zh): 请仔细听这段音频片段，并将其内容逐字记录。
#     ASR with EN(same as AST zh2en): Please listen to the audio snippet carefully and transcribe the content.
#     Speaker Analysis 说话人分析: Based on the speaker's content, speculate on their gender, condition, age range, and health status.
# 一般音频:
#     音频总结: Summarize the main content of the audio. / 总结音频的主要内容。
#     声音场景标记: Utilize one keyword to convey the audio's content or the associated scene. / 使用一个关键词表达音频内容或相关场景。

task_prompt = "Summarize the main content of the audio. \n" # 选择上面的任务Prompt
audio_input, _ = librosa.load('/root/autodl-tmp/MiniCPM-o-2_6/assets/audio_understanding.mp3', sr=16000, mono=True)

msgs = [{'role': 'user', 'content': [task_prompt,audio_input]}]

res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result.wav',
)
print(res)
```

```bash
OmniOutput(text='Birds are chirping and singing, with a distant sound of an emergency vehicle siren.', spk_embeds=None, audio_wav=tensor([-0.0009, -0.0009, -0.0008,  ..., -0.0018,  0.0007, -0.0008]), sampling_rate=24000)
```

[audio_understanding.webm](https://github.com/user-attachments/assets/a944710b-8411-43d1-a28f-738df5fae48a)

#### 语音生成

```python
'''
语音生成任务 Speech Generation Task Prompt:
    Human Instruction-to-Speech: see https://voxinstruct.github.io/VoxInstruct/
    Example:
        # 在新闻中，一个年轻男性兴致勃勃地说：“祝福亲爱的祖国母亲美丽富强！”他用低音调和低音量，慢慢地说出了这句话。
        # Delighting in a surprised tone, an adult male with low pitch and low volume comments:"One even gave my little dog a biscuit" This dialogue takes place at a leisurely pace, delivering a sense of excitement and surprise in the context. 

    Voice Cloning or Voice Conversion: With this mode, model will act like a TTS model. 
'''
# Human Instruction-to-Speech:
task_prompt = '在新闻中，一个年轻男性兴致勃勃地说：“祝福亲爱的祖国母亲美丽富强！”他用低音调和低音量，慢慢地说出了这句话。' #Try to make some Human Instruction-to-Speech prompt (Voice Creation)
msgs = [{'role': 'user', 'content': [task_prompt]}] # you can also try to ask the same audio question

res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result.wav',
)
print(res)
# IPython.display.Audio('result.wav')
```

```bash
OmniOutput(text='祝福亲爱的祖国母亲美丽富强', spk_embeds=None, audio_wav=tensor([-4.8119e-03, -3.6808e-03, -2.6445e-03,  ..., -6.3839e-05,
         6.8862e-06, -1.4487e-04]), sampling_rate=24000)
```

[voice_generation.webm](https://github.com/user-attachments/assets/a1aa8738-27fb-42ff-9f27-c1cfd7d5dc43)

#### 语音克隆

```python
ref_audio, _ = librosa.load('/root/autodl-tmp/MiniCPM-o-2_6/assets/mimick.wav', sr=16000, mono=True) # 加载参考音频

# 声音克隆模式: 
sys_prompt = model.get_sys_prompt(ref_audio=ref_audio, mode='voice_cloning', language='zh')
text_prompt = f"Please read the text below."
user_question = {'role': 'user', 'content': [text_prompt, "全名制作人们大家好，我是小黑子，鸡你太美"]}
msgs = [sys_prompt, user_question]
res = model.chat(
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result.wav',
)

print(res)
```

```bash
OmniOutput(text='全名制作人们大家好，我是小黑子，鸡你太美', spk_embeds=None, audio_wav=tensor([-0.0197,  0.0294,  0.0064,  ..., -0.0006, -0.0006, -0.0005]), sampling_rate=24000)
```

[voice_clone.webm](https://github.com/user-attachments/assets/fd6409f7-2a4e-4d72-a2aa-29575060decc)