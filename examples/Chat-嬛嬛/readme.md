# How Chat-Huanhuan Was Made

**Chat-Zhenhuan** is a chat language model that mimics the tone of Zhenhuan, obtained by **LoRA fine-tuning** based on a large model using all the lines and sentences about Zhenhuan in the script of "Empresses in the Palace" (Zhenhuan Zhuan).

> Zhenhuan is the female lead and core heroine in the novel "Empresses in the Palace" and the TV series "Empresses in the Palace". Originally named Zhen Yuhuan, she changed her name to Zhenhuan because she disliked the vulgarity of the character "Yu". She was the daughter of Zhen Yuandao, a Han Chinese. Later, she was given the surname Niohuru by Yongzheng, raised to the Manchu Bordered Yellow Banner, and named "Niohuru Zhenhuan". She participated in the draft with Shen Meizhuang and An Lingrong and was selected because she looked like the Pure Yuan Empress. After entering the palace, facing the step-by-step persecution of Consort Hua, the injustice of Shen Meizhuang, and the betrayal of An Lingrong, she changed from a green girl who was content with a corner to a master of palace fighting who could cause a bloody storm. After Yongzheng discovered the ambition of the Nian family, he ordered his father Zhen Yuandao to eliminate it. Zhenhuan also used her serial tricks in the harem to help the emperor solve his political enemies, so she was deeply loved by Yongzheng. After several twists and turns, she finally defeated the arrogant and domineering Consort Hua. When Zhenhuan was titled Consort, she was plotted against by Empress Yixiu and was disliked by the emperor. After giving birth to her daughter Longyue, she was disheartened and asked to leave the palace to become a nun. However, she was loved by Prince Guo, and the two fell in love. After learning of Prince Guo's death, she immediately designed to meet Yongzheng again and returned to the palace in glory. Since then, the unjust case of Zhen's father was rehabilitated, the Zhen clan rose again, and she also gave birth to twins. She escaped Yixiu's assassination in various conspiracies such as the blood test to verify relatives, and finally overthrew the Empress behind the scenes by sacrificing her own biological fetus. But Yongzheng forced Zhenhuan to poison Yunli to test Zhenhuan's sincerity, and asked Zhenhuan, who had already given birth to a child, to go to the Dzungar for a marriage alliance. Zhenhuan then regarded the emperor as the object that should be destroyed the most. In the grand finale, she said that "all human struggles are caused by the injustice of the rulers" and poisoned Yongzheng. The fourth prince Hongli ascended the throne as Qianlong, and Zhenhuan was honored as the Holy Mother Empress Dowager, with power over the court and the wild, and spent her old age in peace in Ruyi's Royal Love in the Palace.

Chat-Zhenhuan realizes a complete process of creating a **personalized AI** fine-tuned large model based on novels and scripts, taking "Empresses in the Palace" as the entry point. By providing any novel or script and specifying a character, running the complete process of this project allows every user to create a personalized AI that belongs to them, fits the character persona, and possesses high intelligence based on their favorite novels and scripts.

> *Chat-Huanhuan model cumulative downloads 15.6k, Modelscope Address:* [*Link*](https://www.modelscope.cn/models/kmno4zx/huanhuan-chat-internlm2)   
> *Chat-Huanhuan has accumulated 500 stars, huahuan-chat Project Address:* [*Link*](https://github.com/KMnO4-zx/huanhuan-chat.git), xlab-huanhuan-chat Project Address: [*Link*](https://github.com/KMnO4-zx/xlab-huanhuan.git)  


***OK, next I will lead you to do it yourself, step by step to realize the training process of Chat-Zhenhuan, let's experience it together~***

## Step 1: Environment Preparation

The basic environment of this article is as follows:

```
----------------
ubuntu 22.04
python 3.12
cuda 12.1
pytorch 2.3.0
----------------
```
> This article assumes that the learner has installed the above Pytorch (cuda) environment. If not, please install it yourself.

First, change the `pip` source to accelerate the download and install the dependency packages.

```shell
# Upgrade pip
python -m pip install --upgrade pip
# Change pypi source to accelerate library installation
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope==1.16.1
pip install transformers==4.43.1
pip install accelerate==0.32.1
pip install peft==0.11.1
pip install datasets==2.20.0
```

## Step 2: Data Preparation

First, we need to prepare the script data of "Empresses in the Palace". Here we used the script data of "Empresses in the Palace". We can check the format of the raw data.

```text
Act 2
(Court is adjourned, officials disperse)
Official A: Our Emperor really values General Nian and Lord Longkeduo.
Official B: Lord Longkeduo, congratulations! You are a great contributor to the country!
Official C: General Nian, the Emperor favors you very much!
Official D: Lord Nian, you are the Emperor's right-hand man!
Su Peisheng (catching up with Nian Gengyao): General Nian, please stay. General——
Nian Gengyao: Eunuch Su, what advice do you have?
Su Peisheng: I dare not. The Emperor is worried about your arm injury, General, and specifically asked this slave to give this secret golden wound ointment to you, my lord, and asked you to use it.
Nian Gengyao (cupping his hands towards the Golden Hall from afar): I, Nian Gengyao, thank the Emperor for his grace! Dare I ask Eunuch Su, how is my little sister in the palace today?
Su Peisheng: Consort Hua is graceful and favored above the six palaces, General, you can rest assured.
Nian Gengyao: Then I will trouble Eunuch Su. (Turns and leaves)
Su Peisheng: It is my duty.
```

Each sentence has a character and the corresponding line, so it is very simple to process these data into a dialogue format, as follows:

```
[
    {"rloe":"Official A", "content":"Our Emperor really values General Nian and Lord Longkeduo."},
    {"rloe":"Official B", "content":"Lord Longkeduo, congratulations! You are a great contributor to the country!"},
    {"rloe":"Official C", "content":"General Nian, the Emperor favors you very much!"},
    {"rloe":"Official D", "content":"Lord Nian, you are the Emperor's right-hand man!"},
    {"rloe":"Su Peisheng", "content":"General Nian, please stay. General——"},
    ...
]
```

Then extract the dialogue of the character we are interested in to form QA pairs. For such data, we can use regular expressions or other methods for quick extraction and extract the dialogue of the character we are interested in.

Then in many cases, we don't have such excellent script format data. So we may need to extract character dialogue data from a large piece of text, and then convert it into the format we need.

For example, "Journey to the West Vernacular", we can see that its text is like this. For such text, we need to leverage the ability of large models to extract characters and corresponding dialogues from the text. Then filter out the character dialogues we need.

> You can use a small tool: [*extract-dialogue*](https://github.com/KMnO4-zx/extract-dialogue.git) to extract dialogue from text.
    
```
......
Originally, after Sun Wukong left, a Demon King of Confusion monopolized the Water Curtain Cave and snatched away many monkeys. When Sun Wukong heard this, he gritted his teeth and stamped his feet in anger. He asked clearly about the residence of the Demon King of Confusion, decided to seek revenge on the Demon King of Confusion, and flew towards the north on the somersault cloud.

In a short while, Sun Wukong arrived in front of the Water Dirty Cave of the Demon King of Confusion and shouted to the little demons in front of the door: "That dog-fart Demon King of your family has bullied our monkeys many times. I am here today to compete with that Demon King!"

The little demon ran into the cave and reported to the Demon King. The Demon King hurriedly put on iron armor, carried a big knife, and walked out of the cave door surrounded by little demons.

Sun Wukong, unarmed, snatched the big knife of the Demon King of Confusion and split him in half. Then, he pulled out a handful of hair, chewed it up and sprayed it out. The hair turned into many little monkeys, who killed their way into the cave, killed all the demons, then rescued the snatched little monkeys, and set a fire to burn the Water Dirty Cave.
......
```

> Chat-Zhenhuan raw data: [*Empresses in the Palace*](https://github.com/KMnO4-zx/huanhuan-chat/tree/master/dataset/input/huanhuan)  
> Journey to the West Vernacular raw data: [*Journey to the West*](https://github.com/KMnO4-zx/huanhuan-chat/blob/master/dataset/input/wukong/%E8%A5%BF%E6%B8%B8%E8%AE%B0%E7%99%BD%E8%AF%9D%E6%96%87.txt)

Finally, organize it into `json` format data, as follows:

```
[
    {
        "instruction": "Miss, other show girls are begging to be selected, only our Miss wants to be dropped. The Bodhisattva must remember it clearly——",
        "input": "",
        "output": "Shh——It is said that if you say a wish out loud, it won't work."
    },
    {
        "instruction": "This Imperial Physician Wen is also strange. Who doesn't know that imperial physicians cannot treat people outside the royal family without the Emperor's order. He is good, running to our mansion every ten days or half a month.",
        "input": "",
        "output": "You two talk too much. I should ask Imperial Physician Wen for a dose of medicine to treat you well."
    },
    {
        "instruction": "Sister Huan, I just went to the mansion to feel the pulse, and heard Aunt Zhen say that you came here to offer incense.",
        "input": "",
        "output": "Coming out for a walk is also a distraction."
    }
]
```

> Chat-Huanhuan Data: [*chat-Zhenhuan*](https://github.com/datawhalechina/self-llm/blob/master/dataset/huanhuan.json)

So, the general idea of processing data in this step is:

***1. Extract characters and dialogues from raw data &emsp;2. Filter out the dialogues of the characters we are interested in &emsp;3. Convert dialogues into the format we need***

> *This step can also add a data augmentation link, such as using two or three pieces of data as examples to throw to the LLM, letting it generate data with a similar style. Or you can also find some daily dialogue datasets and use RAG to generate some fixed character style dialogue data. Everyone can boldly try this out completely!*

## Step 3: Model Training

Everyone may be familiar with this step. In every model of `self-llm`, there will be a `Lora` fine-tuning module. We only need to process the data into the format we need, and then call our training script.

Here we choose the `LLaMA3_1-8B-Instruct` model for fine-tuning. First, we still need to download the model. Create a `model_download.py` file and enter the following content:

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download('LLM-Research/Meta-Llama-3.1-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')
```

> Note: Remember to modify `cache_dir` to your model download path~

Secondly, prepare the training code. For students familiar with `self-llm`, this step may be simple. Here I will place `train.py` in the current directory. Everyone can modify the dataset path and model path in it.

> *Of course, you can also use the `lora` fine-tuning tutorial in `self-llm`. Tutorial address: [Link](https://github.com/datawhalechina/self-llm/blob/master/models/LLaMA3/04-LLaMA3-8B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md)*

Run the following command in the command line:
    
```shell
python train.py
```

> *Note: Remember to modify the dataset path and model path in `train.py`~*

Training will take about *20 ~ 30* minutes. After training is completed, a `lora` model will be generated in the `output` directory. You can use the following code to test:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = './LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = './output/llama3_1_instruct_lora/checkpoint-699' # Change this to your lora output corresponding checkpoint address

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# Load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "Huanhuan, what's wrong with you? I will fight for you!"

messages = [
        {"role": "system", "content": "Assume you are the woman beside the Emperor -- Zhenhuan."},
        {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# print(input_ids)

model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('Emperor:', prompt)
print('Huanhuan:', response)
```

```
Emperor: Huanhuan, what's wrong with you? I will fight for you!
Huanhuan: Emperor, this concubine didn't mean it.
```

Next, we can use this Huanhuan model to have a conversation~   
Interested students can try to use other models in `self-llm` for fine-tuning to verify your learning results!

## Final Words

*Chat-Huanhuan was created when LLMs just became popular last year. We felt that if we didn't do something, we might miss a lot of interesting things. So I worked with a few friends and spent a lot of time making this project. In this project, we learned a lot and encountered many problems, but we solved them one by one. And Chat-Huanhuan also won awards, allowing the project to gain a lot of attention. So, I think this project is very meaningful and very interesting.*

- *2023 iFlytek Spark Cup Cognitive Large Model Scene Innovation Competition Top 50*
- *2024 InternLM Large Model Challenge (Spring Competition) Creative Application Award Top 12*

### Chat-Huanhuan Contributors

- [Song Zhixue](https://github.com/KMnO4-zx) (Datawhale Member - China University of Mining and Technology (Beijing))
- [Zou Yuheng](https://github.com/logan-zou) (Datawhale Member - University of International Business and Economics)
- [Wang Yiming](https://github.com/Bald0Wang) (Datawhale Member - Ningxia University)
- [Deng Yuwen](https://github.com/GKDGKD) (Datawhale Member - Guangzhou University)
- [Du Sen](https://github.com/coderdeepstudy) (Datawhale Member - Nanyang Institute of Technology)
- [Xiao Hongru](https://github.com/Hongru0306) (Datawhale Member - Tongji University)