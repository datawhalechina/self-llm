<h1>InternLM3-8b-instruct与o1</h1>

Open AI o1（又名o1，也称草莓模型，这个名字的又来是因为早期人们发现gpt总是数不对strawberry中有几个r），是Open AI于2024年9月12日发布的一款通过大规模强化学习算法训练的大模型。区别于其他大模型产品，o1不需要用户再输入复杂的提示词，而是通过强化学习，将思维链内化后进行持续训练。

<img src="https://github.com/riannyway/self-llm/blob/patch-1/models/InternLM3/images/o1-1.png?raw=true">

<p>如图，通过思维链式的问题拆解，模型可以不断验证、纠错，尝试新的方法，这一过程显著提升了模型的推理能力。</p>
OpenAI o1的关键技术在于RL的搜索与学习机制。基于大型语言模型（LLM）已有的推理能力，通过迭代式的Bootstrap，模型能够产生合理的推理过程。这一过程不仅限于COT，还可能包括在常识问答（Common Sense QA）中对问题答案的潜在推理反思。通过将这种合理推理过程融入训练，模型学会了潜在的推理。随后，利用强大的计算资源实现Post-Training阶段的Scaling。这一技术路线与STaR的扩展版本非常相似。


<p>同时，OpenAI o1的出现验证了后训练扩展律（Post-Training Scaling Laws），为上述技术路线的成功提供了有力支持。后训练扩展律的出现让许多学者重新思考推理的定义，一个合理的回答是“将思考时间转化为能力”，即通过增加思考推理时间来提升模型能力。随着在Post-Training阶段RL搜索的增强和在推理阶段的搜索时间增强，模型的能力得到了提升。模型在这过程中学习的是合理推理的过程，TreeSearch在其中起到了诱导合理推理过程产生的作用，或基于合理推理过程构建相应的偏序对形成系统的奖励信号。在模型的训练过程中，TreeSearch方法有助于构建系统的奖励信号，这在后续的技术路径推演中会提到。在推理阶段，搜索过程可能基于多样的TreeSearch方法实现。更有趣的是，模型的Bootstrap有助于构建新的高质量数据，这些数据中的Rationales促进了模型的进一步提升。</p>

<img src="https://github.com/riannyway/self-llm/blob/patch-1/models/InternLM3/images/70fc262cd3a3cba523257d3a54afb73.png?raw=true">

此外，o1的性能随着更多的强化学习（训练时间计算）和更多的思考时间（测试时间计算）而持续提高。
在数学方面，在2024年的AIME（一个旨在挑战美国最聪明高中生的考试）测评中，GPT-4o只解决了13%的问题，o1的得分是83%。
在编码方面，GPT-4o在竞争性编程问题(Codeforces)上的得分是11%，o1 是89%。
在博士级别的科学问题(GPQA Diamond)，GPT4o是56.1%，o1则超越人类博士69.7%，达到了恐怖的78%。
虽然在写作、文字编辑等自然领域反而逊色于gpt-4o体现出o1仍存在适用性问题，但是强大的推理能力让许多研究者们趋之若鹜，短短数个月的时间里出现了不少类似o1推理链的构想。

## 环境配置依赖

环境依赖如下：
```
----------------------
 Transformer >=4.48 
 Torch == 2.3.0     
 Cuda ==  12.1  
----------------------
```

 >本文默认学习者已安装好以上 Pytorch(cuda) 环境，如未安装请自行安装。

## 准备工作
```shell
# 升级pip
python -m pip install --upgrade pip
# 更换 pypi 源加速库的安装
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope
pip install accelerate
```
> 考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了 InternLM3-8b-Instruct 的环境镜像，点击下方链接并直接创建 AutoDL 示例即可。
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/self-llm-internlm3***

## 模型下载

`modelscope` 是一个模型管理和下载工具，支持从魔搭 (Modelscope) 等平台快速下载模型。

这里使用 `modelscope` 中的 `snapshot_download` 函数下载模型，第一个参数为模型名称，第二个参数 `cache_dir` 为模型的下载路径，第三个参数 `revision` 为模型的版本号。

在 `/root/autodl-tmp` 路径下新建 `model_download.py` 文件并在其中粘贴以下代码，并保存文件。

```python
from modelscope import snapshot_download, AutoModel, AutoTokenizer

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm3-8b-instruct', cache_dir='./', revision='master')
```

> 注意：记得修改 cache_dir 为你的模型下载路径哦~
在终端运行 `python /root/autodl-tmp/model_download.py` 执行下载，模型大小为 18GB 左右，下载模型大概需要5-30分钟。
<img src="https://github.com/riannyway/self-llm/blob/patch-1/models/InternLM3/images/o1.png?raw=true">

## 核心代码
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

class InternLM3_8B_Instruct:
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path: str):
        print("正在从本地加载模型")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    def generate_response(self, prompt: str, max_new_tokens: int = 512):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to('cuda')

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens
            )

        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response

    def generate_response_stream(self, prompt: str, max_new_tokens: int = 512):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to('cuda')

        with torch.no_grad():
            for output in self.model.stream_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens
            ):
                # Decode and yield the output token by token
                output_str = self.tokenizer.decode(output, skip_special_tokens=True)
                yield output_str

if __name__ == '__main__':
    model_path = "/root/autodl-tmp/Shanghai_AI_Laboratory/internlm3-8b-instruct"
    llm = InternLM3_8B_Instruct(model_path)
    prompt = """
你是一位在数学竞赛领域具有丰富经验的数学专家。你通过系统性的思考和严谨的推理来解决问题。
在面对以下数学问题时，请遵循以下思考过程：
## 深入理解问题
在尝试解决问题之前，请花时间完全理解问题：
- 实际上被问的是哪个问题？
- 给定的条件是什么，它们告诉我们什么？
- 是否有任何特殊的限制或假设？
- 哪些信息是关键的，哪些是辅助的？
## 多角度分析
在解决问题之前，进行彻底的分析：
- 涉及哪些数学概念和性质？
- 你能回忆起类似的经典问题或解决方法吗？
- 图表或表格是否有助于可视化问题？
- 是否有需要单独考虑的特殊情况？
## 系统性思考
规划你的解决方案路径：
- 提出多种可能的解决方法
- 分析每种方法的可行性和优点
- 选择最合适的方法，并解释为什么
- 将复杂问题分解为更小、更易管理的步骤
## 严谨证明
在解决问题的过程中：
- 为每一步提供坚实的理由
- 为关键结论提供详细的证明
- 注意逻辑联系
- 警惕可能的疏忽
## 反复验证
完成解决方案后：
- 验证你的结果是否满足所有条件
- 检查是否有被忽视的特殊情况
- 考虑解决方案是否可以优化或简化
- 审视你的推理过程
记住：
1. 花时间深入思考，而不是急于求成
2. 严谨地证明每个关键结论
3. 保持开放的心态，尝试不同的方法
4. 总结有价值的解决问题的方法
5. 保持健康的怀疑态度，多次验证

你的回应应反映出深厚的数学理解力和精确的逻辑思考，使你的解决方案路径和推理对他人清晰易懂。

当准备好后，请以以下格式呈现你的完整解决方案：
- 清晰的问题理解
- 详细的解决方案过程
- 关键见解
- 彻底的验证

专注于清晰、逻辑性的思路进展，详细解释你的数学推理。以提问者使用的语言提供答案，并在最后使用 '\\boxed{}' 重复最终答案。

现在，请解决以下数学问题[5的十次方等于多少？]
"""
    # Generate response in one go
    response = llm.generate_response(prompt=prompt)
    print(response)

    # # 流式输出
    # print("Streaming response:")
    # for part in llm.generate_response_stream(prompt=prompt):
    #     print(part, end='', flush=True)

```
将代码保存到文件中，需要修改存放模型的路径，接着使用命令
```python
python 文件名.py
```
即可运行。

## 结果展示
![image](https://github.com/user-attachments/assets/ca4c7636-33c5-4560-9aaf-0eeb753b137c)

