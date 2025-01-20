<h1>InternLM3-8b-instruct与o1</h1>

OpenAI o1于2024年9月13日正式发布，作为OpenAI最新发布的最强推理模型，标志着AI行业进入了一个新时代。
o1在测试化学、物理和生物学专业知识的基准GPQA-diamond上，全面超过了人类博士专家，OpenAI宣称“通用人工智能(AGI)之路，已经没有任何阻碍”。
不同于传统的语言模型，o1在回答之前会生成一个内部的思维链。这个思路链是一个逐步推导、逐步分解问题的过程，它模拟了人类思考的方式，
使得模型能够更深入地理解问题并给出更准确的答案。虽然深度思考会略微影响模型的回答速度，但准确率却有着显著提高，这使得许多研究者争先恐后的对o1进行“解剖”。

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

首先 `pip` 换源加速下载并安装依赖包：

```shell
# 升级pip
python -m pip install --upgrade pip
pip install modelscope
pip install accelerate
```
> 考虑到部分同学配置环境可能会遇到一些问题，我们在 AutoDL 平台准备了 InternLM3-8b-Instruct 的环境镜像，点击下方链接并直接创建 AutoDL 示例即可。
> ***https://www.codewithgpu.com/i/datawhalechina/self-llm/InternLM3-self-llm***

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
```
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LlamaTokenizerFast
import torch

class InternLM3_8B_Instruct:
    # 自定义类来处理与 internlm3-8b-instruct 模型的交互
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name_or_path: str):
        print("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        print("完成本地模型的加载")

    def generate_response(self, prompt: str, max_new_tokens: int = 512):
        # 准备输入
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to('cuda')

        # 生成响应
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens
            )

        # 解码生成的内容
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response

if __name__ == '__main__':
    # 指定模型路径
    model_path = "/root/autodl-tmp/Shanghai_AI_Laboratory/internlm3-8b-instruct"
    llm = InternLM3_8B_Instruct(model_path)
    prompt="""你是一位在数学竞赛领域具有丰富经验的数学专家。你通过系统性的思考和严谨的推理来解决问题。在面对以下数学问题时，请遵循以下思考过程：
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

现在，请解决以下数学问题[5的十次方等于多少？]"""
    response = llm.generate_response(prompt = prompt)
    print(response)

```
要注意的是，学习者们需要自行在grouq(可访问grouq cloud，可能需要科学上网)并在powershell中配置临时变量api_key，具体代码如下：
```
export GROQ_API_KEY=gsk...
```
将gsk...替换为自己的api_key即可。

## 结果展示
<img src="https://github.com/riannyway/self-llm/blob/patch-1/models/InternLM3/images/o1.png?raw=true">
