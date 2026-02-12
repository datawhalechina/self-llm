# 03 step3.5Flash Lora 微调

本教程旨在帮助学习者基于 Step-3.5 Flash 模型进行 LoRA（Low-Rank Adaptation）微调。Step-3.5 Flash 是阶跃星辰（StepFun）推出的高性能大模型，通过微调可以使其在特定领域（如个性化角色扮演）表现出色。

---

## 大模型微调的意义与 LoRA 优势

**微调的意义：**
预训练模型（Base Model）具备通用的知识处理能力，但往往缺乏特定领域的专业知识、特定的对话风格或遵循特定指令的能力。微调（Fine-tuning）通过在垂直领域数据上进一步训练，使模型能够适配特定场景。

**全量微调 vs. LoRA：**

* **全量微调 (Full Fine-tuning)：** 修改模型的所有参数。
    *   *缺点：* 显存需求极高，计算成本昂贵，容易产生“灾难性遗忘”（即微调后丢失了原有的通用能力）。
*   **LoRA (Low-Rank Adaptation)：** 冻结模型所有原参数，只在原参数矩阵旁引入两个低秩矩阵（A 和 B）进行微调。
    *   *LoRA 的优势：*
        1.  **极低显存：** 训练参数量通常不到原模型的 1%，显存压力骤减。
        2.  **训练高效：** 计算量小，训练速度快。
        3.  **可插拔性：** 训练生成的 Adapter 文件极小（几十MB），可以方便地在不同任务间切换。
        4.  **性能接近：** 在大多数下游任务中，LoRA 的效果与全量微调相当。

---

## 环境配置

*   **硬件：** RTX PRO 6000 (96G 显存) × 8 卡
*   **操作系统：** Ubuntu 22.04
*   **软件栈：** 
    *   Python 3.12
    *   CUDA 12.8
    *   PyTorch 2.8
    *   Transformers 4.57.3

---

## 虚拟环境搭建

```bash
# 创建环境
conda create -n step_tuning python=3.12 pip -y
conda activate step_tuning

# 升级 pip
pip install --upgrade pip

# 安装 PyTorch (根据官网 CUDA 12.8 匹配指令)
pip install torch --index-url https://download.pytorch.org/whl/cu124 # 示例，需匹配2.8

# 安装核心库
pip install pandas datasets transformers==4.57.3 swanlab peft modelscope
```

---

本教程也提供运行的环境镜像，点击[链接](https://codewithgpu.com/i/datawhalechina/self-llm/Step-3.5-Flash_Lora)进行创建AutoDL示例即可。

## 模型下载

使用 `modelscope` 的 `snapshot_download` 快速下载 Step-3.5 Flash 模型权重：

```python
from modelscope import snapshot_download
model_dir = snapshot_download('stepfun-ai/Step-3.5-Flash', cache_dir='./model_weights')
```

---

## 数据集处理 
在本节我们使用由笔者合作开源的[Chat-甄嬛](https://github.com/KMnO4-zx/huanhuan-chat)项目作为示例，我们的目标是构建一个能够模拟甄嬛对话风格的个性化 LLM我们使用 **Dolly 格式** 构建数据集。Dolly 格式通常包含 `instruction` (指令)、`context` (上下文) 和 `response` (回答)。数据集在[这里](https://github.com/datawhalechina/self-llm/tree/master/dataset/dolly_huanhuan.jsonl)

**核心逻辑：**
1.  **DollyProcessor：** 将文本拼接成 `### Instruction:...### Input:...### Response:...` 的格式。

```python
class DollyProcessor:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_fn(self, example):
        # 1. 构建 Prompt
        prompt_text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example.get('context', '')}\n\n### Response:\n"
        full_text = prompt_text + f"{example['response']}{self.tokenizer.eos_token}"
        
        # 2. 对全文进行编码 
        # 注意：使用 add_special_tokens=True 以确保包含 BOS token（如果模型需要）
        encodings = self.tokenizer(
            full_text, 
            truncation=True, 
            max_length=self.max_len, 
            padding=False, 
            add_special_tokens=True 
        )
        
        input_ids = list(encodings["input_ids"])
        attention_mask = list(encodings["attention_mask"])
        labels = copy.deepcopy(input_ids)
        
        # 3. 计算 Prompt 长度（关键：保持与全文编码一致的 special tokens 设置）
        # 这样计算出的长度才能准确匹配到 Response 的起始位置
        prompt_encodings = self.tokenizer(
            prompt_text, 
            add_special_tokens=True, 
            truncation=True, 
            max_length=self.max_len
        )
        prefix_len = len(prompt_encodings["input_ids"])
        
        # 4. 遮掩 Label 中的 Prompt 部分
        for i in range(min(prefix_len, len(labels))):
            labels[i] = -100 
            
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask, 
            "labels": labels, 
            "ref_text": example["response"]
        }
```
2.  **Label Masking：** 在计算损失时，为了让模型只学习“回答”部分，我们将 `Instruction` 和 `Input` 部分的 Label 设为 `-100`，使模型在反向传播时不计算这些 Token 的损失。

```python
# 关键代码片段：Label 遮掩
prompt_encodings = self.tokenizer(prompt_text, add_special_tokens=True, ...)
prefix_len = len(prompt_encodings["input_ids"])
for i in range(min(prefix_len, len(labels))):
    labels[i] = -100 # 遮掩 Prompt 部分
```

---

## 模型配置与 LoRA 注入

在 Step-3.5 Flash 中，由于其特殊的架构（包含 MoE 专家系统），我们需要精准选择注入模块。

*   **加载模型：** 使用 `bf16` 提高性能并降低显存。
*   **LoRA 参数：**
    *   `r=16`：秩大小。
    *   `lora_alpha=32`：缩放因子。
    *   **target_modules：** 仅针对 `q_proj` 和 `k_proj`。避开了 MoE 相关的 `gate_proj/up_proj/down_proj` 以防止兼容性报错。

```python
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj"],
    bias="none",
)
peft_model = get_peft_model(base, lora_cfg)
```

---

## 训练配置与启动

*   **SwanLab：** 用于实时监控训练曲线（Loss、学习率等）。
*   **显存调优：** 开启 `gradient_checkpointing` 以节省显存，关闭 `use_cache` 以兼容梯度检查点。
*   **参数设置：** 10 个 Epoch 保证充分拟合，2e-3 的学习率确保在 LoRA 下能快速收敛。

```python
training_args = TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=32,
    learning_rate=2e-3,
    bf16=True,
    save_total_limit=1,
    # ...
)
# 必须开启梯度检查点和输入梯度要求
peft_model.enable_input_require_grads()
peft_model.gradient_checkpointing_enable()
```
这是训练时的loss变化，我们能看到loss从2.8左右降到了接近0(这是运行了几次的结果，不是单独一次的，但总体趋势相同)

![](images\03-01.png)
![](images\03-08.png)
![](images\03-02.png)
---

## 测试与推理

推理时必须保证 **Prompt 模板与训练时完全一致**。此外，Step-3.5 Flash 默认的 `eos_token_id` 可能是 128001，但在训练编码时通常识别为 128007，因此生成时需显式指定。

```python
# 拼接与训练一致的模板
full_prompt = f"### Instruction:\n{prompt}\n\n### Input:\n\n\n### Response:\n"

gen_kwargs = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.4,
    "eos_token_id": [128007, tokenizer.eos_token_id], # 核心补丁
}
```
当prompt是'你是谁'的时候，它回答：
![](images\03-03.png)

当prompt是'你和温实初是什么关系'时，它回答：
![](images\03-04.png)

---

## 常见报错与避坑指南 (重要)

在微调 Step-3.5 Flash 时，由于其 `modeling` 文件与标准 Transformer 略有差异，需手动修复以下问题：

  **a. pad_token_id 缺失：** 
  
  在 `config.json` 中添加 `pad_token_id`，其值设为与 `eos_token_id` 的第一个元素相同。
  
  **b. KeyError['default']：** 
  
  `transformers` 版本过高导致。使用 `4.57.3` 版本可行（5及以上版本测试不可行）。

  **c. MoE 冲突：** 
  
  `target_modules` 暂不支持 `gate/up/down_proj`，微调时请避开这些层（因为这些层中含有阶跃星辰自己写的MoE，不太支持这种微调）。如图所示，model.layers.3.moe.up_proj等层不支持这种lora微调。
  ![](images\03-05.png)

  **d. 属性缺失：** 
  
  `config.json` 中需补全 `use_cache: false` 和 `attention_dropout: 0`；并在 `modeling_step3p5.py` 的 `Step3p5Attention` 类中手动定义 `self.attention_dropout=config.attention_dropout`。

  **e. 缺失 Loss 计算：** 
  
  原版模型代码可能未在 `forward` 中计算 Loss。需在 `modeling_step3p5.py` 的 `Step3p5ForCausalLM` 类(forward函数)中手动加入 `CrossEntropyLoss` 计算逻辑并执行 `Shift`（错位对齐），并去掉原先的返回值。在`logits = self.lm_head(hidden_states)`之后，加上如下的代码：

  ```python
loss = None
        if labels is not None:
            # 根据你看到的 CausalLM 标准，需要进行 Shift 偏移
            # Logits: [Batch, SeqLen, Vocab] -> 去掉最后一个 token 的预测
            shift_logits = logits[..., :-1, :].contiguous()
            # Labels: [Batch, SeqLen] -> 去掉第一个 token（因为它是被预测的）
            shift_labels = labels[..., 1:].contiguous()
            
            # 使用交叉熵计算损失
            # 根据文档，ignore_index 设为 -100，会自动忽略掉你 labels 里的那些 -100
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            
            # 展平数据以符合 CrossEntropyLoss 的输入要求
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # 确保标签在正确的设备上
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # 3. 按照 return_dict 的要求返回结果
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Step3p5CausalLMOutputWithPast(
            loss=loss, # 这里的 loss 不再是 None，而是上面计算出的结果
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
  ```
  **f. 梯度获取报错：** 
  
  修改 `modeling_step3p5.py` 中的`Step3p5Model`类的 `get_input_embeddings` 函数，简化为直接返回 `self.embed_tokens`。
  ```py
  def get_input_embeddings(self):
        return self.embed_tokens
  ```

---

## Tips 总结

1. **训练端：** 
    *   Epoch 建议 10 左右，Loss 降到 0.1 以下效果更佳。
    *   学习率（LR）控制在 2e-3 左右，避免过小导致角色语态学习不明显。
2.  **推理端：** 
    *   **细节很重要：** Prompt 中的换行符 `\n` 和空格必须与训练代码完全一致。
    *   **终止符：** 由于Step3.5Flash的默认终止符是128001，而我们编码的终止符是128007，因此需要 `generate` 中手动加入 `128007` 作为 `eos_token_id`，否则模型可能会出现“停不下来”或胡言乱语的情况。

