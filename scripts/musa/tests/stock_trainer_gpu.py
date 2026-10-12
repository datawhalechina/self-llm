"""Real MUSA regression for CPU sampling and the unmodified HF Trainer."""
import json
import math
import torch
from torch.utils.data import RandomSampler, DistributedSampler
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

torch.manual_seed(42)
assert torch.musa.is_available()
dataset = list(range(12))
original_iter = RandomSampler.__iter__.__wrapped__
try:
    list(original_iter(RandomSampler(dataset)))
except RuntimeError as exc:
    assert 'generator' in str(exc).lower(), str(exc)
    print('ORIGINAL_RANDOM_SAMPLER_RESULT=' + str(exc))
else:
    print('ORIGINAL_RANDOM_SAMPLER_RESULT=PASS')
print(json.dumps({'default_generator_device':str(torch.Generator().device),
                  'explicit_cpu_generator_device':str(torch.Generator(device='cpu').device),
                  'default_device':str(torch.get_default_device())}))
for sampler in (RandomSampler(dataset), DistributedSampler(dataset, num_replicas=1, rank=0, seed=42)):
    indices = list(sampler)
    assert sorted(indices) == dataset
    print(json.dumps({'sampler':type(sampler).__name__, 'status':'PASS', 'indices':indices}))

model = AutoModelForCausalLM.from_pretrained(
    '/models/Qwen3-0.6B', local_files_only=True,
    torch_dtype=torch.bfloat16, attn_implementation='eager',
).to('cuda:0')
model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
                                       target_modules=['q_proj','v_proj'],
                                       r=8, lora_alpha=16, lora_dropout=0.0))
model.config.use_cache = False
before = {name: p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
assert before and all(p.device.type == 'musa' for p in model.parameters())
examples=[]
for offset in (0,1):
    ids=torch.tensor([1,2,3,4,5,6,7,8],dtype=torch.long)+offset
    examples.append({'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':ids.clone()})
args=TrainingArguments(output_dir='/tmp/local26-stock-trainer',per_device_train_batch_size=1,
                       max_steps=2,learning_rate=1e-4,logging_steps=1,save_strategy='no',
                       bf16=True,fp16=False,report_to='none',dataloader_num_workers=0,
                       seed=42,data_seed=42)
trainer=Trainer(model=model,args=args,train_dataset=examples)
assert type(trainer) is Trainer
result=trainer.train()
assert result.global_step == 2 and math.isfinite(result.training_loss)
changed=[name for name,p in model.named_parameters() if name in before and not torch.equal(before[name],p.detach())]
assert changed, 'No adapter weights changed'
assert all(p.device.type == 'musa' for p in model.parameters())
print(json.dumps({'test':'unmodified_hf_trainer','status':'PASS','model':'Qwen3-0.6B',
                  'physical_device':str(next(model.parameters()).device),'dtype':'bf16',
                  'steps':result.global_step,'loss':result.training_loss,
                  'updated_adapter_tensors':len(changed),'synthetic_token_data':True}))
