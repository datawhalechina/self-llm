import json
from pathlib import Path
import torch
from safetensors.torch import load_file

for name in ('qwen3-lora', 'qwen2-lora'):
    directory=Path('/results')/name
    assert (directory/'adapter_config.json').is_file()
    assert (directory/'tokenizer_config.json').is_file()
    tensors=load_file(str(directory/'adapter_model.safetensors'))
    assert tensors and all(torch.isfinite(t).all().item() for t in tensors.values())
    updated=[key for key,value in tensors.items() if 'lora_B' in key and torch.count_nonzero(value).item()>0]
    assert updated, 'No trained LoRA B weights found'
    print(json.dumps({'adapter':name,'status':'PASS','saved_tensors':len(tensors),'nonzero_lora_B_tensors':len(updated),'files':sorted(p.name for p in directory.iterdir())}))
print('ARTIFACT_CHECK_EXIT=0')
