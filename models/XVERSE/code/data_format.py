from datasets import load_dataset
import json

ds = load_dataset(path='LooksJuicy/ruozhiba')
datas = []
for i in ds['train']:
    instruction = i['instruction']
    output = i['output']
    data = [
        {
            'instruction': '{}'.format(instruction),
            'input': '',
            'output': '{}'.format(output)
        }
    ]
    datas.append(data[0])



# 将data列表中的数据写入到一个名为'ruozhiba.json'的文件中
with open('ruozhiba.json', 'w', encoding='utf-8') as f:
    # 使用json.dump方法将数据以JSON格式写入文件
    # ensure_ascii=False 确保中文字符正常显示
    # indent=4 使得文件内容格式化，便于阅读
    json.dump(datas, f, ensure_ascii=False, indent=4)