import json
import re
import pprint1

with open('./README.md', 'r') as f:
    readme = f.read()

with open('./contributors.json', 'r') as f:
    contributors = json.load(f)

# task清零
keys = contributors.keys()
for key in keys:
    contributors[key]['task_num'] = 0

tasks = readme.split('\n')
tasks = [task for task in tasks if '@' in task][:-1]

# 微调任务+2，普通任务+1
for task in tasks:
    name = task.split('@')[1]
    if name not in keys:
        continue
    if "Lora" in task:
        contributors[name]['task_num'] += 2
    else:
        contributors[name]['task_num'] += 1
# 排序
contributors = dict(sorted(contributors.items(), key=lambda x: x[1]['task_num'], reverse=True))

with open('./contributors.json', 'w') as f:
    json.dump(contributors, f, indent=4, ensure_ascii=False)

for key, value in contributors.items():
    print(f'- {value["info"]}')