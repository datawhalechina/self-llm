import json,subprocess
from pathlib import Path
commands=[['python3','-m','unittest','discover','-s','scripts/musa/tests','-v'],
          ['python3','-m','compileall','-q','scripts/musa'],
          ['git','--no-pager','diff','--check']]
records=[]
for command in commands:
    p=subprocess.run(command,text=True,capture_output=True)
    records.append({'command':command,'location':'macOS dongyang tmux','exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
Path('docs/validation/local26-20261013-static.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print('SAMPLING_STATIC_CHECK_EXIT='+str(int(any(r['exit_code'] for r in records))))
