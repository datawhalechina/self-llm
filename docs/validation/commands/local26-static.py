import json, subprocess
from pathlib import Path
commands=[['python3','-m','compileall','-q','scripts/musa'],['bash','-n','scripts/musa/container.sh','scripts/musa/install.sh'],['git','--no-pager','diff','--check']]
records=[]
for command in commands:
 r=subprocess.run(command,capture_output=True,text=True)
 records.append({'command':command,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
Path('docs/validation/local26-20261012-static.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
print('STATIC_CHECK_EXIT='+str(int(any(r['exit_code']!=0 for r in records))))
