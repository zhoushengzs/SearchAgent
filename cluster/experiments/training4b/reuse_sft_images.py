import json,hashlib,os,time
from pathlib import Path
P=Path('/cluster/home/shenqunlun/zhousheng');dst=P/'datasets/sft/downloads_selected_20261103';old=P/'datasets/sft/OpenSearch-VL-SFT-Agent-NoImages'
manifest=json.loads((P/'training4b/sft_manifest.json').read_text());count=0;total=0
for r in manifest:
 if not r['path'].startswith('data/'):continue
 a=old/r['path'];b=dst/r['path']
 if b.exists() and b.stat().st_size==r['bytes']:continue
 if not a.is_file() or a.stat().st_size!=r['bytes']:continue
 h=hashlib.sha256()
 with a.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 if h.hexdigest()!=r['sha256']:continue
 b.parent.mkdir(parents=True,exist_ok=True);tmp=b.with_name(b.name+'.verified-reuse')
 if tmp.exists():tmp.unlink()
 os.link(a,tmp);tmp.replace(b);count+=1;total+=r['bytes']
(dst/'REUSED_IMAGES.json').write_text(json.dumps({'verified_reused':count,'bytes':total,'at':time.time()}));print(count,total,flush=True)
