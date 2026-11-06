from pathlib import Path
import json,hashlib,time
P=Path('/cluster/home/shenqunlun/zhousheng');src=P/'datasets/sft/downloads_selected_20261103';out=P/'datasets/sft/smoke_selected_20261103';out.mkdir(parents=True,exist_ok=True)
manifest={r['path']:r for r in json.loads((P/'training4b/sft_manifest.json').read_text())}
def verify(name):
 p=src/name;r=manifest[name];assert p.is_file() and p.stat().st_size==r['bytes'],name
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert h.hexdigest()==r['sha256'],name
verify('train.jsonl');verify('dataset_info.json')
lines=[]
with (src/'train.jsonl').open() as f:
 for line in f:
  if line.strip():lines.append(line)
  if len(lines)==16:break
assert len(lines)==16
images={name for line in lines for name in json.loads(line).get('images',[])}
for name in images:verify(name)
(out/'train.jsonl').write_text(''.join(lines));(out/'dataset_info.json').write_text((src/'dataset_info.json').read_text())
c=json.loads((P/'training4b/sft_smoke.json').read_text());c.update(dataset_dir=str(out),media_dir=str(src),output_dir=str(P/'results4b/sft_early_smoke'))
(P/'training4b/sft_early_smoke.json').write_text(json.dumps(c,indent=2))
(out/'VERIFIED.json').write_text(json.dumps({'rows':16,'verified_images':len(images),'source':str(src),'at':time.time()}))
print('EARLY_SMOKE_DATA_VERIFIED',len(images),flush=True)
