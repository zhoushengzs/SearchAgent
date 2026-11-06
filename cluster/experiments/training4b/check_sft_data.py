import hashlib,json,time
from pathlib import Path
P=Path('/cluster/home/shenqunlun/zhousheng');d=P/'datasets/sft/downloads_selected_20261103';rows=json.loads((P/'training4b/sft_manifest.json').read_text())
for attempt in range(1440):
 bad=[r for r in rows if not (d/r['path']).is_file() or (d/r['path']).stat().st_size!=r['bytes']]
 if not bad:break
 if attempt%10==0:print('waiting_files',len(bad),flush=True)
 time.sleep(30)
else:raise RuntimeError('Data upload not ready after 12 hours')
for r in rows:
 h=hashlib.sha256()
 with (d/r['path']).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==r['sha256'],r['path']
assert json.loads((P/'Qwen3-VL-4B-Instruct/DOWNLOAD_COMPLETE.json').read_text())['repo']=='Qwen/Qwen3-VL-4B-Instruct'
assert (P/'training4b/SFT_ENV_READY.json').is_file()
(d/'UPLOAD_VERIFIED.json').write_text(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'verified_at':time.time()}))
print('DATA_AND_ENV_READY',flush=True)
