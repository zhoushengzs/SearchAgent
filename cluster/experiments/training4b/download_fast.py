import os,sys,json,time,hashlib,concurrent.futures,threading
from pathlib import Path
PERSONAL=Path('/cluster/home/shenqunlun/zhousheng')
os.environ['HF_HOME']=str(PERSONAL/'cache/huggingface')
os.environ['TMPDIR']=str(PERSONAL/'tmp/model-download')
Path(os.environ['TMPDIR']).mkdir(parents=True,exist_ok=True)
import requests
from huggingface_hub import HfApi,hf_hub_download
repo,revision,destination=sys.argv[1:];target=Path(destination);target.mkdir(parents=True,exist_ok=True)
info=HfApi(endpoint='https://hf-mirror.com').model_info(repo,revision=revision,files_metadata=True,timeout=60)
assert info.sha==revision
state=target/'.fast-download';state.mkdir(exist_ok=True)
lock=threading.Lock(); chunk=16*1024*1024

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def download_file(s):
 name=s.rfilename;dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True)
 if not s.lfs:
  if dst.is_file():
   b=dst.read_bytes()
   if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==s.blob_id:return
  hf_hub_download(repo,filename=name,revision=revision,local_dir=target,endpoint='https://hf-mirror.com');return
 size=s.size;expected=s.lfs.sha256
 if dst.is_file() and dst.stat().st_size==size and digest(dst)==expected:return
 key=name.replace('/','_');partial=state/(key+'.partial');journal=state/(key+'.json')
 if journal.exists():j=json.loads(journal.read_text())
 else:
  candidates=list((target/'.cache/huggingface/download').glob('*.'+expected+'.incomplete'))
  candidate=max(candidates,key=lambda p:p.stat().st_size,default=None)
  prefix=0
  if candidate is not None and candidate.stat().st_size<=size:
   prefix=candidate.stat().st_size;candidate.rename(partial)
  j={'prefix':prefix,'done':[],'size':size,'sha256':expected}
  journal.write_text(json.dumps(j))
 assert j['size']==size and j['sha256']==expected
 fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o600);os.ftruncate(fd,size)
 jobs=[(start,min(start+chunk,size)-1) for start in range(j['prefix'],size,chunk) if start not in j['done']]
 def part(item):
  start,end=item
  for attempt in range(8):
   try:
    url=(f'https://modelscope.cn/models/{repo}/resolve/master/{name}?range_start={start}' if attempt % 2 == 0 else f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}?download=true&range_start={start}')
    with requests.get(url,headers={'Range':f'bytes={start}-{end}','Cache-Control':'no-cache','Pragma':'no-cache'},stream=True,timeout=(30,120)) as r:
     assert r.status_code==206,f'HTTP {r.status_code}'
     assert r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'), 'Unexpected range'
     pos=start
     for data in r.iter_content(1024*1024):
      assert pos+len(data)<=end+1
      n=os.pwrite(fd,data,pos);assert n==len(data);pos+=n
     assert pos==end+1,'Short read'
    with lock:
     os.fsync(fd);j['done'].append(start);temp=journal.with_suffix('.tmp');temp.write_text(json.dumps(j));temp.replace(journal)
     print('PART',name,start,end,flush=True)
    return
   except Exception as e:
    print('RETRY',name,start,type(e).__name__,str(e) if isinstance(e,AssertionError) else '',flush=True);time.sleep(min(30,2**attempt))
  raise RuntimeError(f'Unable to download {name} range {start}')
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(part,jobs))
 finally:os.close(fd)
 assert digest(partial)==expected,f'Checksum mismatch: {name}'
 partial.replace(dst);journal.unlink();print('VERIFIED',name,flush=True)
files=[s for s in info.siblings if not s.rfilename.endswith(('.bin','.pt','.pth'))]
# Complete smaller metadata first; parallel ranges speed up each large weight file.
for s in sorted(files,key=lambda s:s.size or 0):download_file(s)
(target/'DOWNLOAD_COMPLETE.json').write_text(json.dumps({'repo':repo,'revision':revision,'completed_at':time.time(),'method':'8 parallel ranges, sha256 checked'}))
print('DOWNLOAD_COMPLETE',repo,flush=True)
