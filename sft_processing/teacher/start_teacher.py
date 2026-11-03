#!/usr/bin/env python3
"""Interactive three-job pilot. Secrets stay in process environment, not files."""
import argparse
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--limit',type=int,default=3,help='Number of new jobs, default 3')
    ap.add_argument('--max-api-calls',type=int,default=8,help='Total POST limit including retries, default 8')
    ap.add_argument('--model'); ap.add_argument('--reviewer-model')
    args=ap.parse_args(); here=Path(__file__).resolve().parent
    cfg=json.loads((here/'config.json').read_text()); env=os.environ.copy()
    model=args.model or env.get('OPENROUTER_MODEL') or cfg.get('teacher_model')
    if not model:
        model=input('请输入支持图片输入的 OpenRouter 模型 ID（可从模型页面复制）：').strip()
    if not model: raise SystemExit('模型 ID 为空，未发出请求。')
    env['OPENROUTER_MODEL']=model
    reviewer=args.reviewer_model or env.get('OPENROUTER_REVIEWER_MODEL') or cfg.get('reviewer_model') or model
    env['OPENROUTER_REVIEWER_MODEL']=reviewer
    if not env.get('OPENROUTER_API_KEY'):
        key=getpass.getpass('请输入 OpenRouter API Key（输入不会显示，也不会保存到文件）：').strip()
        if not key: raise SystemExit('密钥为空，未发出请求。')
        env['OPENROUTER_API_KEY']=key
    command=[sys.executable,str(here/'teacher_stage2.py'),'run','--config',str(here/'config.json'),
             '--limit',str(args.limit),'--max-api-calls',str(args.max_api_calls),'--export']
    print(f'开始处理最多 {args.limit} 条新样本。教师：{model}；审核：{reviewer}。')
    result=subprocess.run(command,env=env)
    raise SystemExit(result.returncode)

if __name__=='__main__': main()
