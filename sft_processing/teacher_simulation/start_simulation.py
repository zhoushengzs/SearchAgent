#!/usr/bin/env python3
"""Start an answer-conditioned replay demonstration pilot; no key is saved."""
import argparse
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=root/'.codex/teacher_simulation_20261026/config.json')
    p.add_argument('--limit',type=int,default=3); p.add_argument('--max-api-calls',type=int,default=8)
    p.add_argument('--model'); p.add_argument('--reviewer-model')
    a=p.parse_args(); cfg=json.loads(a.config.read_text()); env=os.environ.copy()
    model=a.model or env.get('OPENROUTER_MODEL') or cfg.get('teacher_model')
    if not model: model=input('请输入支持图片的 OpenRouter 教师模型 ID：').strip()
    if not model: raise SystemExit('模型 ID 为空，未调用模型。')
    env['OPENROUTER_MODEL']=model
    env['OPENROUTER_REVIEWER_MODEL']=a.reviewer_model or env.get('OPENROUTER_REVIEWER_MODEL') or cfg.get('reviewer_model') or model
    if not env.get('OPENROUTER_API_KEY'):
        env['OPENROUTER_API_KEY']=getpass.getpass('请输入 OpenRouter 密钥（不显示、不保存）：').strip()
    if not env['OPENROUTER_API_KEY']: raise SystemExit('密钥为空，未调用模型。')
    command=[sys.executable,str(Path(__file__).with_name('simulate_backtracking.py')),'run','--config',str(a.config),
             '--limit',str(a.limit),'--max-api-calls',str(a.max_api_calls)]
    print('教师将看到完整过程和现有答案，编排回探示范；仅回放已有记录，不实际调用调查工具。')
    raise SystemExit(subprocess.run(command,env=env).returncode)
if __name__=='__main__': main()
