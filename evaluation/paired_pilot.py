"""Prepare matching pilot bundles and run SFT, then Base, under one agent protocol."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys

from evaluation.common import load_tasks, write_json_atomic
from evaluation.pilot import ROOT, source_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sft-model',type=Path,required=True)
    p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    commands=[]
    for name,model in [('sft',a.sft_model),('base',a.base_model)]:
        commands.append([sys.executable,'-u','-m','evaluation.pilot','--model',str(model.resolve()),
            '--output',str(a.output/name),'--routing','training','--device','mps',
            '--search-key-env','SERPER_API_KEY_4','--max-steps','10','--max-new-tokens','1024',
            '--repetition-guard','--max-generation-seconds','120'])
    for command in commands:
        subprocess.run(command+['--prepare-only'],cwd=ROOT,check=True)
    plans=[json.loads((a.output/name/'pilot.json').read_text()) for name in ['sft','base']]
    for key in ['tasks_sha256','references_sha256','routing','generation_budget','shared_execution_policy','code_sha256']:
        if plans[0][key]!=plans[1][key]:raise ValueError('Paired configuration differs: '+key)
    if [{r['benchmark']:r['ids'] for r in plan['routes']} for plan in plans][0] != {r['benchmark']:r['ids'] for r in plans[1]['routes']}:
        raise ValueError('Paired selected IDs differ')
    for left,right in zip(plans[0]['routes'],plans[1]['routes']):
        if load_tasks(left['tasks']) != load_tasks(right['tasks']):
            raise ValueError('Paired task contents differ')
    manifest={'schema':'paired-training-pilot-v1','commands':commands,'code_sha256':source_identity(),
              'selected_ids':{r['benchmark']:r['ids'] for r in plans[0]['routes']},
              'count_per_model':sum(r['count'] for r in plans[0]['routes']),
              'generation_budget':plans[0]['generation_budget'],'model_order':['sft','base'],
              'scoring':'separate stage; benchmark-specific scoring rules retained'}
    manifest_path=a.output/'pair.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Pair configuration changed; choose a new output directory')
    write_json_atomic(manifest_path,manifest)
    if a.prepare_only:return
    state={'status':'running','pid':os.getpid(),'models':{'sft':'running','base':'queued'}}
    write_json_atomic(a.output/'status.json',state)
    for name,command in zip(['sft','base'],commands):
        if source_identity()!=manifest['code_sha256']:
            state.update(status='needs_review',error='Evaluation code changed between paired runs')
            write_json_atomic(a.output/'status.json',state)
            raise SystemExit(state['error'])
        state['models'][name]='running';write_json_atomic(a.output/'status.json',state)
        print('Starting paired model:',name,flush=True)
        with (a.output/(name+'.log')).open('a') as stream:
            result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        file=a.output/name/'completion.json'
        completion=json.loads(file.read_text()) if file.exists() else {}
        state['models'][name]='complete' if result.returncode==0 and completion.get('complete') else 'needs_review'
        write_json_atomic(a.output/'status.json',state)
    state['status']='complete' if all(v=='complete' for v in state['models'].values()) else 'needs_review'
    write_json_atomic(a.output/'status.json',state)


if __name__=='__main__':
    main()
