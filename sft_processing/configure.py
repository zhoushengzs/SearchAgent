#!/usr/bin/env python3
"""Create local teacher config: python sft_processing/configure.py --data-root /absolute/data/path."""
import argparse,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True,help='Original dataset directory; never uploaded')
    ap.add_argument('--output',type=Path,help='Default: teacher/config.json; must not already exist')
    args=ap.parse_args(); here=Path(__file__).resolve().parent; data=args.data_root.resolve()
    for p in (data/'train.jsonl',data/'cleaned_v2_renumbered_20261026/train.jsonl',data/'.codex/sft_cleaning_20261025/row_manifest.jsonl'):
        if not p.is_file(): ap.error('Required local input not found: '+str(p))
    path=args.output or here/'teacher/config.json'
    if path.exists(): ap.error('Refusing to overwrite an existing configuration: '+str(path))
    if path.resolve()==(here/'teacher/config.example.json').resolve(): ap.error('Cannot overwrite template')
    cfg=json.loads((here/'teacher/config.example.json').read_text())
    cfg.update(prompts_dir=str(here/'teacher/prompts'),cleaned_dir=str(data/'cleaned_v2_renumbered_20261026'),media_dir=str(data),
               cleaner_script=str(here/'cleaning/clean_sft.py'),
               jobs_file=str(data/'.codex/teacher_from_github/jobs/pilot100.jsonl'),
               runs_dir=str(data/'teacher_stage2_runs_v2'),recovery_source=str(data/'train.jsonl'),
               recovery_manifest=str(data/'.codex/sft_cleaning_20261025/row_manifest.jsonl'))
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: f.write(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    print('Local configuration: '+str(path))
if __name__=='__main__': main()
