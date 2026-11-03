#!/usr/bin/env python3
"""Include pending rows on explicit user request, without claiming content repair.

Preserves train/validation split. Leaves source files untouched. Normalizes pending
row numbering with the existing validated renumbering implementation; keeps a
row-level provenance manifest outside the training file.
"""
import argparse
import collections as C
import hashlib
import importlib.util
import json
import shutil
import sqlite3
import tempfile
import re
from pathlib import Path


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--media-dir',type=Path,required=True)
    args=parser.parse_args()
    source=args.input.resolve(); output=args.output.resolve(); audit=args.audit.resolve(); media=args.media_dir.resolve()
    if output.exists() or audit.exists(): parser.error('Refusing to overwrite an existing output or audit directory')
    module_path=Path(__file__).resolve().parents[1]/'renumbering/renumber_sft.py'
    spec=importlib.util.spec_from_file_location('numbering',module_path)
    numbering=importlib.util.module_from_spec(spec); spec.loader.exec_module(numbering)
    inputs=[source/d/(split+'.jsonl') if d else source/(split+'.jsonl') for split in ('train','val') for d in ('','review_pending','review_renumbering')]
    inputs=[p for p in inputs if p.exists()]
    before={str(p):{'sha256':sha_file(p),'bytes':p.stat().st_size} for p in inputs}
    # Verify every pending record before creating a training output.
    pending_counts=C.Counter(); normalization_counts=C.Counter(); pending_tasks=set()
    for split in ('train','val'):
        for directory in ('review_pending','review_renumbering'):
            p=source/directory/(split+'.jsonl')
            if not p.exists(): continue
            with p.open() as f:
                for ln,line in enumerate(f,1):
                    row=json.loads(line); pending_tasks.add(row.get('trajectory_id') or row['id'].split(':')[0]); normalized,change=numbering.transform(row)
                    numbering.validate(normalized)
                    pending_counts[split]+=1; normalization_counts[split]+=bool(change['changed'])
    # A task cannot mix compressed and uncompressed histories under one step
    # numbering scheme. For tasks with pending rows, restore the original history
    # for their already-admitted positives too. Excluded target rows stay excluded.
    def original_id(identifier):
        match=re.match(r'^(.*):teacher-step:\d+:source-step:(\d+)$',identifier)
        if match: return match.group(1)+':teacher-step:'+match.group(2)
        return identifier
    wanted=set()
    for split in ('train','val'):
        with (source/(split+'.jsonl')).open() as f:
            for line in f:
                r=json.loads(line); tid=r.get('trajectory_id') or r['id'].split(':')[0]
                if tid in pending_tasks: wanted.add(original_id(r['id']))
    temporary=tempfile.TemporaryDirectory(prefix='sft_include_pending_')
    database=sqlite3.connect(str(Path(temporary.name)/'originals.sqlite'))
    database.execute('PRAGMA journal_mode=OFF'); database.execute('PRAGMA synchronous=OFF')
    database.execute('CREATE TABLE originals (id TEXT PRIMARY KEY, payload BLOB)')
    found=set()
    for split in ('train','val'):
        p=media/(split+'.jsonl')
        if not p.is_file(): raise ValueError('Original rows needed to rebuild shared histories: '+str(p))
        before[str(p)]={'sha256':sha_file(p),'bytes':p.stat().st_size}
        inputs.append(p)
        with p.open('rb') as f:
            for raw in f:
                r=json.loads(raw)
                if r['id'] in wanted:
                    database.execute('INSERT INTO originals VALUES (?,?)',(r['id'],raw)); found.add(r['id'])
        database.commit()
    if found!=wanted: raise ValueError('Cannot find original source rows: '+str(sorted(wanted-found)[:5]))
    print('PREFLIGHT pending rows:',dict(pending_counts),'numbering changes:',dict(normalization_counts),flush=True)
    output.mkdir(parents=True); audit.mkdir(parents=True)
    shutil.copyfile(source/'dataset_info.json',output/'dataset_info.json')
    shutil.copytree(source/'configs',output/'configs')
    counts={}; images=set(); ids_by_split={}; final_counts=C.Counter(); output_hashes={}; history_signatures={}
    with (audit/'row_provenance.jsonl').open('w') as provenance:
        for split in ('train','val'):
            count=C.Counter(); ids=set(); tids=set(); final_count=0
            with (output/(split+'.jsonl')).open('wb') as target:
                for directory in ('','review_pending','review_renumbering'):
                    p=source/directory/(split+'.jsonl') if directory else source/(split+'.jsonl')
                    if not p.exists(): continue
                    with p.open('rb') as f:
                        for ln,raw in enumerate(f,1):
                            original=json.loads(raw); working=original; restored=False
                            tid=original.get('trajectory_id') or original['id'].split(':')[0]
                            source_original_id=None
                            if not directory and tid in pending_tasks:
                                source_original_id=original_id(original['id'])
                                payload=database.execute('SELECT payload FROM originals WHERE id=?',(source_original_id,)).fetchone()[0]
                                working=json.loads(payload); restored=True
                                assert working.get('images')==original.get('images')
                                assert numbering.get_calls(working['conversations'][-1]['value'])==numbering.get_calls(original['conversations'][-1]['value'])
                            if directory or restored:
                                row,change=numbering.transform(working)
                                numbering.validate(row)
                                encoded=(json.dumps(row,ensure_ascii=False)+'\n').encode() if change['changed'] or restored else raw
                            else:
                                row=original; numbering.validate(row)
                                change={'changed':False}; encoded=raw
                            if row['id'] in ids: raise ValueError('duplicate sample ID: '+row['id'])
                            ids.add(row['id'])
                            tid=row.get('trajectory_id') or row['id'].split(':')[0]
                            tids.add(tid)
                            assert row.get('images')==original.get('images')
                            assert numbering.get_calls(row['conversations'][-1]['value'])==numbering.get_calls(original['conversations'][-1]['value'])
                            for old,new in zip(numbering.history(working),numbering.history(row)):
                                assert old['call']==new['call'] and old['observation']['result']==new['observation']['result']
                            for event in numbering.history(row):
                                signature=numbering.canon([event['call'],event['observation']['result']]); key=(tid,event['step'])
                                if key in history_signatures and history_signatures[key]!=signature: raise ValueError('Cross-prefix step conflict: '+str(key))
                                history_signatures[key]=signature
                            images.update(row.get('images',[]))
                            final=not numbering.get_calls(row['conversations'][-1]['value'])
                            final_count+=final; count['output_rows']+=1
                            count['pending_rows_added' if directory else 'previous_clean_rows']+=1
                            count['renumbered_pending_rows']+=bool(directory and change['changed'])
                            count['existing_rows_rebuilt_from_original_history']+=restored
                            target.write(encoded)
                            provenance.write(json.dumps({'output_file':split+'.jsonl','output_line':count['output_rows'],'output_id':row['id'],
                                'source_file':str(p),'source_line':ln,'source_id':original['id'],'source_row_sha256':hashlib.sha256(raw).hexdigest(),
                                'status':'pending_included_without_content_repair' if directory else 'previously_cleaned',
                                'numbering_updated':change['changed'],'shared_history_restored':restored,'original_source_id':source_original_id},ensure_ascii=False)+'\n')
            ids_by_split[split]=tids; final_counts[split]=final_count; count['final_answer_rows']=final_count
            assert count['pending_rows_added']==pending_counts[split]
            counts[split]=dict(count); output_hashes[split]=sha_file(output/(split+'.jsonl'))
            print(split,dict(count),flush=True)
    if ids_by_split['train'] & ids_by_split['val']: raise ValueError('Train/validation task overlap')
    missing=[p for p in sorted(images) if not (Path(p) if Path(p).is_absolute() else media/p).is_file()]
    if missing: raise ValueError('Missing images: '+str(missing[:5]))
    for name,run_name in [('lora.json','agent_lora'),('full.json','agent_full')]:
        path=output/'configs'/name
        if path.exists():
            cfg=json.loads(path.read_text()); cfg['dataset_dir']=str(output); cfg['media_dir']=str(media); cfg['output_dir']=str(output/'runs'/run_name)
            if 'deepspeed' in cfg: cfg['deepspeed']=str(output/'configs/zero3.json')
            dump(path,cfg)
    for p in inputs:
        assert sha_file(p)==before[str(p)]['sha256'], 'Source file changed: '+str(p)
    # Independently read the written training files; verify full row counts,
    # numbering and unique IDs on the actual bytes that will be uploaded.
    for split in ('train','val'):
        observed=0; seen=set()
        with (output/(split+'.jsonl')).open() as f:
            for line in f:
                r=json.loads(line); numbering.validate(r)
                assert r['id'] not in seen; seen.add(r['id']); observed+=1
        assert observed==counts[split]['output_rows']
    report={'completed':True,'input':str(source),'output':str(output),'media_dir':str(media),'audit_dir':str(audit),
        'counts':counts,'unique_image_paths':len(images),'missing_images':0,'source_integrity':before,'all_source_files_unchanged':True,
        'tasks_with_restored_shared_history':len(pending_tasks),'cross_prefix_event_consistency_verified':True,'output_sha256':output_hashes,'train_validation_tasks_overlap':0,'all_written_rows_reread_and_validated':True,
        'pending_samples_included':True,'pending_content_repaired':False,'history_duplicates_in_pending_preserved':True,
        'teacher_called':False,'training_run':False,'mask_history':True}
    dump(audit/'summary.json',report)
    dump(output/'inclusion_manifest.json',report)
    notes={'purpose':'User-requested inclusion of pending rows, with validated continuous numbering',
        'counts':counts,'pending_content_repaired':False,'mask_history':True,'teacher_called':False,'audit_dir':str(audit)}
    # Keep former export notes as a historical record rather than presenting them as current counts.
    old=output/'configs/export_notes.json'
    if old.exists():
        backup=output/'configs/export_notes_before_inclusion.json'; n=1
        while backup.exists():
            backup=output/'configs'/('export_notes_before_inclusion_'+str(n)+'.json'); n+=1
        old.rename(backup)
    dump(old,notes)
    database.close(); temporary.cleanup()
    print('COMPLETED',str(output),flush=True)
    return report

if __name__=='__main__': main()
