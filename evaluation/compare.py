"""Compare two complete, paired runs under exactly the same evaluation policy."""
import argparse
import collections
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.prepare import dump, rows
from evaluation.run import fingerprint


def cross_backend_differences(base, sft):
    """Permit documented deployment differences, never different tasks or budgets."""
    protocols = [base['protocol'], sft['protocol']]
    if sum(p.get('device') == 'openrouter' for p in protocols) != 1:
        raise ValueError('Cross-backend comparison requires exactly one OpenRouter run')
    for run in (base, sft):
        if fingerprint(run['protocol']) != run['protocol_sha256']:
            raise ValueError('Protocol metadata was changed after inference')
    deployment_fields = {'version', 'device', 'dtype', 'versions', 'api'}
    shared = [{k: v for k, v in p.items() if k not in deployment_fields} for p in protocols]
    if shared[0] != shared[1]:
        raise ValueError('Tasks, prompts, tools, budgets or shared evaluation code differ')
    return {k: {'base': protocols[0].get(k), 'sft': protocols[1].get(k)}
            for k in sorted(deployment_fields) if protocols[0].get(k) != protocols[1].get(k)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-scores', required=True, type=Path)
    p.add_argument('--sft-scores', required=True, type=Path)
    p.add_argument('--base-run', required=True, type=Path)
    p.add_argument('--sft-run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--allow-cross-backend', action='store_true',
                   help='Explicitly label OpenRouter/local deployment differences; still require identical tasks and budgets')
    a = p.parse_args()
    base_config = json.loads((a.base_scores / 'score_config.json').read_text())
    sft_config = json.loads((a.sft_scores / 'score_config.json').read_text())
    base_run = json.loads((a.base_run / 'run.json').read_text())
    sft_run = json.loads((a.sft_run / 'run.json').read_text())
    deployment_differences = {}
    if base_run['protocol_sha256'] != sft_run['protocol_sha256']:
        if not a.allow_cross_backend:
            p.error('Models used different protocols. OpenRouter/local comparisons require --allow-cross-backend.')
        try:
            deployment_differences = cross_backend_differences(base_run, sft_run)
        except ValueError as exc:
            p.error(str(exc))
    if base_config['policy_id'] != sft_config['policy_id']:
        p.error('Judge policies differ')
    for config, run in [(base_config, base_run), (sft_config, sft_run)]:
        if config['run_id'] != run['run_id']:
            p.error('Score directory does not belong to the selected model run')
    base = {r['id']: r for r in rows(a.base_scores / 'scores.jsonl')}
    sft = {r['id']: r for r in rows(a.sft_scores / 'scores.jsonl')}
    expected = set(base_run['protocol']['selected_ids'])
    if set(base) != expected or set(sft) != expected:
        p.error('Both models must have a score for every fixed selected task')
    for data, config, run_dir in [(base, base_config, a.base_run), (sft, sft_config, a.sft_run)]:
        if any(r['policy_id'] != config['policy_id'] for r in data.values()):
            p.error('Mixed judge policies in score records')
        predictions = {r['id']: r for r in rows(run_dir / 'predictions.jsonl')}
        if set(predictions) != expected or any(r.get('infrastructure_error') for r in predictions.values()):
            p.error('Prediction run is incomplete or still has infrastructure errors')
        if any(data[key]['prediction_sha256'] != fingerprint(predictions[key]) for key in expected):
            p.error('Predictions changed after scoring. Re-run scoring before comparing.')
    groups = collections.defaultdict(list)
    for key in sorted(expected):
        if base[key]['input_fingerprint'] != sft[key]['input_fingerprint'] or base[key]['source_dataset'] != sft[key]['source_dataset']:
            p.error('Task identities differ')
        groups[base[key]['source_dataset']].append(key)
    table = []
    for source, ids in sorted(groups.items()):
        n = len(ids)
        base_score = 100 * sum(base[i]['score'] for i in ids) / n
        sft_score = 100 * sum(sft[i]['score'] for i in ids) / n
        table.append({'benchmark': source, 'n': n, 'base_score_pct': base_score,
                      'sft_score_pct': sft_score, 'delta_percentage_points': sft_score - base_score,
                      'sft_wins': sum(sft[i]['score'] > base[i]['score'] for i in ids),
                      'sft_losses': sum(sft[i]['score'] < base[i]['score'] for i in ids),
                      'base_answer_rate_pct': 100 * sum(base[i]['status'] == 'answered' for i in ids) / n,
                      'sft_answer_rate_pct': 100 * sum(sft[i]['status'] == 'answered' for i in ids) / n,
                      'base_mean_tool_calls': sum(base[i]['tool_calls'] for i in ids) / n,
                      'sft_mean_tool_calls': sum(sft[i]['tool_calls'] for i in ids) / n})
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    macro_base = sum(r['base_score_pct'] for r in table) / len(table)
    macro_sft = sum(r['sft_score_pct'] for r in table) / len(table)
    report = {'metric': base_config['policy']['metric'], 'datasets': table,
              'macro_base_pct': macro_base, 'macro_sft_pct': macro_sft,
              'macro_delta_percentage_points': macro_sft - macro_base,
              'base_model': base_run['model'], 'sft_model': sft_run['model'],
              'comparison_type': 'cross_backend' if deployment_differences else 'same_protocol',
              'deployment_differences': deployment_differences}
    dump(a.output / 'comparison.json', report)
    text = ['# 本地留出集对比', '',
            '以下为项目统一参考答案裁判正确率，不是各 benchmark 的官方完整榜单分数。', '',
            '| Benchmark | 题数 | 原始模型 | SFT 模型 | 差值（百分点） |',
            '|---|---:|---:|---:|---:|']
    if deployment_differences:
        text[0] = '# OpenRouter / 本地 SFT 留出集对比'
        text.insert(2, '这是不同部署后端的比较：服务端聊天模板、图片处理、精度和 token 计数可能不同，不能把分差完全归因于 SFT。具体差异见 comparison.json。\n')
    for row in table:
        text.append(f'| {row["benchmark"]} | {row["n"]} | {row["base_score_pct"]:.2f}% | {row["sft_score_pct"]:.2f}% | {row["delta_percentage_points"]:+.2f} |')
    text.extend(['', f'各数据集等权平均：原始模型 {macro_base:.2f}%，SFT {macro_sft:.2f}%，差值 {macro_sft-macro_base:+.2f} 个百分点。',
                 '', '这是开发留出集比较。请结合逐题记录检查分数变化，最终结论需要足够题量。'])
    (a.output / 'comparison.md').write_text('\n'.join(text) + '\n')
    print('\n'.join(text))


if __name__ == '__main__':
    main()
