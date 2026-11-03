"""Check the six real project tools without loading a model or allocating GPUs."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'integration'))


def run_checks(live=False):
    from PIL import Image
    from project_runtime.session import ProjectSession, get_all_tools
    from project_runtime.serper_keys import configured_keys
    if live and not configured_keys():
        raise RuntimeError('请在 integration/local_training.env 填写 SERPER_API_KEYS 或 SERPER_API_KEY。不需要 COS。')
    summary = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'live_network': live, 'requires_cos': False, 'model_loaded': False,
               'tools': sorted(get_all_tools()), 'checks': {}}
    with tempfile.TemporaryDirectory(prefix='grpo-tool-check-') as directory:
        image = Image.new('RGB', (160, 100), 'red')
        session = ProjectSession('Check the supplied image and retrieve public evidence.', [image], directory, 12)
        calls = [('vs_inspect_image', {'asset_id': 'image_1'}),
                 ('vs_crop', {'asset_id': 'image_1', 'bbox': [0.25, 0.2, 0.75, 0.8]}),
                 ('vs_calculate', {'expression': '(12+8)*3'})]
        step = 0
        def execute(name, args):
            nonlocal step
            step += 1
            result, message = session.execute(name, args, step)
            if result['status'] != 'ok':
                raise RuntimeError(f'{name} 失败：{result.get("error_type")}，HTTP {result.get("http_status", "未提供")}')
            return result, message
        for name, args in calls:
            result, message = execute(name, args)
            if name == 'vs_calculate':
                assert result['data']['value'] == 60
            else:
                assert len(message.get('images', [])) == 1
                if name == 'vs_crop':
                    assert message['images'][0].size == (80, 60)
            summary['checks'][name] = 'passed'
        if live:
            print('正在检查真实网页搜索……', flush=True)
            web, _ = execute('vs_web_search', {'query': 'Python official documentation tutorial'})
            rows = web['data']['results']
            if not rows:
                raise RuntimeError('网页搜索没有返回结果，无法验证打开网页。')
            summary['checks']['vs_web_search'] = 'passed'
            visit_failures = []
            for row in rows:
                print('正在读取搜索返回的网页……', flush=True)
                result, _ = session.execute('vs_visit', {'url': row['url']}, step + 1)
                step += 1
                if result['status'] == 'ok' and result['data']['text'].strip():
                    summary['checks']['vs_visit'] = 'passed'
                    break
                visit_failures.append({'error_type':result.get('error_type'), 'http_status':result.get('http_status')})
            else:
                raise RuntimeError('搜索成功，但返回的网页均未能读取：' + json.dumps(visit_failures))
            print('正在检查文字找图片……', flush=True)
            images, _ = execute('vs_image_search', {'query': 'domestic cat photograph'})
            summary['checks']['vs_image_search'] = 'passed'
            for row in images['data']['results']:
                if not row.get('image_url'):
                    continue
                print('正在下载搜索返回的图片……', flush=True)
                result, message = session.execute('vs_inspect_image', {'image_url': row['image_url']}, step + 1)
                step += 1
                if result['status'] == 'ok' and message.get('images'):
                    summary['checks']['search_image_loaded_into_next_message'] = 'passed'
                    break
            else:
                raise RuntimeError('图片搜索返回后，尚无一张结果图片能下载并交给下一轮输入。')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='实际搜索网页和图片，各调用一次搜索服务')
    args = parser.parse_args()
    output = ROOT / '.codex' / ('project_tools_live_check.json' if args.live else 'project_tools_local_check.json')
    output.parent.mkdir(exist_ok=True)
    env_file = ROOT / 'integration/local_training.env'
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    try:
        report = run_checks(args.live)
    except Exception as exc:
        output.write_text(json.dumps({'passed':False, 'live_network':args.live,
                                     'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                     'failure':str(exc)},ensure_ascii=False,indent=2) + '\n')
        print('检查未通过：' + str(exc))
        raise SystemExit(1)
    report['passed'] = True
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('检查结果：', output)
    if not args.live:
        print('只检查了本地看图、裁剪、计算；联网搜索和网页读取需要加 --live 检查。')


if __name__ == '__main__':
    main()
