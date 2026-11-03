"""Read-only server diagnostics; works without installing project dependencies."""
import argparse
import sys

if sys.version_info < (3, 8):
    print('当前 Python:', sys.version)
    print('此检查脚本至少需要 Python 3.8；本项目建议单独准备 Python 3.11 环境。')
    sys.exit(1)

import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess


def requirements(path, result=None):
    result = {} if result is None else result
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('-r '):
            requirements((path.parent / line[3:].strip()).resolve(), result)
        elif line:
            match = re.fullmatch(r'([A-Za-z0-9_.-]+)(==|>=)([0-9.]+)', line)
            if match:
                name, operator, version = match.groups()
                result[name] = (operator, version)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', type=int, default=2, help='Number of GPUs needed; default: 2')
    a = p.parse_args()
    if a.gpus < 1:
        p.error('--gpus must be positive')
    issues = []
    print('=== 系统与当前 Python 环境 ===', flush=True)
    print('系统:', platform.system(), platform.release(), platform.machine())
    os_release = Path('/etc/os-release')
    if os_release.is_file():
        for line in os_release.read_text().splitlines():
            if line.startswith('PRETTY_NAME='):
                print('发行版:', line.split('=', 1)[1].strip('"'))
    print('Python:', platform.python_version())
    print('Python 地址:', sys.executable)
    if sys.version_info < (3, 11):
        issues.append('当前 Python 低于项目建议的 3.11；请单独准备 Python 3.11 环境。')

    print('\n=== NVIDIA 显卡与驱动 ===', flush=True)
    if shutil.which('nvidia-smi'):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print('卡号, 型号, 显存 MiB, 驱动版本')
                print(result.stdout.strip())
            else:
                print('nvidia-smi 执行失败；需要平台检查显卡驱动或容器 GPU 挂载。')
        except subprocess.TimeoutExpired:
            print('nvidia-smi 超时；需要平台检查驱动状态。')
    else:
        print('没有找到 nvidia-smi；可能未装驱动，或当前容器未提供该工具。')

    print('\n=== 已安装依赖（仅检查，不安装） ===', flush=True)
    expected = requirements(Path(__file__).resolve().parent / 'requirements.txt')
    torch_installed = False
    for name, (operator, wanted) in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            print(f'[缺少] {name}；项目要求 {operator}{wanted}')
            issues.append(f'缺少 {name}{operator}{wanted}')
            continue
        if name == 'torch':
            torch_installed = True
        public = actual.split('+', 1)[0]
        if operator == '==':
            matches = public == wanted
        else:
            matches = tuple(map(int, re.findall(r'\d+', public)[:3])) >= tuple(map(int, wanted.split('.')))
        print(f'[{"符合" if matches else "版本不同"}] {name}: {actual}；项目要求 {operator}{wanted}')
        if not matches:
            issues.append(f'{name} 当前为 {actual}，与项目要求 {operator}{wanted} 不同')

    print('\n=== 当前 PyTorch 能否使用显卡 ===', flush=True)
    if torch_installed:
        probe = '''import json
try:
 import torch
 available = torch.cuda.is_available()
 count = torch.cuda.device_count() if available else 0
 devices = [{'index': i, 'name': torch.cuda.get_device_properties(i).name,
             'memory_gib': round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1)}
            for i in range(count)]
 print(json.dumps({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                   'cuda_available': available, 'gpu_count': count, 'devices': devices}))
except Exception as exc:
 print(json.dumps({'error': type(exc).__name__, 'detail': str(exc)[:400]}))
'''
        try:
            result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=40)
            data = json.loads(result.stdout.strip().splitlines()[-1])
            if 'error' in data:
                print('PyTorch 导入或 CUDA 检查失败:', data['error'], data.get('detail', ''))
                issues.append('PyTorch 导入或 CUDA 检查失败')
            else:
                print('PyTorch:', data['torch'])
                print('PyTorch 自带 CUDA 版本:', data['cuda_runtime'])
                print('GPU 可用:', data['cuda_available'])
                print('可见 GPU 数量:', data['gpu_count'])
                for device in data['devices']:
                    print('GPU {index}: {name}, {memory_gib} GiB'.format(**device))
                if not data['cuda_available']:
                    issues.append('当前 PyTorch 无法使用 NVIDIA GPU；需区分 CPU 安装包、驱动和容器配置问题')
                elif data['gpu_count'] < a.gpus:
                    issues.append(f'当前只能看到 {data["gpu_count"]} 张 GPU，需要 {a.gpus} 张')
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            print('PyTorch 检查未正常返回。')
            issues.append('PyTorch 检查失败或超时')
    else:
        print('尚未安装 PyTorch，无法继续检查 CUDA。')

    print('\n=== 检查结果 ===', flush=True)
    if issues:
        for issue in issues:
            print('- ' + issue)
        print('以上只列出差异，没有执行安装、卸载或驱动修改。')
        print('版本不同不一定代表完全不能运行；正式比较时两个模型需使用同一套确定的环境。')
    else:
        print(f'依赖版本及 {a.gpus} 张 GPU 的基本检查通过；下一步运行少量题目检查模型加载和显存。')
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
