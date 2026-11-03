"""Group existing commands without importing GPU frameworks or changing their defaults."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    'trajectory': {
        'prepare': ('准备采集题目', 'prepare_hf_data.py'),
        'split': ('划分训练与保留题目', 'partition_trajectory_data.py'),
        'collect': ('采集模型搜索与回答轨迹', 'agent_pipeline.py', 'run'),
        'review': ('审核轨迹与答案', 'agent_pipeline.py', 'verify'),
        'export': ('把审核后的轨迹导出为训练数据', 'agent_pipeline.py', 'export'),
        'stats': ('统计数据', 'stats.py'),
    },
    'sft': {
        'clean': ('清洗SFT数据', 'sft_processing/cleaning/clean_sft.py'),
        'check': ('检查清洗后的数据', 'sft_processing/cleaning/verify_cleaned.py'),
        'renumber': ('整理步骤编号', 'sft_processing/renumbering/renumber_sft.py'),
        'teacher': ('运行教师数据处理', 'sft_processing/teacher/start_teacher.py'),
        'configure': ('创建本地教师处理配置', 'sft_processing/configure.py'),
    },
    'grpo': {
        'control': ('数据注册、检查及GRPO训练控制', 'datasets/grpo/scripts/training_control.py'),
        'check-tools': ('检查训练工具', 'datasets/grpo/scripts/check_project_tools.py'),
    },
    'evaluate': {
        'run': ('让已有模型回答题目', '-m', 'evaluation.launch'),
        'score': ('给模型答案评分', '-m', 'evaluation.score'),
        'package': ('打包固定题目与图片', '-m', 'evaluation.package'),
        'compare': ('比较评测结果', '-m', 'evaluation.compare'),
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='按阶段查找命令。先用 阶段 --help 查看操作，再用 阶段 操作 --help 查看参数。')
    parser.add_argument('--dry-run', action='store_true', help='只显示命令，不执行；必须放在阶段名称前')
    stages = parser.add_subparsers(dest='stage', required=True)
    for stage, actions in COMMANDS.items():
        group = stages.add_parser(stage, help={'trajectory':'采集轨迹','sft':'SFT数据处理','grpo':'GRPO强化学习','evaluate':'评测和评分'}[stage])
        group.add_argument('action', choices=actions, nargs='?')
        group.epilog = '\n'.join(f'{key}: {value[0]}' for key,value in actions.items())
        group.formatter_class = argparse.RawDescriptionHelpFormatter
    supplied = list(sys.argv[1:] if argv is None else argv)
    boundary = next((i + 2 for i, item in enumerate(supplied[:-1])
                     if item in COMMANDS and supplied[i + 1] in COMMANDS[item]), None)
    args = parser.parse_args(supplied if boundary is None else supplied[:boundary])
    extra = [] if boundary is None else supplied[boundary:]
    if not args.action:
        parser.error('请指定操作；例如 python -m workflows trajectory --help')
    entry = COMMANDS[args.stage][args.action]
    command = [sys.executable, *entry[1:], *extra]
    if args.dry_run:
        print('执行目录:', ROOT)
        print(shlex.join(command))
        return 0
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == '__main__':
    raise SystemExit(main())
