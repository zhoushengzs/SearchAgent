# 服务器评估：环境与启动

> 在项目代码根目录执行；数据、模型路径按你机器上的实际位置替换即可。
> 系统盘够，环境直接建在默认位置。

## 1. 先看 CUDA 版本

```bash
nvidia-smi | head -6
```

看 `CUDA Version` 这一项：

| CUDA Version | 装哪个 torch |
|---|---|
| 13.x | cu130 |
| 12.6 – 12.9 | cu126 |
| 12.0 – 12.5 | 需要先升级驱动 |

驱动号对照：580→13.0，570→12.8，560→12.6，550→12.4，535→12.2，530→12.1，525→12.0。

不需要安装 CUDA Toolkit。

## 2. 建环境（二选一）

**有 conda（推荐）：**

```bash
conda create -n vs-eval python=3.12 -y
conda activate vs-eval
python -m pip install -U pip
```

**没有 conda：**

```bash
python3.12 -m venv ~/vs-eval-env
source ~/vs-eval-env/bin/activate
python -m pip install -U pip
```

确认当前解释器：

```bash
which python
```

## 3. 装依赖

**先装 torch / torchvision**（按第 1 步的 CUDA 版本选一条）：

```bash
# CUDA 13.x
python -m pip --isolated install "torch==2.14.0" "torchvision==0.29.0" \
  --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.6 – 12.9
python -m pip --isolated install "torch==2.14.0" "torchvision==0.29.0" \
  --index-url https://download.pytorch.org/whl/cu126
```

**再装其余依赖：**

```bash
cd <代码目录>
python -m pip install -r evaluation/requirements.txt
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "from transformers import AutoProcessor; print('AutoProcessor OK')"
```

`is_available()` 必须是 True，`device_count()` 是实际卡数。

## 4. 开始评估

先跑 4 题确认链路（`--gpus` 按实际卡数改）：

```bash
cd <代码目录>
python -u -m evaluation.launch \
  --model ~/models/qwen3-8b-sft \
  --tasks ~/data/server_eval_778/tasks.jsonl \
  --expected-count 778 \
  --output outputs/smoke4 \
  --gpus 0,1 --limit 4 \
  --max-steps 10 --max-new-tokens 4096 --context-tokens 32768 \
  --max-pixels 262144 --image-side 1536 \
  --protocol training --repetition-penalty 1.10 --repeat-limit 2
```

成功标志：

```
Complete: 4 tasks; merged output: ...
```

再跑 16 题：

```bash
python -u -m evaluation.launch \
  --model ~/models/qwen3-8b-sft \
  --tasks ~/data/server_eval_778/tasks.jsonl \
  --expected-count 778 \
  --output outputs/smoke16 \
  --gpus 0,1 --limit 16 \
  --max-steps 10 --max-new-tokens 4096 --context-tokens 32768 \
  --max-pixels 262144 --image-side 1536 \
  --protocol training --repetition-penalty 1.10 --repeat-limit 2
```

全量 778 题（卡多就改 `--gpus`，建议 tmux 里跑）：

```bash
python -u -m evaluation.launch \
  --model ~/models/qwen3-8b-sft \
  --tasks ~/data/server_eval_778/tasks.jsonl \
  --expected-count 778 \
  --output outputs/full778 \
  --gpus 0,1,2,3,4,5,6,7 \
  --max-steps 10 --max-new-tokens 4096 --context-tokens 32768 \
  --max-pixels 262144 --image-side 1536 \
  --protocol training --repetition-penalty 1.10 --repeat-limit 2
```

结果在 `outputs/full778/merged/predictions.jsonl`。

中断后同一命令加 `--retry-infrastructure-errors` 续跑。

## 5. 常见问题

| 问题 | 处理 |
|---|---|
| `nvidia-cudnn-cu13==9.24.0.43` 找不到 | torch 没走 CUDA 索引，检查 `--index-url` |
| `different CUDA major versions` | torch 和 torchvision 没用同一个索引 |
| `Could not import module 'AutoProcessor'` | torchvision 缺失或版本不匹配，按第 3 步重装 |
| `Model path is absent` | `--model` 路径不对 |
| OOM | 减少卡数或改用更大显存卡 |
