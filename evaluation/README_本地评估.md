# 在这台 Mac 上比较原始模型与 SFT 模型

## 现在执行到哪一步

原始模型已于 2026-10-19 下载完成，四个权重分片、权重索引、tokenizer 和配置文件均已到位。现在执行第一步进入项目，然后执行第三步，先跑原始模型的 18 道试跑题，完成后再跑 SFT 模型的同一批题。第二步重新下载命令目前不需要执行。

两个模型的逐题记录都保留 `source_dataset`，可在生成答案后按 benchmark 分别评分。第四步现有脚本使用统一参考答案裁判，再逐 benchmark 分别统计；如果目标是每个 benchmark 各自的官方指标，先保存第三步生成的答案，待对应官方评分器接入后再计算。完整官方评分适配尚未完成。

先检查 18 道试跑题的完成状态和工具记录，再决定是否执行第五步的 3,103 道完整推理。

已将每轮输出上限从 2,048 调整为 4,096 个词元，总上下文预算从 8,192 调整为 32,768 个词元。下方两模型命令使用相同新参数，并改用带 `_32k` 的新结果目录，避免与旧预算记录混用。已启动的进程仍使用启动时的旧参数；如需改用新预算，在运行窗口按 `Control+C` 停止旧进程，等终端提示符重新出现后再执行下方新命令。旧结果保留。

长上下文会增加内存占用和耗时；已验证本机可以加载模型和短上下文推理，尚未验证在 32,768 预算接近用满时的内存情况。出现内存不足时，应为两个模型共同调整预算，并使用新的结果目录。

## 当前已完成

- 电脑：Apple M5，32GB 统一内存。
- SFT 模型：`/Users/ss/models/OpenSearch-VL-qwen3-8b`，完整 BF16 权重，约 16GB。
- 已实际用 Apple GPU 读图生成；还实际执行了一条留出题的两次搜索工具调用，两次成功。这次仅给了两步，最终停在步骤上限，不是正式成绩。
- 基于现有留出集准备了评估副本：排除 15 道 M3VQA 题目，因为其初始图片或 512 像素预览图出现在本地 SFT 训练数据中。剩余 3,103 题。
- 小规模试跑固定为每个已存在的 benchmark 3 题，共 18 题。
- DynVQA 不在现有留出数据中，暂缺。未重新划分原数据。
- 环境位于项目 `.venv-eval`，使用电脑现有 `llamafactory` 环境的依赖；未向原 `visual-search-agent` 环境安装包。

准备后的数据仍称为“项目开发留出集”：它包含训练验证集相关题目，也未排除所有感知近重复。最终论文测试需要另外确认模型选择和训练数据来源。

## 文件地址

项目目录：

```text
/Users/ss/代码and Paper/paper/visual-search-agent/VS_Captioning new
```

所有后续命令都在这个目录执行。

| 内容 | 项目内位置 |
|---|---|
| 评估脚本和本说明 | `evaluation/` |
| 全部固定评估题 | `outputs/local_benchmark_eval/data/tasks.jsonl` |
| 18 道试跑题 | `outputs/local_benchmark_eval/data/pilot_tasks.jsonl` |
| 独立答案文件，仅评分程序读取 | `outputs/local_benchmark_eval/data/references.jsonl` |
| 数据核对、排除数量、文件校验值 | `outputs/local_benchmark_eval/data/manifest.json` |
| 被排除题目及原因 | `outputs/local_benchmark_eval/data/excluded_train_overlap.json` |
| 原始模型下载目标（项目外） | `/Users/ss/models/Qwen3-VL-8B-Instruct/` |
| 下载日志 | `outputs/sft_eval_preparation/local_smoke/download_base.log` |
| 单图 GPU 运行检查 | `outputs/sft_eval_preparation/local_smoke/sft_gpu_smoke.json` |
| 两步真实工具检查 | `outputs/local_benchmark_eval/sft_pipeline_check_v2/` |

原始留出集文件没有修改。数据副本使用本机图片的绝对路径，复制到另一台电脑后需重新生成对应路径。

## 第一步：进入项目目录

打开 Mac 的“终端”，复制执行：

```bash
cd "/Users/ss/代码and Paper/paper/visual-search-agent/VS_Captioning new"
```

`cd` 表示切换到一个文件夹；双引号是因为路径中有空格。

不需要先激活 Conda 环境。后面使用 `.venv-eval/bin/python`，它表示这个项目评估环境中的 Python。

## 第二步：准备原始模型

原始模型已下载完成，目录为 `/Users/ss/models/Qwen3-VL-8B-Instruct/`。当前直接跳到第三步即可。以下下载命令仅保留作以后修复缺少文件时使用；不要在另一份下载尚在运行时重复启动。

如果下载进程已经结束但文件不完整，可执行以下命令重新下载缺少的文件。当前安装的 huggingface_hub 1.31.0 会复用完整文件，但不会接着下载未完成的大文件；中断或重启会重下对应分片，因此请让现有下载进程继续运行。

```bash
HF_HOME="$PWD/outputs/local_benchmark_eval/hf_cache" \
  .venv-eval/bin/python evaluation/download_base.py
```

`HF_HOME` 把下载缓存放到项目内；`$PWD` 表示当前所在目录。反斜杠 `\` 表示同一条命令换行。

默认下载官方 `Qwen/Qwen3-VL-8B-Instruct`，与本地原训练配置一致。不能用 Thinking 版本或其他人的已量化模型来替代这个基线而不另行说明。

原始模型及已有下载文件已按要求移到 `/Users/ss/models/Qwen3-VL-8B-Instruct/`，后续文件下载到此目录。搬移后重启时，约 2.9GB 旧临时文件没有被当前下载库接着使用，不能计入有效进度。

官方四个权重分片合计约 17.53GB（16.33GiB），另外还有 tokenizer 和配置文件。未完成的旧临时文件会额外占用空间，不能计为完成进度；下载结束前不能把目标目录当成完整模型。

## 第三步：先跑 18 道题

搜索需要已有 `.env` 中的 Serper 密钥。程序会读取配置，运行日志不会输出密钥。两模型都使用相同的 6 个项目 `vs_*` 工具；未提供反向搜图、外部 OCR 或任意代码执行工具。这是项目工具环境，不是对各 benchmark 官方环境的完全复现。

先运行原始模型：

```bash
.venv-eval/bin/python evaluation/run.py \
  --model "/Users/ss/models/Qwen3-VL-8B-Instruct" \
  --tasks "$PWD/outputs/local_benchmark_eval/data/pilot_tasks.jsonl" \
  --max-new-tokens 4096 \
  --context-tokens 32768 \
  --output "$PWD/outputs/local_benchmark_eval/base_pilot_32k"
```

完成后运行 SFT 模型：

```bash
.venv-eval/bin/python evaluation/run.py \
  --model "/Users/ss/models/OpenSearch-VL-qwen3-8b" \
  --tasks "$PWD/outputs/local_benchmark_eval/data/pilot_tasks.jsonl" \
  --max-new-tokens 4096 \
  --context-tokens 32768 \
  --output "$PWD/outputs/local_benchmark_eval/sft_pilot_32k"
```

参数含义：

- `--model`：本次要加载的模型目录。
- `--tasks`：固定题目文件；两模型必须用同一个文件。
- `--max-new-tokens 4096`：每轮回答最多生成 4,096 个词元，是上限，不会强制模型输出这么长。
- `--context-tokens 32768`：图片、问题、历史工具结果与本轮预留输出共同使用的总预算。程序会先预留 4,096 个输出词元，所以输入历史最多可占 28,672 个词元。
- `--output`：本次结果保存到哪里；两个模型各用自己的目录。

默认设置：Apple GPU、BF16、每题最多 30 次决策、每次最多生成 4,096 个词元、总上下文预算 32,768 个词元。词元是模型处理文字的基本单位，不等于汉字数。达到上限的题目会保留为失败记录，不会从题数中删除。

题目会逐条保存在 `predictions.jsonl`，实际搜索、图片裁剪等材料保存在 `artifacts/`，本次模型和参数保存在 `run.json`。

中断后执行原命令，会跳过已经有结果的题目。需要重试网络等基础设施错误时，在同一命令后加：

```text
--retry-infrastructure-errors
```

如果改了脚本、模型或参数，使用新的 `--output` 目录。程序会拒绝把不同配置的结果混在一起。

## 第四步：评分并比较

当前提供的评分方式是“项目统一参考答案裁判正确率”，用于同条件开发比较。它不是 M3VQA、TIR-Bench 等各自官方评分器的替代品，也不能写成官方完整榜单分数。官方 M3VQA 包含多答案评分，TIR-Bench 包含按任务类型区分的规则。

本地运行的两个模型负责作答。评分使用同一个外部裁判模型，因此评分需要网络和裁判 API，会产生对应服务费用。以下沿用项目 README 使用过的 DeepSeek 裁判名称；如果该名称在账号中不可用，需为两个模型共同更换一个可用且独立于 Qwen 的裁判。

```bash
.venv-eval/bin/python evaluation/score.py \
  --run "$PWD/outputs/local_benchmark_eval/base_pilot_32k" \
  --output "$PWD/outputs/local_benchmark_eval/base_pilot_32k_scores" \
  --judge-model deepseek/deepseek-v4.1-flash

.venv-eval/bin/python evaluation/score.py \
  --run "$PWD/outputs/local_benchmark_eval/sft_pilot_32k" \
  --output "$PWD/outputs/local_benchmark_eval/sft_pilot_32k_scores" \
  --judge-model deepseek/deepseek-v4.1-flash
```

`--run` 指向已经生成答案的结果目录；`--judge-model` 指定裁判。程序读取 `.env` 中的 `OPENROUTER_API_KEY`。不把模型名称告诉裁判，也不把参考答案提供给作答模型。

如果有未处理的网络/推理错误，评分会停止，先重试这些题。模型自己达到步数上限或输出格式错误，则按没有完整答案记 0 分。

两个模型都完成评分后生成对照表：

```bash
.venv-eval/bin/python evaluation/compare.py \
  --base-run "$PWD/outputs/local_benchmark_eval/base_pilot_32k" \
  --sft-run "$PWD/outputs/local_benchmark_eval/sft_pilot_32k" \
  --base-scores "$PWD/outputs/local_benchmark_eval/base_pilot_32k_scores" \
  --sft-scores "$PWD/outputs/local_benchmark_eval/sft_pilot_32k_scores" \
  --output "$PWD/outputs/local_benchmark_eval/pilot_comparison_32k"
```

输出 `comparison.csv`、`comparison.json`、`comparison.md`，显示逐 benchmark 原模型分数、SFT 分数、百分点差、作答完成率和平均工具次数。

比如从 30% 到 40%，差值是“增加 10 个百分点”。试跑每个数据集只有 3 题，只用来检查流程，不能据此判断训练是否有效。

## 第五步：正式完整比较

确认试跑正常后，将上面的 `--tasks` 改为：

```text
outputs/local_benchmark_eval/data/tasks.jsonl
```

两个模型均使用此文件，并使用新的输出目录，例如 `base_full` 和 `sft_full`。评分时也必须明确传入同一个完整任务文件：

```text
--tasks "$PWD/outputs/local_benchmark_eval/data/tasks.jsonl"
```

默认评分输入仍是试跑任务；不要漏掉这个参数。完整的多轮本地评估可能很久，不要同时加载两个模型。

## 官方评分的衔接范围

评分程序会额外导出 `m3vqa_official_input.jsonl`，供 M3VQA 官方评分脚本使用。若模型没有输出 JSON 列表，则保留整段答案作为一个条目，不擅自按逗号拆分人名或地名。

`tirbench_answers_for_adapter.json` 保留模型答案和当前元数据，但还需要原始 TIR 任务元数据、官方答案提取步骤，才可运行官方分类评分。当前没有实现全部 6 个 benchmark 的官方评分适配。

参考：

- [M3VQA 官方评分](https://github.com/CASIA-IVA-Lab/M3VQA/blob/main/run_eval.py)
- [TIR-Bench 官方评分](https://github.com/agents-x-project/TIR-Bench/blob/main/calculate_score.py)
- [MMBrowseComp 官方评估说明](https://github.com/MMBrowseComp/MM-BrowseComp#-quick-start)

## 已验证与未完成

已验证：模型本地读图、两次真实搜索工具调用、固定留出文件和图片完整性，以及防止漏题、混用配置、使用过期评分等 10 项自动检查。

尚未完成：两个模型的完整试跑、最终跑分、全部官方评分适配、DynVQA 数据准备、服务器实际训练数据与当前模型的来源绑定。因此本目录当前不包含真实的两模型成绩对照表。
