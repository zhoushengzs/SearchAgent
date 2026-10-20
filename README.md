# VS_Captioning

服务器评估入口：`python -m evaluation.launch --help`。八卡按题目分工，使用独立模型副本；评分和比较仍由 `evaluation/score.py`、`evaluation/compare.py` 完成。
迁移固定评估题和原始图片可使用 `python -m evaluation.package --help`；数据、模型和真实密钥需单独传输，不包含在源码中。服务器额外依赖见 `evaluation/requirements.txt`。

## 数据集

| CLI key | 项目 / 数据 | 状态 |
| --- | --- | --- |
| agentvista | [项目](https://github.com/hkust-nlp/AgentVista) · [HF](https://huggingface.co/datasets/Warrieryes/AgentVista) | DONE |
| mmsearch_plus | [项目](https://mmsearch-plus.github.io/) · [HF](https://huggingface.co/datasets/Cie1/MMSearch-Plus) | DONE |
| mmbrowsecomp | [项目](https://github.com/MMBrowseComp/MM-BrowseComp) · [数据](https://github.com/MMBrowseComp/MM-BrowseComp/blob/cf82a2390aa7a7ed3d96f72969d22548eff3e101/data/MMBrowseComp_400.jsonl) · [HF](https://huggingface.co/datasets/mmbrowsecomp/MMBrowseComp) | DONE |
| browsecomp_v3 | [项目](https://github.com/Halcyon-Zhang/BrowseComp-V3) · [HF](https://huggingface.co/datasets/Halcyon-Zhang/BrowseComp-V3) | DONE |
| m3vqa | [项目](https://github.com/CASIA-IVA-Lab/M3VQA) · [HF](https://huggingface.co/datasets/CASIA-IVA-Lab/M3VQA) | DONE |
| tirbench | [项目](https://github.com/agents-x-project/TIR-Bench) · [HF](https://huggingface.co/datasets/Agents-X/TIR-Bench) | DONE |
| dynvqa | [项目](https://github.com/Alibaba-NLP/OmniSearch) · [数据](https://github.com/Alibaba-NLP/OmniSearch/blob/b01e78edb694fa0e8b05b808d2722d046f3f204b/dataset/DynVQA_en/DynVQA_en.202502.jsonl) · [HF](https://huggingface.co/datasets/zhzhen23/DynVQA) | DONE |
| — / VisualToolBench | [论文](https://arxiv.org/abs/2510.12712) · [HF](https://huggingface.co/datasets/ScaleAI/VisualToolBench) | TODO |

## 安装

~~~bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OPENROUTER_API_KEY=...
export SERPER_API_KEY=...
~~~

## 准备与划分

~~~bash
python prepare_hf_data.py \
  --sources agentvista,mmsearch_plus,mmbrowsecomp,browsecomp_v3,m3vqa,tirbench \
  --format benchmark --streaming --keep-all \
  --output data/benchmarks/tasks.jsonl

python partition_trajectory_data.py \
  --input data/benchmarks/tasks.jsonl \
  --output-prefix data/seeds/search_v1 --split-name search-v1 \
  --holdout-fraction 0.2 --seed 42
~~~

## 全量生成与审核

以下两款模型使用 `low` 推理强度。遇到 429 后等待 30、60 秒重试；单条任务连续三次限流后记为失败，并继续处理后续任务。外层括号避免报错关闭当前终端。降低推理强度可能影响复杂题表现，也不保证消除输出截断。

~~~bash
(
set -e
set -a
source .env
set +a
for split in train holdout; do
  python agent_pipeline.py run \
    --input "data/seeds/search_v1_${split}.jsonl" \
    --output "data/rollouts/search_v1_qwen38flash_ds41flash_low_${split}.jsonl" \
    --provider openrouter --model qwen/qwen3.8-flash \
    --max-steps 30

  python agent_pipeline.py verify \
    --input "data/rollouts/search_v1_qwen38flash_ds41flash_low_${split}.jsonl" \
    --output "data/verified/search_v1_qwen38flash_ds41flash_low_${split}.jsonl" \
    --references data/benchmarks/tasks_references.jsonl \
    --provider openrouter --judge-model deepseek/deepseek-v4.1-flash

  python stats.py --input "data/verified/search_v1_qwen38flash_ds41flash_low_${split}.jsonl"
done
)
~~~

## 导出训练集

~~~bash
python agent_pipeline.py export \
  --input data/verified/search_v1_qwen38flash_ds41flash_low_train.jsonl \
  --output data/train/agent_calls_qwen38flash_ds41flash_low.jsonl
~~~
