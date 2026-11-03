# SearchAgent：视觉搜索、GRPO训练与评测

这个项目让模型结合图片、网页搜索和工具回答问题，也包含训练与评测代码。请先区分三件事：**训练**会改变模型权重；**评测作答**让已有模型回答题目，不改变权重；**评分**把模型答案与参考答案交给裁判模型判断。

主分支已汇总原服务器训练代码、评测代码和本地数据处理工具。模型、图片、训练数据、真实密钥和运行环境需要另行准备。部分实验入口未逐个做GPU验证，不能将“代码已汇总”理解为所有入口都可以直接运行。

## 1. 目录是什么

| 目录或文件 | 用途 | 什么时候使用 |
|---|---|---|
| `agent/` | 搜索工具、作答流程和记录 | 调整模型如何搜索、看图、回答 |
| `core/`、`sources/`、`prompts/` | 基础配置、数据来源、提示词 | 开发或检查作答规则 |
| `evaluation/` | 单卡/多卡作答、合并结果、裁判评分 | 测试已有模型 |
| `evaluation/training_protocol/` | 训练风格评测所用提示词 | 核对模型收到什么指令 |
| `cluster/` | 原集群的环境、提交与训练启动脚本 | 在原服务器启动GRPO |
| `datasets/grpo/scripts/` | 数据注册、检查和训练控制 | 准备训练数据、检查工具 |
| `datasets/grpo/integration/` | 项目工具与训练框架的连接代码、奖励规则 | 核对训练过程和评分奖励 |
| `train/OpenSearch-VL-Training/` | 训练框架及相关依赖源码 | GRPO及上游训练流程，保留原许可证 |
| `sft_processing/` | SFT数据清洗、编号、教师处理等工具 | 制作或修订SFT数据，不等同于开始训练 |
| `official_eval/` | 部分公开评测的专用入口 | 需要对应评测流程时，先阅读其帮助 |
| `prepare_hf_data.py`、`partition_trajectory_data.py` | 准备和划分题目 | 制作新的数据集 |
| `agent_pipeline.py` | 数据作答、审核、导出流程 | 制作训练轨迹 |
| `.env.example` | 配置格式参考 | 新建本机配置，勿提交真实密钥 |
| `outputs/` | 评测结果的建议保存位置 | 运行后查看结果；不会提交到Git |
| `.codex/` | 本地整理说明 | 不是运行所需代码，也不会提交到Git |

## 2. 下载代码

在准备存放项目的目录执行：

```bash
git clone https://github.com/zhoushengzs/SearchAgent.git
cd SearchAgent
```

第一条把GitHub主分支下载成`SearchAgent`文件夹；第二条进入它。后面的相对路径命令均从这个文件夹执行。已经有这份仓库时，不要重复下载。

```bash
git status --short
git pull --ff-only origin main
```

第一条查看是否有尚未提交的改动；第二条更新主分支。若提示冲突或不能更新，先保留自己的修改，不要用强制覆盖解决。

本次整理使用的电脑目录是：`/Users/ss/code/paper/visual-search-agent/SearchAgent-server-merged-20261103`。GitHub更新不会自动更新服务器上的运行目录。

## 3. 环境安装：请选择与你的用途对应的一种

### 3.1 已部署的原服务器：复用现有环境

原服务器已验证的主要版本是Python 3.12、PyTorch 2.8.0/CUDA 12.8、Transformers 4.57.1。训练脚本通过`cluster/run_container.sh`进入已准备的Ubuntu环境，再由`cluster/runtime.sh`选用现有Conda环境。

**不要在正在训练的环境中安装`evaluation/requirements.txt`。** 这个文件面向另一套评测依赖，目前固定了PyTorch 2.14.0和Transformers 5.6.0，与上述服务器环境不同。

### 3.2 新机器：建立独立评测环境

下面是以原服务器主要版本为基础的新环境安装示例，适用于具有兼容NVIDIA驱动的Linux机器。它不包含完整GRPO训练依赖，也不保证适用于所有显卡和模型。

```bash
conda create -n searchagent-eval python=3.12 -y
conda activate searchagent-eval
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt transformers==4.57.1 accelerate==1.15.0 python-dotenv qwen-vl-utils
python -m pip check
```

- `conda create`创建名为`searchagent-eval`的独立环境；`-y`表示确认创建。
- `conda activate`进入该环境。以后新开终端运行评测，也要先执行它。
- 第一条安装命令安装匹配的GPU计算库；地址是PyTorch的CUDA 12.8安装源。
- 第二条安装本项目基础依赖、模型加载和图片处理依赖。
- `pip check`检查依赖之间是否冲突；出现错误时先修复，再运行任务。

```bash
nvidia-smi
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print('GPU可用:', torch.cuda.is_available())"
python -m evaluation.launch --help
```

这三条分别查看显卡、检查当前Python能否使用GPU、查看评测程序的参数说明。集群上必须在分配到GPU的计算任务内检查GPU；登录节点可能没有显卡。`--help`成功只说明入口能启动，不代表模型已成功运行。

### 3.3 新机器需要训练时

GRPO除了上述基础库，还依赖SGLang、Ray、Megatron、Transformer Engine、编译工具和数据注册。`cluster/requirements-a100.txt`与`cluster/constraints-a100.txt`记录原服务器部分依赖与版本限制，**不是完整的一键部署脚本**。请结合`train/OpenSearch-VL-Training/`内的说明部署，再检查模型、数据、工具请求与短程训练。

原集群还需要`containers/ubuntu22-rootfs`、`envs/search-grpo-a100`、CUDA工具链及Singularity模块，这些不在Git仓库中。换服务器时，需修改`cluster/`和`evaluation/server_benchmark*`中的原服务器绝对路径，不能原样照抄运行。

## 4. 配置密钥与数据

新环境可以从模板创建配置（已存在时保留原文件）：

```bash
cp -n .env.example .env
chmod 600 .env
```

`cp -n`只在目标不存在时复制；`chmod 600`让配置只允许当前用户读写。用文本编辑器填写自己的搜索服务密钥和评分服务配置。`.env`不会上传GitHub。

训练使用另一份配置：

```bash
cp -n datasets/grpo/integration/training.env.example datasets/grpo/integration/local_training.env
chmod 600 datasets/grpo/integration/local_training.env
```

按模板填写`SERPER_API_KEYS`或单个搜索密钥，以及`JUDGE_BASE_URL`、`JUDGE_API_KEY`、`JUDGE_MODEL`。它们分别表示搜索凭据、裁判服务地址、裁判密钥、裁判模型名。服务器已有配置时不需要重新创建。多个任务共享搜索密钥状态时，不要手动重置状态文件。

评测需要完整固定题包，例如`outputs/server_eval_778/`，包含`tasks.jsonl`、`references.jsonl`、`manifest.json`及图片。`tasks.jsonl`供模型作答，`references.jsonl`仅用于评分；不能把参考答案拼进模型问题中。重新从网上下载同名数据集，不能保证与既有778题完全一致。

模型目录需包含权重、配置、分词器和图片处理文件。训练用数据也需单独准备；仅下载代码不会得到训练题目和图片。

## 5. 在原集群运行GRPO训练

这部分**只适用于已经部署好的原服务器**。脚本读取`/cluster/home/shenqunlun/zhousheng`下的现有模型、环境和数据。`GRPO_STAGE=sft`表示从已有SFT模型继续做GRPO，不是重新进行SFT训练。

先登录并检查已有任务：

```bash
ssh shenqunlun@10.50.27.186
cd /cluster/home/shenqunlun/zhousheng
squeue -u "$USER"
```

第一条连接服务器，密码按终端提示输入；第二条进入个人工作目录；第三条查看当前账号的任务。账号可能被多人使用，只能操作自己确认归属的任务。已有同一训练任务时不要重复提交。

确认部署检查通过、没有同一任务正在运行后，以下两条**选一条**执行：

```bash
sbatch --nodes=1 --job-name=zs-base-grpo --export=ALL,GRPO_STAGE=base visual-search-agent/cluster/train.sbatch
```

从原始Base模型做GRPO。`--nodes=1`明确使用一台机器，覆盖脚本默认的两台；脚本每台申请8张卡。`--job-name`设置任务显示名称；`--export`传入训练分支。提交成功会返回任务号，但排队不等于已经开始训练。

```bash
sbatch --nodes=1 --job-name=zs-sft-grpo --export=ALL,GRPO_STAGE=sft visual-search-agent/cluster/train.sbatch
```

从已有SFT模型做GRPO，同样申请单机8卡。脚本先跑3批短程检查，再运行正式一轮。当前固定数据下正式目标为165批，正式每批48题、每题4份回答。不同数据规模不能直接沿用165这一数字。

脚本检查`DEPLOYMENT_READY`；没有它时应补齐部署检查，不能随便创建空文件跳过。SFT模型兼容转换由`prepare_sft_compat.py`处理。已有`SUCCESS.json`的对应运行会被跳过；自动恢复依赖实际完整存档，未保存批次可能需要重算。

```bash
squeue -j 4410713
tail -n 50 logs/train-4410713.log
```

把示例任务号换成提交时返回的号码。第一条看运行或排队状态；第二条看最近50行日志。任务被锁住、失败或重新排队时先查原因，不要重复提交。

训练结果在`results/base_grpo/`或`results/sft_grpo/`。`checkpoints/global_step_数字/`保存训练进度、模型和优化器等；`actor/huggingface/`是用于推理的模型；`SUCCESS.json`是脚本在检查最终模型后写入的完成记录。目录存在并不等于训练完整成功，应同时核对任务退出状态、最终批次和模型文件。

## 6. 让已有模型回答评测题

以下从仓库根目录运行，在独立评测环境、且已分配GPU的机器上执行。先设置路径，替换示例中的模型位置：

```bash
MODEL_PATH="/你的模型目录"
BUNDLE_PATH="$PWD/outputs/server_eval_778"
```

`MODEL_PATH`指向要测的实际模型，例如Base或GRPO后的模型；`BUNDLE_PATH`指向固定题包。变量只对当前终端生效，换终端需要重新设置。

先单卡测试两题：

```bash
python -m evaluation.launch \
  --model "$MODEL_PATH" --tasks "$BUNDLE_PATH/tasks.jsonl" \
  --output outputs/model-smoke --expected-count 778 --limit 2 \
  --gpus 0 --mode agent --protocol training --dtype bfloat16 \
  --max-steps 10 --max-new-tokens 4096 --context-tokens 32768 \
  --max-pixels 262144 --image-side 1536 --no-tool-summaries --env-file .env
```

- 反斜杠`\`表示命令换行后继续，不要在它后面加文字。
- `--model`是模型路径，`--tasks`是题目文件，`--output`是新结果目录。
- `--expected-count 778`检查完整题包规模；`--limit 2`只取前两题检查流程，不是正式成绩。
- `--gpus 0`使用当前可见的第一张GPU；集群上只填写分配给本任务的卡。
- `--mode agent`允许使用工具；`--protocol training`选择训练风格提示词，**不会训练模型**。
- `--dtype bfloat16`选择数字计算格式，适用于已验证的A100配置；V100通常需要`float16`，且显存配置要另行实测。
- `--max-steps 10`限制作答步骤；`--max-new-tokens 4096`限制每次生成长度；`--context-tokens 32768`限制上下文容量。
- `--max-pixels`与`--image-side`控制图片处理预算；`--no-tool-summaries`关闭工具结果摘要；`--env-file`指定配置。

确认模型加载、工具请求、结果文件和评分都正常后，8卡正式作答：

```bash
python -m evaluation.launch \
  --model "$MODEL_PATH" --tasks "$BUNDLE_PATH/tasks.jsonl" \
  --output outputs/model-778 --expected-count 778 \
  --gpus 0,1,2,3,4,5,6,7 --mode agent --protocol training --dtype bfloat16 \
  --max-steps 10 --max-new-tokens 4096 --context-tokens 32768 \
  --max-pixels 262144 --image-side 1536 --no-tool-summaries --env-file .env
```

与小测相比，去掉`--limit 2`、使用8张卡并改用新的输出目录。程序把题目分给各卡，每张卡分别加载模型，因此不能把8张卡的显存简单当成一张大卡使用。结果合并到`outputs/model-778/merged/`，其中`predictions.jsonl`是答案，`run.json`记录运行配置。不同模型、参数实验请使用不同目录。

原服务器的`evaluation/server_benchmark*.sh`还支持固定778题、模型来源与文件校验，以及跨节点运行，但绑定原目录和共享配置；普通新机器优先用上面的`evaluation.launch`入口。独立评测无需复用训练密钥池；若显式使用`VS_SERPER_BACKEND=training`，必须同时准备训练代码路径、完整密钥配置和共享状态路径。

## 7. 给答案打分

裁判评分不加载被测模型，只调用评分服务。先在`.env`中配置自己的`JUDGE_API_KEY`。以下使用本项目曾使用的服务和模型名称，服务可用性及额度需自行确认：

```bash
python -m evaluation.score \
  --run outputs/model-778/merged \
  --tasks "$BUNDLE_PATH/tasks.jsonl" --references "$BUNDLE_PATH/references.jsonl" \
  --output outputs/model-778-score \
  --judge-model deepseek-flash --base-url https://api.deepseek.com/v1 \
  --key-env JUDGE_API_KEY --env-file .env \
  --judge-thinking provider-default --judge-max-tokens 4096 --workers 3
```

- `--run`必须指向包含`run.json`与答案的`merged`目录。
- `--references`提供独立参考答案，仅送给裁判。
- `--output`保存评分文件；改变裁判设置时使用新目录。
- `--judge-model`和`--base-url`指定裁判模型与服务地址；`--key-env`只写密钥变量名，不直接写密钥。
- `--judge-thinking provider-default`沿用服务默认思考设置；`--judge-max-tokens 4096`给裁判足够空间写出结论。
- `--workers 3`表示最多同时处理3个评分请求。

小测评分时，把`--run`换成`outputs/model-smoke/merged`，`--output`换成`outputs/model-smoke-score`，其余设置保持一致。评分会把题目、模型答案、参考答案发送给所选外部服务，并消耗该账号额度。

`scores.jsonl`记录逐题评分，`summary.json`记录完整汇总；出现`judge_errors.json`表示部分评分没有成功。裁判必须完整返回布尔值`correct`和文字`reason`。服务超时、截断和格式错误属于评分失败，不能当成答错。相同设置重跑可复用已保存评分。

本项目曾有7题因512的裁判输出额度被思考耗尽，后将这7题提高到4096完成补评。历史结果110/778（14.14%）属于“首次512、失败题4096补评”，不是所有历史请求都使用4096。补评汇总在原服务器`benchmark-vs778/outputs/judge7-length4096/summary.json`。

评分主要判断答案含义，允许等价名称、单位及无害排版差异；没有可提取最终答案会按未完成回答处理。作答格式检查与裁判返回格式检查是两件事。裁判可能判断不一致，重要结论应抽样复核。

**统一参考答案裁判分数不是各公开benchmark的官方成绩。** Base与GRPO模型比较时，应保持题目、图片、提示词、工具、步骤预算、生成参数、答案提取和裁判设置一致，并保留`run.json`与评分配置。

## 8. 最常见的使用顺序

1. 只想测试已有模型：准备模型和固定题包 → 独立评测环境 → 两题作答和评分 → 全量作答和评分。
2. 想继续GRPO：先确认训练环境、数据和完整存档 → 原集群单机8卡提交 → 短程检查 → 正式训练 → 核对最终保存。
3. 想制作SFT数据：查看`sft_processing/`各工具的`--help`和配置模板；这与GRPO启动入口不同。
4. 想比较训练前后：使用相同评测命令，只更换模型路径及输出目录；不能只看训练批次正确率判断能力变化。

```bash
python -m evaluation.launch --help
python -m evaluation.score --help
python -m evaluation.package --help
```

分别查看作答、评分、固定题包打包工具的全部参数。最后一个工具用于已有题单和图片的打包与校验，不会自动替你获得相同的778题数据。

## 9. 代码更新与安全保存

不要提交`.env`、`local_training.env`、模型权重、完整存档、含密钥的日志或未脱敏结果。第三方源码保留原许可证。旧Git提交历史可能仍保留历史配置，移除当前文件不会删除历史中的内容。

修改后先检查内容，再提交：

```bash
git status --short
git diff
```

第一条列出变化的文件，第二条查看具体修改；确认只包含希望上传的代码和说明后再选择文件提交。不要直接把所有运行产物加入Git。
