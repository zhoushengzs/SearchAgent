# InternVL3-14B 原始 SFT 数据试跑包（2026-11-04）

这是本次新准备的候选适配，不是已经在 GPU 上训练成功的配方。已完成真实 tokenizer/processor、8条真实样本的多图/工具历史/最后一轮学习目标检查，以及不加载权重的 LoRA 目标层检查。未做真实 14B 权重前向、反向、保存、恢复或完整训练。不要直接启动全量训练。

## 固定来源

- 完整代码：本仓库 `train/OpenSearch-VL-Training/SFT/`，本次从服务器保存的完整源码恢复18个遗漏文件；其余已有源码与服务器备份逐字一致。保留服务器的Qwen图片截断对齐补丁，该补丁只针对Qwen模板，不套用InternVL。实际使用的Git提交会写入工作目录 `versions.json`。
- 模型：https://huggingface.co/OpenGVLab/InternVL3-14B-hf ，revision `e22931943e5336f85e06f4e2b38f3e5e6ee4de3b`。
- 数据：https://huggingface.co/datasets/zhoushengzs/OpenSearch-VL-SFT-Original ，revision `29211998029f2ab3481604bd3bf30d60dd99a51d`。
- 模板：本包 `protocol.py` 注册的 `intern_vl_agent`。保留 InternVL 的图像处理，工具说明用 `<tool_call>`，工具返回用 `<tool_response>` 包装。`qwen` 在此仅是工具文字格式的名字，不代表加载 Qwen3-VL 模型。

## 文件用途

- `protocol.py`：训练与重载检查共用的对话/工具格式。
- `train.py`：训练入口，结束时保存每张卡的峰值显存和训练日志。
- `download.py`：按固定版本下载完整模型和数据；模型约30GB，数据约8.17GB，请预留额外训练输出空间。
- `check_cpu.py`：用真实图像和文字处理器选择8条长度不超过8192的短样本，并验证最后回答的学习目标；仅用于试跑，不能用于评价效果。
- `make_config.py`：生成绝对路径的试跑参数及 DeepSpeed ZeRO-2 配置。
- `run_smoke.sh`：先检查数据，再启动单机8卡3步 LoRA 训练。
- `reload_check.py`：训练后检查 LoRA 参数有更新，重新加载并生成一条输出；不会把最后的参考回答喂给模型，也不代表回答正确率验证。
- `configs/smoke.json`、`configs/zero2.json`：可直接查看的试跑参数；`${SFT_WORK}` 由make_config.py替换为你的绝对路径。
- `constraints.txt`：候选环境的主要版本约束，不是经过完整GPU实测的全部依赖锁文件。
- `Dockerfile`：可选候选镜像，基础镜像标签和摘要已查到，但本次没有构建/运行。
- 已完成检查：恢复后的210个源码文件与服务器备份逐字一致；8条真实样本通过CPU处理；训练入口帮助命令可运行；配置生成支持带空格的路径。CPU检查使用临时隔离环境 Transformers 4.57.1 / Torch 2.14.0，不代表下列 Torch 2.8.0 CUDA 环境已实测。

## 建议目录

```text
~/SearchAgent/                  # 从GitHub下载本项目
  training_examples/internvl3_14b/ # 本试跑入口
  train/OpenSearch-VL-Training/SFT/ # 完整训练源码
~/internvl3-sft-work/
  model/                        # InternVL3-14B-hf 全部文件
  data/                         # HF 原始数据及图片
    train.jsonl
    val.jsonl
    dataset_info.json
    data/images/...
    configs/...
  prepared/                     # 8条试跑样本及检查结果
  smoke.json                    # 本次候选参数
  zero2.json                    # 显存分摊设置
  outputs/smoke-lora/            # 训练输出
  reload-check.json             # 重载检查结果
```

## 首选：独立 Conda 环境

以下命令在对方 Linux GPU 服务器上执行。先下载项目：

```bash
cd "$HOME"
git clone https://github.com/zhoushengzs/SearchAgent.git
```

`cd`进入个人目录，`git clone`创建SearchAgent文件夹并下载主分支。如果已有仓库，先在该仓库运行 `git status --short` 检查自己的未提交修改，再运行 `git pull --ff-only origin main` 更新，不要重复克隆或强制覆盖。

需要8张已分配给自己的卡；示例以80GB卡作为目标，不保证其他卡或32000长度可用。不要在别人的环境中直接升级依赖。

```bash
conda create -n internvl3-sft python=3.11.13 -y
conda activate internvl3-sft
```
第一条创建名为 internvl3-sft 的独立环境，固定 Python 版本；`-y` 表示接受安装。第二条切换到该环境。

```bash
export SFT_WORK="$HOME/internvl3-sft-work"
mkdir -p "$SFT_WORK"
cd "$HOME/SearchAgent/training_examples/internvl3_14b"
```
`SFT_WORK` 是这次训练的工作目录；`mkdir -p` 创建目录；`cd` 进入本仓库的试跑目录。此后下载和输出均使用这个绝对路径，不依赖从数据目录启动。

```bash
test -f "$HOME/SearchAgent/train/OpenSearch-VL-Training/SFT/src/llamafactory/data/__init__.py"
git -C "$HOME/SearchAgent" rev-parse HEAD
```

第一条检查恢复的数据处理模块是否存在；没有输出且退出正常表示存在。第二条打印本次使用的代码版本号，交接时应保存它。以下命令直接安装本仓库源码，不需要再下载另一份OpenSearch-VL。

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 python -m pip install -c constraints.txt -e "$HOME/SearchAgent/train/OpenSearch-VL-Training/SFT" deepspeed==0.18.8
python -m pip check
python -m pip freeze > "$SFT_WORK/installed-freeze.txt"
```
第一条安装 CUDA 12.8 对应的 PyTorch 软件；服务器驱动需要支持它。第二条安装训练程序及依赖，`-c` 限制主要包版本，`-e` 表示直接使用指定源码目录，`DS_BUILD_OPS=0` 避免安装时预编译全部 DeepSpeed 扩展。第三条检查包之间是否有冲突；必须无冲突后继续。第四条记录这台机器最终实际安装的所有 Python 包版本。`>` 将输出写入文件。

主要候选版本：Python3.11.13 / PyTorch2.8.0+cu128 / torchvision0.23.0 / Transformers4.57.1 / DeepSpeed0.18.8 / Accelerate1.11.0 / PEFT0.18.1 / Datasets4.0.0 / TRL0.24.0。此组合参照已有Qwen服务器环境和代码依赖范围制定，不称为InternVL训练验证环境。

```bash
python download.py
```
按上述固定版本下载模型和数据，保持原始JSONL和图片目录。若仓库访问要求登录，先执行 `hf auth login` 并在提示时输入自己的HF访问凭据；不要把凭据写在命令或发给别人。

```bash
bash run_smoke.sh
```
执行3步试跑。脚本先完成8条短样本处理检查，再由 `torchrun` 启动8个训练进程、每张卡一个。不使用 Ray，不修改原始HF配置。若输出目录已经存在，脚本停止，避免覆盖已有训练。

```bash
python reload_check.py
```
在当前作业可见的第一张显卡上检查刚保存的结果；保留调度器设置的显卡范围，不要自行改写 CUDA_VISIBLE_DEVICES。脚本检查LoRA参数已更新，然后加载基础模型与LoRA并生成。输出保存在工作目录的 `reload-check.json`。

注意：仅安装环境成功、或3步命令退出，都不能宣称完整训练成功。应同时检查 loss/梯度不是NaN或Inf、存档存在、每卡 memory-rank 文件、重新加载生成通过。然后才设计100步小实验和正式训练。

## 可选：Docker

有 Conda 可用就不必另外使用 Docker。以下基础镜像存在：
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel@sha256:a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6`。
这只是固定基础环境，Dockerfile尚未实测；集群不允许Docker时遵守集群的容器方式。

```bash
cd "$HOME/SearchAgent/training_examples/internvl3_14b"
docker build -f Dockerfile -t internvl3-sft-candidate ../..
docker run --gpus all --shm-size=16g -it --rm -v "$HOME/internvl3-sft-work:/work" internvl3-sft-candidate
```
第一条进入本包；第二条根据Dockerfile创建候选环境；第三条打开容器并把工作目录接入 `/work`。`--gpus all` 只应在专属GPU机器或正确配置的调度作业内使用，不能借此占用别人的卡。`--rm` 退出后删除容器本身，但接入的工作目录保留。

进入容器后执行：
```bash
python download.py
bash run_smoke.sh
python reload_check.py
```
用途与Conda路线相同。

## 试跑参数和下一阶段

- LoRA rank16/alpha32/dropout0.05，语言部分训练，视觉塔和连接层冻结。
- 每卡1条，8卡，累计1次，因此每次更新8条；3步只检查流程。学习率5e-5是候选起点，不是已验证最优值。
- 长度8192，只筛选不会被截断的8条短样本；原数据没有改动。正式训练必须重新统计完整数据长度，不能直接把所有长历史截断至8192。
- image_max_pixels262144/min1024/crop_to_patches=false，处理器最终每图单块448×448，256个图像位置。单块会损失细节，这是降低试跑开销的选择。正式是否多块，要另测显存和小字/细节任务效果；本版多块代码固定最多12块，不能照搬Qwen的像素设置并认为信息量相同。
- ZeRO-2只用于此LoRA试跑。全参数训练的优化器状态更大，应单独用ZeRO-3实测，不能把本配置直接当full配置。
- 后续先比较 reviewed 9604条与全部52446条，在固定独立任务集上比较回答正确率和工具成功率。保持计算预算可比；不能只比较各训1epoch，因为更新次数不同。
- 原JSONL的reviewed标签表示审核来源，不保证所有历史轮次正确；mask_history仅监督最后一轮。全量中未独立审核的41165条占78.49%，不可默认全对。

## 已确认的旧包问题

SearchAgent提交98fa1d4d19f167d8dd2b80e8cbe85a406ecc47bf的 `.gitignore` 第5行 `data/` 会连同源码中的 data 目录一起忽略。修复应恢复完整源码并将规则限定到真正的数据目录；不能只新建空目录，也不能把所有真实训练图片提交到GitHub。本次已恢复18个遗漏源码文件，并把根目录忽略规则改成 `/data/`，避免再次误排除源码中的data目录；没有将真实训练图片加入GitHub。

HF原 `export_notes.json` 中“不含图片”是早期导出的旧说明。当前版本包含16728张图，引用路径全部存在。训练集12011个任务，验证集1295个任务，按64位任务ID无交集；图片路径与文件SHA256也无交集。这不等于完成题目语义近重复或图片近重复检查。

实际数据角色：human/gpt/observation，没有function_call记录；工具动作以gpt文本中的 `<tool_call>` 表达。dataset_info的function_call映射只是预留支持。默认intern_vl追加Action/Action Input说明，与数据自己的system和目标文本冲突；不能只换模型名和template就当适配完成。

## 推理与评测边界

默认官方HF chat_template没有把传入的tools自动变为这套训练工具说明。生产推理必须复用 `protocol.py` 或生成完全一致的模板，保持工具返回包装、图像处理、停止符和回答解析一致。本包 `reload_check.py` 演示了相同训练格式的单次生成，但不含完整搜索工具执行循环，也不是778题评测入口。

原生 InternVL3-14B 的 `.chat()` 路线与 -hf 的 AutoProcessor/AutoModel 路线不可混用。已有原生InternVL基础模型评测记录不证明本次-hf SFT/LoRA已训练成功。对全量评测的模型加载、LoRA加载/合并、提示词、图像、停止符和工具循环适配仍应先用2题验证，然后再跑778题。
