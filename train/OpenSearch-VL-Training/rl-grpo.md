# GRPO 依赖安装、测试与正式训练

本项目使用与你的 SFT 一致的六种工具：网页搜索、文字找图片、读取网页、查看图片、裁剪、计算。文字找图片不上传本地图片，不需要 COS、upload.py 或 env.py。

本说明用于本项目在 Linux、NVIDIA 显卡上的安装与启动。固定 Python 3.12、PyTorch 2.8 和 CUDA 12 配套版本；不是任意 Python、显卡都能直接使用同一个安装文件。

先准备完整项目代码、模型和训练数据，再按顺序安装。已有环境和已安装成功的包可以跳过；测试命令按显卡数量选择，不需要每一种都运行。

换服务器时，先把下面的项目地址改成实际地址。后续命令在同一个终端执行，复用这个地址：

```bash
export GRPO_PROJECT_ROOT='/root/autodl-tmp/visual-search-agent'
```

## 从当前服务器下载代码

使用服务器文件管理器下载以下四个目录，并在接收代码的机器上保持原来的目录关系：

- `/root/autodl-tmp/visual-search-agent/train/OpenSearch-VL-Training/`：训练框架和安装说明。
- `/root/autodl-tmp/visual-search-agent/datasets/grpo/scripts/`：数据准备、环境检查和训练启动脚本。
- `/root/autodl-tmp/visual-search-agent/datasets/grpo/integration/`：训练接入代码、六种工具实现和服务配置模板。
- `/root/autodl-tmp/visual-search-agent/datasets/grpo/tests/`：工具检查代码。

下载后应保持下面的结构，其中最外层地址就是前面设置的 `GRPO_PROJECT_ROOT`：

```text
visual-search-agent/
├── train/
│   └── OpenSearch-VL-Training/
└── datasets/
    └── grpo/
        ├── scripts/
        ├── integration/
        └── tests/
```

服务器代码已经包含本次修复。保留上述目录结构后，直接按下面步骤安装环境、配置服务和训练即可。

训练题目、图片和 `audit/` 数据核对文件需另行准备在 `datasets/grpo/` 下；模型需另行准备。实际服务密钥在第7步配置。只下载代码不会自动带上这些训练材料。

## 1. 进入 Conda 环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh

# 仅新服务器第一次创建环境时执行这一行。
# 已有 search-grpo 环境时跳过，不要重复创建。
conda create -n search-grpo python=3.12 -y

conda activate search-grpo
which python
cd "$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training"
mkdir -p "$GRPO_PROJECT_ROOT/datasets/grpo/.codex"
```

`which python` 应显示 `/root/miniconda3/envs/search-grpo/bin/python`；后续安装命令都在这个环境中执行。

## 2. 安装基础工具、PyTorch 和项目依赖

```bash
python -m pip install \
  setuptools==79.0.1 \
  wheel==0.45.1 \
  pybind11==2.13.6 \
  ninja==1.13.2 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple


python -m pip install --no-cache-dir \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install --no-cache-dir --no-build-isolation \
  -e "$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/rllm" \
  -e "$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/rllm/verl" \
  -e "$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/Megatron-LM" \
  -e "$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/mbridge" \
  "torch==2.8.0+cu128" \
  "torchvision==0.23.0+cu128" \
  "torchaudio==2.8.0+cu128" \
  "numpy==1.26.4" \
  "pyarrow==19.0.1" \
  "cachetools==5.5.2" \
  "sglang==0.5.5" \
  "transformers==4.57.1" \
  "ray[default]==2.41.0" \
  "requests==2.32.3" \
  python-dotenv \
  qwen-vl-utils \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/ \
  --report "$GRPO_PROJECT_ROOT/datasets/grpo/.codex/install-training-dependencies.json"
```

`-e` 使用指定目录的项目代码；固定版本避免安装时自动换掉已配套的组件。

`ninja==1.13.2` 用于生成程序首次运行时编译小组件，来源为上述 PyPI 镜像。当前迁移环境已安装，无需重装。可用 `python -m ninja --version` 检查版本。启动脚本会把当前 Conda 环境的命令目录放到前面，确保子进程能找到它。

## 3. 安装 Transformer Engine

```bash
MAX_JOBS=2 python -m pip install --no-cache-dir --no-build-isolation \
  "transformer-engine[pytorch]==2.6.0" \
  "torch==2.8.0+cu128" \
  "numpy==1.26.4" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --report "$GRPO_PROJECT_ROOT/datasets/grpo/.codex/install-transformer-engine.json"
```

`MAX_JOBS=2` 限制安装时同时编译的任务数，减少内存占用。

## 4. 下载并安装 FlashAttention 2.8.1

先检查安装文件是否匹配。下列检查报错时停止，不继续安装。

```bash
python - <<'PY'
import platform
import sys
import torch
print('系统：', platform.system(), platform.machine())
print('Python：', sys.version.split()[0])
print('PyTorch：', torch.__version__)
print('CUDA：', torch.version.cuda)
print('编译设置：', torch._C._GLIBCXX_USE_CXX11_ABI)
assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
assert sys.version_info[:2] == (3, 12)
assert torch.__version__.split('+')[0].startswith('2.8.')
assert torch.version.cuda and torch.version.cuda.startswith('12.')
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
PY
```

下载并核对官方文件。核对显示 `OK` 后执行安装。

```bash
curl --fail --location --retry 3 \
  --header 'Accept: application/octet-stream' \
  "https://api.github.com/repos/Dao-AILab/flash-attention/releases/assets/309735649?nocache=$(date +%s)" \
  --output '/tmp/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'

echo '55331d797171973c8babf2b5e5fce5f78859b1cd298d54a136e8994147fe9e95  /tmp/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' | sha256sum --check
```

```bash
python -m pip install --no-deps --no-cache-dir \
  '/tmp/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'

python -c "import flash_attn; print(flash_attn.__version__)"
```

`--no-deps` 只安装这个包，不自动更换 PyTorch；`--no-cache-dir` 不额外保留安装缓存。版本检查应显示 `2.8.1`。

## 5. 处理 CUDA 文件路径和 Blackwell 训练冲突

启动脚本已自动处理以下设置；手动启动其他训练入口时，在同一个终端执行：

```bash
# 让新进程找到当前 Conda 环境里的 CUDA 文件。
export LD_LIBRARY_PATH="$(python - <<'PY'
import os
import sysconfig
from pathlib import Path
site = Path(sysconfig.get_paths()['purelib'])
paths = [str(p) for p in sorted((site / 'nvidia').glob('*/lib')) if p.is_dir()]
if (site / 'torch/lib').is_dir():
    paths.append(str(site / 'torch/lib'))
paths.extend(p for p in os.environ.get('LD_LIBRARY_PATH', '').split(':') if p)
print(':'.join(dict.fromkeys(paths)))
PY
)"

# 仅对当前 RTX PRO 6000 Blackwell 这类显卡关闭冲突的计算方式。
# H800 不会执行这里的 export。
if python - <<'PY'
import torch
supported = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 12
raise SystemExit(0 if supported else 1)
PY
then
    export NVTE_FUSED_ATTN=0
fi
```

## 6. 检查代码和内部服务

下面三条命令分别检查训练批数与保存安排、显卡计算方式选择、内部服务启动。只检查，不修改代码，也不开始训练。

```bash
cd "$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/rllm"
python tests/test_training_completion.py
python tests/test_sglang_backend_selection.py
python tests/test_http_server_startup.py
```

继续检查本项目的六种工具、裁剪坐标、图片传给下一轮输入、失败处理。这些检查不访问外网、不使用显卡：

```bash
cd "$GRPO_PROJECT_ROOT/datasets/grpo"
python tests/test_project_tools.py
python tests/test_project_workflow.py
python scripts/check_project_tools.py
```

每次读取网页最多返回4000个字符，可以使用 `vs_visit` 的 `offset` 继续读取。最后一次操作会提示模型作答。相同参数的成功工具调用不会再次访问服务，会提示模型换一种操作或作答；失败请求仍允许重试。

工具代码随 `integration/project_runtime/` 一起下载，使用第2步已有的 `requests` 和 Pillow；无需增加安装命令或更换版本。`tests/` 也需要从当前服务器一并下载。


## 7. 设置模型地址、配置搜索与评分服务

```bash
cd "$GRPO_PROJECT_ROOT/datasets/grpo"

# 首次使用或迁移后执行，按提示填写 rllm 和模型的实际地址。
python scripts/training_control.py setup
```

训练代码填写 `$GRPO_PROJECT_ROOT/train/OpenSearch-VL-Training/RL/rllm` 对应的实际完整地址，模型填写已经准备好的 SFT 模型文件夹地址。提示框里填写实际地址，不要原样输入 `$GRPO_PROJECT_ROOT` 文字。

随后编辑 `$GRPO_PROJECT_ROOT/datasets/grpo/integration/local_training.env` 对应文件：

- 搜索：填写 `SERPER_API_KEYS`（JSON 数组）或单个 `SERP_API_KEY` / `SERPER_API_KEY`。网页搜索和文字找图片共用这些密钥。
- 评分：填写 `JUDGE_BASE_URL`、`JUDGE_API_KEY`、`JUDGE_MODEL`。按原服务需要填写 `JUDGE_THINKING`。
- 网页由服务器直接打开；无需填写图片上传配置。网页直接读取，不需要额外的网页读取服务密钥。

```bash
# 检查依赖、模型地址、显卡及服务是否已配置，不开始训练。
python scripts/training_control.py check
```
## 8. 检查实际搜索和图片返回

```bash
cd "$GRPO_PROJECT_ROOT/datasets/grpo"
python scripts/check_project_tools.py --live
```

这条命令实际调用一次网页搜索和一次图片搜索，会消耗搜索额度；随后打开搜索得到的网页、下载搜索得到的图片，并检查图片是否放进下一轮模型输入。也会检查本地看图、裁剪和计算。它不加载模型、不进行模型更新。出现 `passed` 才表示对应检查通过。

检查记录保存在 `datasets/grpo/.codex/project_tools_live_check.json`。实际启动训练前，启动脚本也会执行这项检查，失败时停止启动。

`integration/` 中已保存项目接入代码。`python scripts/build_integration.py` 现在只检查这些代码，不再用旧官方副本覆盖它们。

## 8.1 H20 单卡三批检查

当前单卡检查使用一张 H20：每批一道题、每题生成两份回答、检查三批，不保存模型。训练更新记录的75%放在普通内存、25%放在显卡；梯度留在显卡，避免普通内存同时存两份较大的记录；更新时减少同时复制与计算造成的峰值，并关闭更新记录的大块锁页内存预留。生成程序使用约四分之一显存，上下文上限为8192。底层代码同时修复了忽略该设置、以及模型回到显卡后仍保留普通内存副本的问题。此设置已在当前 H20 实例完成三批实际训练和每批验证，程序正常结束；本次不保存模型。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate search-grpo
cd "$GRPO_PROJECT_ROOT/datasets/grpo"
# 小张量检查：不加载完整模型，检查内存搬运和参数更新。
python tests/test_h20_memory_helpers.py

python scripts/training_control.py debug1 --gpus 0 --no-save
```

第一行加载 Conda，第二行进入已有训练环境，第三行进入实际启动目录，最后一行只用第0张显卡检查三批。结束后不保存模型，不能从本次检查结果继续训练。启动前自动检查真实搜索、网页读取和图片返回。代码与配置调整没有新增或更换依赖。

## 9. 双卡测试

常规双卡测试，包含模型保存：

```bash
cd "$GRPO_PROJECT_ROOT/datasets/grpo"
python scripts/training_control.py debug2 --gpus 0,1
```

如果暂时只检查作答、评分和更新，不保存模型，改用下面这一条，不要和上一条同时运行：

```bash
python scripts/training_control.py debug2 --gpus 0,1 --no-save
```

`--no-save` 只适合临时检查，结束后不能从本轮结果继续训练。正常保存需要足够磁盘空间；只训练一批也要保存完整模型及更新记录。

## 10. 八卡小规模测试

有八张可用显卡时执行，先做三批检查并保存模型：

```bash
cd "$GRPO_PROJECT_ROOT/datasets/grpo"
python scripts/training_control.py smoke8 --gpus 0,1,2,3,4,5,6,7
```

## 11. 八卡正式训练

小规模测试通过，模型保存空间也足够后执行：

```bash
cd "$GRPO_PROJECT_ROOT/datasets/grpo"
python scripts/training_control.py train8 --gpus 0,1,2,3,4,5,6,7
```

`train8` 使用正式题单训练一轮，默认每50批验证、保存，正常结束时补存最终模型。正式训练不使用 `--no-save`。新运行默认从设置的SFT模型开始，不会自动接着上次保存记录继续。

日志中的 `tools/calls` 表示工具调用次数，`tools/errors` 表示失败次数，`tools/images_returned` 表示返回给模型的图片数。它们用于确认模型有没有实际使用工具。

以上命令启动时会显示日志地址；输出默认放在 `$GRPO_PROJECT_ROOT/datasets/grpo/runs/` 下各自独立的目录。

## 12. 启动前只看命令或检查配置

在实际启动前，可以先检查八卡命令。下面两条都不开始训练：

```bash
# 只显示将要使用的参数。
python scripts/training_control.py train8 --gpus 0,1,2,3,4,5,6,7 --dry-run

# 进一步让训练框架读取并检查完整配置。
python scripts/training_control.py train8 --gpus 0,1,2,3,4,5,6,7 --config-only
```
