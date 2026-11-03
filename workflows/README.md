# 按工作阶段使用项目

所有命令从仓库根目录执行。这里是统一导航和调用入口；底层算法仍在原目录中，已有命令继续可用。

| 阶段 | 输入 | 输出 | 原代码位置 |
|---|---|---|---|
| 1. 采集轨迹 | 题目、图片、作答模型与搜索配置 | 包含搜索、工具调用和答案的记录 | `agent_pipeline.py`、`agent/` |
| 2. SFT数据处理 | 采集且审核后的记录 | 清洗、编号后的训练样本 | `sft_processing/` |
| 3. SFT训练 | 训练样本、初始模型、适配后的训练配置 | SFT模型 | `train/OpenSearch-VL-Training/SFT/` |
| 4. GRPO强化学习 | Base或SFT模型、训练题目与奖励规则 | GRPO后的模型及可恢复存档 | `cluster/`、`datasets/grpo/`、`train/.../RL/` |
| 5. 评测和评分 | 已有模型、固定题包、独立参考答案 | 作答结果与正确率 | `evaluation/` |

## 统一入口怎么用

```bash
python -m workflows --help
python -m workflows trajectory --help
python -m workflows sft --help
python -m workflows grpo --help
python -m workflows evaluate --help
```

第一条列出所有阶段，后四条分别列出阶段内的操作。选择操作后，加`--help`查看底层程序的真实参数，例如：

```bash
python -m workflows trajectory collect --help
python -m workflows sft clean --help
python -m workflows grpo control --help
python -m workflows evaluate run --help
```

分别查看采集、清洗、强化学习控制和评测的参数，不会开始任务。下面命令只打印将要运行的命令：

```bash
python -m workflows --dry-run evaluate run --model /模型目录 --tasks /题包/tasks.jsonl --output outputs/example
```

`--dry-run`放在阶段名称前；去掉它才真正执行。打印命令不检查模型、密钥或显存是否准备完成。

## 采集轨迹

顺序：`prepare`准备题目 → `split`划分题目 → `collect`采集 → `review`审核 → `export`导出。`stats`查看统计。

```bash
python -m workflows trajectory collect --input data/train_tasks.jsonl --output outputs/trajectories.jsonl --provider openrouter --model 你的模型名称 --max-steps 30
```

把输入路径与模型名称替换为真实值。`--output`保存轨迹；`--max-steps`限制作答步骤。需要事先配置搜索与模型服务，此命令会请求外部服务并产生费用。审核命令使用独立参考答案；导出不等于完成模型训练。

## SFT：数据处理和模型训练分开

统一入口的`sft`目前管理数据处理：`clean`清洗、`check`检查、`renumber`整理编号、`teacher`教师处理、`configure`生成配置。每个工具的输入要求不同，先查看对应`--help`，不要把不同版本数据路径混用。

实际SFT训练请阅读`train/OpenSearch-VL-Training/SFT/README.md`。其中配置原本面向多机大集群，不能直接当成本项目单机8卡配置。仓库尚无经过本项目验证的一键SFT训练入口，因此这里不提供会误启动大规模训练的统一快捷命令。已有SFT模型可以直接进入GRPO阶段。

## GRPO强化学习

`grpo control`连接现有训练控制工具；`grpo check-tools`检查训练工具。原服务器的单机8卡提交方式、模型路径、环境要求和日志查询见根目录README第5节。

`GRPO_STAGE=base`从Base开始GRPO；`GRPO_STAGE=sft`从已有SFT模型开始GRPO。后者不会重做SFT。不要直接绕过控制层运行`official_8gpu_grpo.sh`：它的默认参数与项目控制层覆盖后的参数不同。

## 评测与评分

`evaluate run`生成答案，`evaluate score`打分，`evaluate package`打包题目图片，`evaluate compare`比较结果。完整参数示例在根目录README第6、7节。

将原命令`python -m evaluation.launch ...`写成`python -m workflows evaluate run ...`即可；评分同理。新入口原样传递参数，不改变提示词、评分规则或生成设置。

## 底层目录为什么仍保留

`cluster/`负责原服务器资源提交，`datasets/grpo/integration/`负责工具与奖励连接，`train/`保留第三方训练框架。把它们直接搬进一个文件夹会影响导入、数据注册及服务器绝对路径，因此此次通过统一入口组织使用方式，不批量移动正在使用的底层源码。后续新功能应先归入以上阶段，避免继续增加根目录独立脚本。
