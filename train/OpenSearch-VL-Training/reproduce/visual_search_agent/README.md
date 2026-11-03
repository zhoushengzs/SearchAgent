# 全参数 SFT 训练
## 安装依赖

先在训练使用的 Python 环境中安装依赖：

```bash
cd code/SFT
pip install -e ".[torch,metrics,deepspeed,ray]"
pip install qwen-vl-utils pillow av decord torchvision flash-attn
```

`code` 替换为实际下载的仓库目录。例如：

```bash
cd /root/test/OpenSearch-VL-Training/SFT
```

安装完成后，回到仓库根目录，再执行下面的训练命令：

```bash
cd ..
```


使用框架已安装的 Python 环境，在 OpenSearch-VL 仓库目录执行：


```bash
python reproduce/visual_search_agent/run.py train \
  --gpus 8 \
  --data-dir /path/to/OpenSearch-VL-SFT-Agent-NoImages \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --output-root /path/to/training_outputs
```

## 需要修改的参数

| 参数 | 修改内容 |
| --- | --- |
| `--gpus` | 同一台服务器实际使用的 GPU 数量；型号自动识别，无需填写 |
| `--data-dir` | 收到的数据文件夹路径，包含 `configs/full.json` |
| `--model` | 已下载的原始模型文件夹路径 |
| `--output-root` | 训练结果保存路径，需有足够空间保存模型和检查点 |

训练配置读取数据包中的 `configs/full.json`，按需修改：

| 配置项 | 当前值 | 含义 |
| --- | --- | --- |
| `per_device_train_batch_size` | 1 | 每张卡的批量 |
| `gradient_accumulation_steps` | 8 | 梯度累积次数 |
| `learning_rate` | 0.00002 | 学习率 |
| `num_train_epochs` | 2 | 训练轮数 |
| `cutoff_len` | 32000 | 文本截断长度 |
| `image_max_pixels` | 262144 | 图片像素上限 |
| `save_steps` | 200 | 每多少个更新步骤保存检查点 |
| `save_total_limit` | 3 | 最多保留的检查点数量 |

8 卡使用当前配置时，有效批量为 64；如需保持双卡测试时的有效批量 16，将梯度累积改为 2。

