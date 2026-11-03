#!/usr/bin/env python3
"""Portable launcher: the separately distributed dataset owns configs/full.json."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"路径越出数据目录: {name}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify", "prepare", "smoke", "train"])
    parser.add_argument("--data-dir", default=os.getenv("OPENSEARCH_DATA_DIR"))
    parser.add_argument("--model", default=os.getenv("OPENSEARCH_MODEL_DIR"))
    parser.add_argument("--output-root", default=os.getenv("OPENSEARCH_OUTPUT_ROOT"))
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload ZeRO optimizer states/computation to CPU memory")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.data_dir:
        data = Path(args.data_dir).expanduser().resolve()
    else:
        data = repo.parent / "OpenSearch-VL-SFT-Agent-NoImages"
    if not (data / "configs/full.json").is_file():
        parser.error(f"找不到 {data}/configs/full.json；请用 --data-dir 指定数据目录。")
    original = read(data / "configs/full.json")
    if original.get("finetuning_type") != "full":
        raise ValueError("configs/full.json 不是全参数训练配置。")
    manifest = read(data / "configs/image_paths.json")["images"]
    audit = read(data / "configs/image_copy_verification.json")
    registry = read(data / "dataset_info.json")
    selected, counts, hashes, referenced = {}, {}, {}, set()
    for split, limit in [("train", 256), ("val", 32)]:
        path = inside(data, registry[split]["file_name"])
        hashes[split] = digest(path)
        expected = audit["dataset_sha256"][registry[split]["file_name"]]
        if hashes[split] != expected:
            raise ValueError(f"{path} 校验失败，内容与图片复制核验记录不一致。")
        rng = random.Random(original.get("seed", 42))
        reservoir = []
        count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                referenced.update(row.get("images", []))
                count += 1
                if len(reservoir) < limit:
                    reservoir.append(row)
                else:
                    index = rng.randrange(count)
                    if index < limit:
                        reservoir[index] = row
        selected[split], counts[split] = reservoir, count
    if referenced != set(manifest):
        raise ValueError("训练/验证数据引用的图片集合与清单不一致。")
    print(f"数据: {counts}；核验 {len(referenced)} 张图片…", flush=True)
    for i, name in enumerate(sorted(referenced), 1):
        if digest(inside(data, name)) != manifest[name]["sha256"]:
            raise ValueError(f"图片校验失败: {name}")
        if i % 4000 == 0:
            print(f"已核验 {i}/{len(referenced)}", flush=True)
    print("数据和图片 SHA256 全部通过。", flush=True)
    if args.mode == "verify":
        return
    output_root = (Path(args.output_root).expanduser() if args.output_root
                   else repo.parent / "opensearch_training_outputs").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=args.mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_", dir=output_root))
    config = dict(original)
    config["overwrite_cache"] = True  # rebuild tokenized data after image-alignment fix
    config.update(dataset_dir=str(data), media_dir=str(inside(data, original.get("media_dir", "."))),
                  deepspeed=str(inside(data, original["deepspeed"])), output_dir=str(work / "model"))
    if args.model:
        candidate = Path(args.model).expanduser().resolve()
        if not (candidate / "config.json").is_file():
            raise ValueError(f"本地模型不存在: {candidate}")
        config["model_name_or_path"] = str(candidate)
    else:
        candidate = repo.parent / "models" / original["model_name_or_path"].split("/")[-1]
        if (candidate / "config.json").is_file():
            config["model_name_or_path"] = str(candidate.resolve())
    if args.mode in {"smoke", "prepare"}:
        small = work / "sample"
        small.mkdir()
        for split, rows in selected.items():
            registry[split]["file_name"] = split + ".jsonl"
            with (small / (split + ".jsonl")).open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        dump(small / "dataset_info.json", registry)
        config.update(dataset_dir=str(small), max_steps=10, save_strategy="no", save_only_model=True)
    ds_source = inside(data, original["deepspeed"])
    ds_config = read(ds_source)
    if args.cpu_offload:
        ds_config["zero_optimization"]["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    # Snapshot the DeepSpeed config too, so a later source edit cannot alter this run.
    dump(work / "deepspeed.json", ds_config)
    config["deepspeed"] = str(work / "deepspeed.json")
    overrides = {k: {"source": original.get(k), "runtime": v}
                 for k, v in config.items() if original.get(k) != v}
    runtime = work / "resolved.yaml"
    # JSON syntax is also valid YAML. The .yaml extension selects OmegaConf's
    # working YAML loader, avoiding this fork's broken json.load(Path(...)).
    dump(runtime, config)
    record = {"source_config_sha256": digest(data / "configs/full.json"), "dataset_sha256": hashes,
              "source_config": original, "overrides": overrides, "data_dir": str(data),
              "verified_images": len(referenced), "gpus": args.gpus, "python": sys.version,
              "cpu_offload": args.cpu_offload, "deepspeed_config": ds_config,
              "source_deepspeed_sha256": digest(ds_source)}
    versions = {}
    for name in ["torch", "transformers", "datasets", "accelerate", "deepspeed", "flash-attn", "qwen-vl-utils", "llamafactory"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    record["versions"] = versions
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    record["git_commit"] = git.stdout.strip() or "unavailable"
    dump(work / "run_record.json", record)
    print(f"本次配置和记录: {work}")
    print(f"输出磁盘剩余: {shutil.disk_usage(work).free / 1024**3:.1f} GiB")
    if args.mode == "prepare":
        print("仅准备测试配置，尚未开始训练。")
        return
    import torch
    import llamafactory
    framework = Path(llamafactory.__file__).resolve()
    if not framework.is_relative_to(repo / "SFT"):
        raise ValueError(f"加载的框架不是当前仓库: {framework}；请在 {repo}/SFT 执行 pip install -e .")
    if args.gpus < 1 or torch.cuda.device_count() < args.gpus:
        raise ValueError(f"需要 {args.gpus} 张可见 GPU，实际 {torch.cuda.device_count()}")
    env = os.environ.copy()
    env.update(USE_RAY="0", FORCE_TORCHRUN="1", NNODES="1", NODE_RANK="0", NPROC_PER_NODE=str(args.gpus))
    # Use the installed entry point: some forks do not implement python -m cli.
    cli = shutil.which("llamafactory-cli")
    if not cli:
        raise RuntimeError("找不到 llamafactory-cli，请先安装当前仓库 SFT。")
    command = [cli, "train", str(runtime)]
    print("开始训练；完整日志保存在 " + str(work / "train.log"), flush=True)
    with (work / "train.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo / "SFT", env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    record["exit_code"] = code
    dump(work / "run_record.json", record)
    if code:
        raise SystemExit(code)
    print(f"训练命令完成。模型目录: {work / 'model'}；请核对 train.log 和训练指标。")


if __name__ == "__main__":
    main()
