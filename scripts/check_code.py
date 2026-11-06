#!/usr/bin/env python3
"""Run project static checks and offline tests without loading a model."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "train/OpenSearch-VL-Training"


def check_sources():
    expected_docs = {"README.md", "datasets/grpo/RECOIL.md"}
    actual_docs = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in (".md", ".rst")
        and not any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(ROOT).parts)
        and not any(part in ("outputs", "results", "runs", "registered", "registry") for part in path.relative_to(ROOT).parts)
        and path.relative_to(ROOT).parts[:3] != ("datasets", "grpo", "sources")
    }
    if actual_docs != expected_docs:
        raise ValueError(f"Keep only the root README and RECOIL document: {sorted(actual_docs)}")
    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_symlink() and not path.exists():
            raise ValueError(f"Broken compatibility link: {relative}")
        if relative.parts[0] == "train" or any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        if relative.parts[:2] == ("evaluation", "vendor"):
            continue
        if path.suffix == ".py" and path.is_file():
            ast.parse(path.read_text(), filename=str(relative))
            files.append(path)
        elif path.suffix in (".sh", ".sbatch") and path.is_file():
            subprocess.run(["bash", "-n", str(path)], check=True)

    for name in ("RL/rllm/rllm/data/dataset.py", "SFT/src/llamafactory/data/__init__.py"):
        if not (VENDOR / name).is_file():
            raise ValueError(f"Required upstream source missing: {name}")

    # These callable names come from configuration, so ordinary reference
    # counting cannot distinguish them from unused functions.
    sys.path.insert(0, str(ROOT))
    from sources.registry import DATASET_REGISTRY

    for config in DATASET_REGISTRY.values():
        loader = config.get("custom_loader")
        if loader:
            module_name, function_name = loader.split(":")
            spec = importlib.util.find_spec(module_name)
            if spec is None or spec.origin is None:
                raise ValueError(f"Missing loader module: {loader}")
            tree = ast.parse(Path(spec.origin).read_text())
            if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name for node in tree.body):
                raise ValueError(f"Missing configured loader: {loader}")
    print(f"Source syntax, shell syntax and configured loaders: OK ({len(files)} Python files)", flush=True)


def run(command, env):
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout + result.stderr, end="", flush=True)
        raise SystemExit(result.returncode)
    # Test fixtures print simulated model results; keep successful output short.
    output = (result.stderr or result.stdout).strip()
    print("\n".join(output.splitlines()[-4:]) if output else "OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    check_sources()
    run([sys.executable, "-m", "ruff", "check", "--no-cache", "."], env)
    run([sys.executable, "-m", "vulture"], env)
    if not args.static_only:
        env["NATIVE_GRPO_RUNTIME"] = str(ROOT / "datasets/grpo/integration")
        env["TRAINING_SERPER_MODULE_FILE"] = str(ROOT / "datasets/grpo/integration/project_runtime/serper_keys.py")
        for folder, pattern in (
            ("evaluation", "test_*.py"),
            ("datasets/grpo/tests", "test_project_tools.py"),
            ("sft_processing/cleaning", "test_*.py"),
            ("sft_processing/renumbering", "test_*.py"),
            ("sft_processing/teacher", "test_*.py"),
            ("sft_processing/teacher_simulation", "test_*.py"),
        ):
            run([sys.executable, "-B", "-m", "unittest", "discover", "-s", folder, "-p", pattern], env)


if __name__ == "__main__":
    main()
