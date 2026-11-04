import os, json, subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(
    os.environ.get("SFT_WORK", str(Path.home() / "internvl3-sft-work"))
).resolve()
versions = {
    "code": subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if (Path(__file__).resolve().parents[2] / ".git").exists()
    else "container-source-see-build-record",
    "model": "e22931943e5336f85e06f4e2b38f3e5e6ee4de3b",
    "dataset": "29211998029f2ab3481604bd3bf30d60dd99a51d",
}
for repo, kind, dest, key in [
    ("OpenGVLab/InternVL3-14B-hf", "model", "model", "model"),
    ("zhoushengzs/OpenSearch-VL-SFT-Original", "dataset", "data", "dataset"),
]:
    snapshot_download(
        repo_id=repo, repo_type=kind, revision=versions[key], local_dir=root / dest
    )
(root / "versions.json").write_text(json.dumps(versions, indent=2))
print("Downloaded to", root)
