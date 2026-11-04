import os, json
from pathlib import Path

root = Path(
    os.environ.get("SFT_WORK", str(Path.home() / "internvl3-sft-work"))
).resolve()
root.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
for name in ["smoke.json", "zero2.json"]:
    content = json.loads((here / "configs" / name).read_text())

    def expand(value):
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        if isinstance(value, str):
            return value.replace("${SFT_WORK}", str(root))
        return value

    (root / name).write_text(
        json.dumps(expand(content), ensure_ascii=False, indent=2) + "\n"
    )
print(root / "smoke.json")
