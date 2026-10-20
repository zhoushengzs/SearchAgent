'Build the prompt sent to the VLM for each sample.'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "annotation.txt"
DEFAULT_ADDENDA_DIR = Path(__file__).resolve().parent.parent / "prompts" / "sources"

# The addendum goes right before this heading so the model reads it as part of
# the instruction, not as part of the sample.
_FINAL_HEADING = "## 11. Now annotate the following sample."


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class Instruction:
    'A loaded prompt file, carrying its identity for provenance.'

    def __init__(self, path: str | Path = DEFAULT_PROMPT_PATH,
                 addenda_dir: str | Path | None = DEFAULT_ADDENDA_DIR) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {self.path}")
        self.text = self.path.read_text(encoding="utf-8").strip()
        if not self.text:
            raise ValueError(f"Prompt file is empty: {self.path}")
        self.name = self.path.name
        self.sha = _sha(self.text)
        self.addenda: dict[str, str] = {}
        if addenda_dir is not None and Path(addenda_dir).is_dir():
            for f in sorted(Path(addenda_dir).glob("*.txt")):
                body = f.read_text(encoding="utf-8").strip()
                if body:
                    self.addenda[f.stem] = body

    def __len__(self) -> int:
        return len(self.text)

    def text_for(self, source: str) -> str:
        """The instruction for one source: base text plus its addendum, if any."""
        addendum = self.addenda.get(source)
        if not addendum:
            return self.text
        head, sep, tail = self.text.rpartition(_FINAL_HEADING)
        if not sep:  # heading edited away in an A/B copy: append at the end
            return f"{self.text}\n\n{addendum}"
        return f"{head}{addendum}\n\n{sep}{tail}"

    def sha_for(self, source: str) -> str:
        return _sha(self.text_for(source)) if source in self.addenda else self.sha

    def addendum_name(self, source: str) -> str | None:
        return f"{source}.txt" if source in self.addenda else None


def build_prompt(sample: dict[str, Any],
                 instruction: Instruction | None = None) -> str:
    'Assemble the text prompt for one sample.'
    source = str(sample.get("source_dataset", ""))
    body = (instruction or Instruction()).text_for(source)
    question = str(sample.get("question", "")).strip()
    variant = str(sample.get("trace_variant", "default"))
    size = sample.get("image_size") or {}
    width, height = size.get("width"), size.get("height")

    parts: list[str] = [
        body,
        "",
        "### Input",
        "question: " + json.dumps(question, ensure_ascii=False),
    ]
    if width and height:
        parts.append(f"image_size: {width} x {height}")
    parts += [
        f"trace_variant: {variant}",
    ]
    parts.append("")
    parts.append(
        "### Output\nFollow the CURRENT OPERATION appended by the executor. "
        "Return only its JSON object. The answer is withheld."
    )
    return "\n".join(parts)
