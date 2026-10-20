'Registry of supported HuggingFace data sources.'
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Research candidates are protected even before they have download adapters.
# They must not appear in resolve_source_names until their adapters are implemented.
TRAJECTORY_CATALOG = json.loads(
    (Path(__file__).resolve().parents[1] / "trajectory_sources.json").read_text(encoding="utf-8")
)["sources"]

DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    # Search-grounded sources.
    "mmsearch": {
        "name": "MMSearch (end2end)",
        "hf_id": "CaraJ/MMSearch",
        "revision": "9c7f07a9822e7fb05ade6fa1ef38d0e6ad26205c",
        "config_name": "end2end",
        "split": "end2end",
        # image_search_result is a retrieved screenshot, not the question's
        # image; text-only queries have no query_image and are skipped.
        "image_fields": ["query_image"],
        "question_fields": ["query"],
        "answer_fields": ["gt_answer"],
        # Retained as source metadata; never sent to the generator.
        "reasoning_fields": ["gt_requery"],
        "id_fields": ["sample_id"],
        "domain": "multimodal_search",
        "enabled": True,
        "usage": "evaluation",
        "environment": {"kind": "multimodal_web_search", "required_tools": ["web_search", "image_search", "visit"],
                        "official_runner": "https://github.com/CaraJ7/MMSearch"},
    },
    "agentvista": {
        "name": "AgentVista",
        "hf_id": "Warrieryes/AgentVista",
        "config_name": None,
        "split": "train",
        "revision": "61de767ad8c72e258ac71a0aef952b89ca920a16",
        # The visual module handles one image; benchmark mode preserves all.
        "image_fields": ["images"],
        "question_fields": ["problem", "question"],
        "answer_fields": ["answer", "solution"],
        "reasoning_fields": [],
        "id_fields": ["question_id", "doc_id", "id"],
        "domain": "agentic_visual_tool_use",
        "enabled": True,
        "usage": "evaluation",
        "environment": {"kind": "long_horizon_multimodal_agent", "required_tools": ["web_search", "image_search", "visit", "code_interpreter"],
                        "official_runner": "https://github.com/hkust-nlp/AgentVista"},
    },
    "mmsearch_plus": {
        "name": "MMSearch-Plus",
        "hf_id": "Cie1/MMSearch-Plus",
        # Question/answer are XOR-encrypted on the Hub; the custom loader
        # decrypts them and keeps single-image tasks only.
        "custom_loader": "sources.mmsearch_plus:load_mmsearch_plus",
        "canary": "MMSearch-Plus",
        "revision": "161a60d469b67cd14a0621ae2ab66d902df02df9",
        "config_name": None,
        "split": "train",
        "image_fields": ["img_1"],
        "benchmark_image_fields": ["img_1", "img_2", "img_3", "img_4", "img_5"],
        "question_fields": ["question"],
        "answer_fields": ["answer"],
        "reasoning_fields": [],
        "id_fields": ["id"],
        "domain": "fine_grained_multimodal_search",
        "enabled": True,
        "usage": "evaluation",
        "environment": {"kind": "long_horizon_multimodal_search", "required_tools": ["web_search", "image_search", "visit", "crop"],
                        "official_runner": "https://mmsearch-plus.github.io/"},
    },
    "textvqa": {
        "name": "TextVQA", "hf_id": "lmms-lab-encoder/textvqa", "config_name": "default",
        "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "split": "train", "image_fields": ["image"], "question_fields": ["question"],
        "answer_fields": ["answers"], "answer_selection": "majority",
        "reasoning_fields": [], "id_fields": ["question_id"], "domain": "scene_text_reasoning",
        "usage": "train", "enabled": False,
    },
    "tallyqa_train": {
        "name": "TallyQA (Cauldron training subset)", "hf_id": "HuggingFaceM4/the_cauldron",
        "revision": "847a98a779b1652d65111daf20c972dfcd333605",
        "config_name": "tallyqa", "split": "train", "image_fields": ["images"],
        "qa_field": "texts", "qa_question_field": "user", "qa_answer_field": "assistant",
        "question_fields": ["question"], "answer_fields": ["answer"],
        "reasoning_fields": [], "id_fields": ["question_id"], "domain": "counting",
        "usage": "train", "enabled": False,
    },
    "infographic_train": {
        "name": "InfographicVQA (Cauldron training subset)", "hf_id": "HuggingFaceM4/the_cauldron",
        "revision": "847a98a779b1652d65111daf20c972dfcd333605",
        "config_name": "infographic_vqa", "split": "train", "image_fields": ["images"],
        "qa_field": "texts", "qa_question_field": "user", "qa_answer_field": "assistant",
        "question_fields": ["question"], "answer_fields": ["answer"],
        "reasoning_fields": [], "id_fields": ["question_id"], "domain": "infographic_reasoning",
        "usage": "train", "enabled": False,
    },
    "vstar_bench": {
        "name": "V* Bench", "hf_id": "lmms-lab-encoder/vstar-bench", "config_name": "default",
        "revision": "b44023b4dca749ed8a76b85eb576627d05a1c174",
        "split": "test", "image_fields": ["image"], "question_fields": ["text"],
        "answer_fields": ["label"], "reasoning_fields": [], "id_fields": ["question_id"],
        "domain": "fine_grained_visual_search", "usage": "evaluation", "enabled": False,
    },
    "docvqa": {
        "name": "DocVQA", "hf_id": "lmms-lab-encoder/DocVQA", "config_name": "DocVQA",
        "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "split": "validation", "image_fields": ["image"], "question_fields": ["question"],
        "answer_fields": ["answers"], "reasoning_fields": [], "id_fields": ["questionId", "question_id"],
        "domain": "document_reasoning", "usage": "evaluation", "enabled": False,
    },
    "infographic_bench": {
        "name": "InfographicVQA", "hf_id": "lmms-lab-encoder/DocVQA", "config_name": "InfographicVQA",
        "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "split": "validation", "image_fields": ["image"], "question_fields": ["question"],
        "answer_fields": ["answers"], "reasoning_fields": [], "id_fields": ["questionId", "question_id"],
        "domain": "infographic_reasoning", "usage": "evaluation", "enabled": False,
    },
}

# Author releases that require decoding, URL assets or archive-member access.
for _key, _loader, _split, _images, _answers, _ids in (
    ("mmbrowsecomp", "load_mmbrowsecomp", "release-400", "images", "answer", "id"),
    ("browsecomp_v3", "load_browsecomp_v3", "train", "images", "answer", "id"),
    ("dynvqa", "load_dynvqa", "English-202502", "images", "answer", "question_id"),
    ("m3vqa", "load_m3vqa", "test", "images", "answers", "data_id"),
    ("tirbench", "load_tirbench", "test", "images", "answer", "id"),
):
    _entry = next(entry for entry in TRAJECTORY_CATALOG if entry["key"] == _key)
    DATASET_REGISTRY[_key] = {
        "name": _entry["name"], "hf_id": _entry.get("hf_id"), "config_name": None,
        "revision": _entry.get("official_revision", _entry["revision"]), "split": _split,
        "custom_loader": "sources.public_benchmarks:" + _loader, "supports_local_data": True,
        "image_fields": [_images], "question_fields": ["question"], "answer_fields": [_answers],
        "reasoning_fields": [], "id_fields": [_ids], "enabled": False, "usage": "evaluation",
        "domain": "tool_integrated_visual_reasoning" if _key == "tirbench" else "multihop_visual_language_retrieval",
        "allow_text_only": _key == "mmbrowsecomp",
        "answer_selection": "all" if _key == "m3vqa" else "first",
        "reference_fields": {"mmbrowsecomp": ["checklist", "checklist_property", "source"],
                             "browsecomp_v3": ["metadata", "sub_goals"],
                             "dynvqa": ["golden_query"],
                             "m3vqa": ["answer_evals", "img_entity_names", "evidence", "evidence_urls",
                                       "evidence_img_ids", "evidence_section_ids", "evidence_section_titles",
                                       "evidence_url_titles"], "tirbench": ["meta_data"]}[_key],
        "environment": {"kind": "static_visual_tool_task" if _key == "tirbench" else "multihop_visual_language_retrieval",
                        "required_tools": ["inspect_image", "crop", "calculate"] if _key == "tirbench" else
                                          ["web_search", "image_search", "visit", "inspect_image", "crop"],
                        "official_runner": _entry["official_url"],
                        "default_protocol": "adapted_open_web_v1"},
    }

def dataset_usage(config: dict[str, Any], split: str) -> str:
    """Hub split labels do not override a benchmark's evaluation-only purpose."""
    if config.get("usage") != "train":
        return "evaluation"
    return "train" if split.split("[", 1)[0] == "train" else "evaluation"


def is_evaluation_source(source: str, provenance: dict | None = None) -> bool:
    """Protect known benchmarks even in older/local inputs lacking data_usage."""
    hf_id = provenance.get("hf_id") if isinstance(provenance, dict) else None
    return any(config.get("usage") == "evaluation" and
               (source in {key, config["hf_id"]} or (hf_id is not None and hf_id == config["hf_id"]))
               for key, config in DATASET_REGISTRY.items()) or any(
                   source in {entry["key"], entry.get("hf_id"), *entry.get("aliases", [])}
                   or (hf_id is not None and hf_id == entry.get("hf_id"))
                   for entry in TRAJECTORY_CATALOG)


def get_enabled_sources() -> list[str]:
    """Return the names of all enabled sources in registry order."""
    return [k for k, v in DATASET_REGISTRY.items() if v.get("enabled", True)]


def resolve_source_names(sources: list[str] | None) -> list[str]:
    'Resolve a user-provided list of source names against the registry.'
    if not sources:
        return get_enabled_sources()
    unknown = [s for s in sources if s not in DATASET_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown source(s): {unknown}. "
            f"Available: {sorted(DATASET_REGISTRY.keys())}"
        )
    return sources
