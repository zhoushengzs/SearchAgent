"""Structured response contracts for collection and independent audit."""
from __future__ import annotations


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def array(items):
    return {"type": "array", "items": items}


TEXT = {"type": "string"}
TEXTS = array(TEXT)
STEPS = array({"type": "integer", "minimum": 0})
STATUS = {"type": "string", "enum": ["supported", "contradicted", "unverifiable"]}
KIND = {"type": "string", "enum": ["visual", "language", "derived", "conditional", "uncertainty"]}
REQUEST_SCHEMA = object_schema({
    "tool": {"type": "string", "enum": ["web_search", "image_search", "visit", "inspect_image", "crop", "calculate", "finish", "abstain"]},
    "arguments": {"anyOf": [
        object_schema({"query": TEXT}), object_schema({"url": TEXT, "offset": {"type": "integer", "minimum": 0}}),
        object_schema({"asset_id": TEXT}), object_schema({"image_url": TEXT}),
        object_schema({"asset_id": TEXT, "bbox": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}, "minItems": 4, "maxItems": 4}}),
        object_schema({"expression": TEXT}),
        object_schema({"answers": TEXTS, "reasoning": TEXT, "evidence_steps": STEPS}),
        object_schema({"reason": TEXT, "missing_evidence": TEXTS})]},
    "rationale": TEXT, "information_need": TEXT,
    "gap_type": {"type": "string", "enum": ["visual", "language", "mixed", "none"]},
    "depends_on": array({"type": "integer", "minimum": 1})})
REQUEST_SCHEMA["properties"]["tool"]["description"] = (
    "web_search and image_search are TEXT-query searches; both require arguments.query. "
    "image_search does not accept an image or asset ID and cannot do reverse-image search. "
    "Use inspect_image with arguments.asset_id to inspect an existing image.")
REQUEST_SCHEMA["properties"]["arguments"]["description"] = (
    "Select the argument shape that matches the chosen tool: web_search/image_search={query}; "
    "visit={url,offset}; inspect_image={asset_id} or {image_url}; crop={asset_id,bbox}; "
    "calculate={expression}; finish={answers,reasoning,evidence_steps}; abstain={reason,missing_evidence}. "
    "Never combine image_search with asset_id or image_url. Query must be descriptive text, not image_1.")
OBSERVATION_SCHEMA = object_schema({"visual_facts": TEXTS, "language_facts": TEXTS, "reasoning": TEXT,
    "hypothesis": {"anyOf": [TEXT, {"type": "null"}]},
    "revision_of_step": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]}, "open_questions": TEXTS})
READER_SCHEMA = object_schema({"facts": TEXTS, "uncertain_details": TEXTS})
EVALUATION_SCHEMA = object_schema({"correct": {"type": "boolean"}, "complete": {"type": "boolean"}, "explanation": TEXT})


def request_schema(prefix_length):
    schema = {**REQUEST_SCHEMA, "properties": dict(REQUEST_SCHEMA["properties"])}
    schema["properties"]["depends_on"] = (array({"type": "integer", "minimum": 1, "maximum": prefix_length})
        if prefix_length else {"type": "array", "items": {"type": "integer"}, "maxItems": 0})
    return schema


def audit_schema(unit_ids, context_limit):
    return object_schema({"complete": {"type": "boolean"}, "units": array(object_schema({
        "unit_id": {"type": "string", "enum": unit_ids}, "status": STATUS,
        "claims": array(object_schema({"candidate_quote": TEXT, "kind": KIND, "status": STATUS,
            "evidence_steps": array({"type": "integer", "minimum": 0, "maximum": context_limit})}))}))})
