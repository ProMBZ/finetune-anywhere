"""Score n8n repair model outputs against held-out references and workflow fixtures.

This evaluates structured diagnosis and safe, applicable JSON patches. It does not
execute a live n8n workflow and must not be reported as runtime repair success.
"""
from __future__ import annotations

import json
import re
from typing import Any

CATEGORIES = {
    "expression",
    "data_mapping",
    "item_linking",
    "node_configuration",
    "workflow_structure",
    "code_node",
    "execution_api",
    "credentials",
}
PATCH_OPS = {"add", "replace", "remove"}
SENSITIVE_SEGMENTS = {
    "credential", "credentials", "password", "token", "apikey",
    "api_key", "secret", "privatekey", "private_key",
}


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating a short preamble or Markdown fence."""
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    candidate = re.sub(r"^\x60{3}(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*\x60{3}$", "", candidate)
    start = candidate.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def schema_errors(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "category", "diagnosis_code", "faulty_node",
        "root_cause", "patch", "needs_human_review",
    }
    if set(value) != required:
        errors.append("wrong top-level keys")
    if value.get("category") not in CATEGORIES:
        errors.append("invalid category")
    for key in ("diagnosis_code", "faulty_node", "root_cause"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            errors.append(f"{key} must be nonempty text")
    if not isinstance(value.get("patch"), list):
        errors.append("patch must be an array")
    else:
        for i, operation in enumerate(value["patch"]):
            if not isinstance(operation, dict):
                errors.append(f"patch[{i}] must be an object")
                continue
            if operation.get("op") not in PATCH_OPS:
                errors.append(f"patch[{i}] has an unsupported operation")
            if not isinstance(operation.get("path"), str) or not operation["path"].startswith("/"):
                errors.append(f"patch[{i}] has an invalid JSON Pointer")
            if operation.get("op") in {"add", "replace"} and "value" not in operation:
                errors.append(f"patch[{i}] is missing value")
            if operation.get("op") == "remove" and "value" in operation:
                errors.append(f"patch[{i}] remove must not include value")
    if not isinstance(value.get("needs_human_review"), bool):
        errors.append("needs_human_review must be boolean")
    if value.get("category") == "credentials":
        if value.get("patch") != [] or value.get("needs_human_review") is not True:
            errors.append("credential cases must abstain without a patch")
    return errors


def _pointer_parts(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must start with '/'")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _is_sensitive(pointer: str) -> bool:
    parts = [part.lower() for part in _pointer_parts(pointer)]
    return any(
        part in SENSITIVE_SEGMENTS
        or any(secret in part for secret in ("password", "token", "secret", "apikey", "credential"))
        for part in parts
    )


def apply_patch(document: Any, operations: list[dict[str, Any]]) -> Any:
    """Apply the supported RFC 6902 add/replace/remove subset to a copy."""
    root = json.loads(json.dumps(document))
    for operation in operations:
        op = operation["op"]
        path = operation["path"]
        if _is_sensitive(path):
            raise ValueError("patch targets a sensitive field")
        parts = _pointer_parts(path)
        if not parts:
            if op == "replace":
                root = operation["value"]
                continue
            raise ValueError("root add/remove is not supported")
        parent = root
        for part in parts[:-1]:
            if isinstance(parent, list):
                parent = parent[int(part)]
            elif isinstance(parent, dict) and part in parent:
                parent = parent[part]
            else:
                raise KeyError(f"parent path does not exist: {path}")
        key = parts[-1]
        if isinstance(parent, list):
            if key == "-" and op == "add":
                parent.append(operation["value"])
            else:
                index = int(key)
                if op == "add":
                    if index < 0 or index > len(parent):
                        raise IndexError(path)
                    parent.insert(index, operation["value"])
                elif op == "replace":
                    parent[index] = operation["value"]
                elif op == "remove":
                    del parent[index]
        elif isinstance(parent, dict):
            if op == "add":
                parent[key] = operation["value"]
            elif op == "replace":
                if key not in parent:
                    raise KeyError(f"replace target does not exist: {path}")
                parent[key] = operation["value"]
            elif op == "remove":
                if key not in parent:
                    raise KeyError(f"remove target does not exist: {path}")
                del parent[key]
        else:
            raise TypeError(f"parent is not an object or array: {path}")
    return root


def _prompt_case(prompt: str) -> dict[str, Any] | None:
    if not isinstance(prompt, str):
        return None
    try:
        value = json.loads(prompt)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _workflow_is_structurally_sound(workflow: Any) -> bool:
    if not isinstance(workflow, dict):
        return False
    nodes = workflow.get("nodes")
    connections = workflow.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        return False
    names = {node.get("name") for node in nodes if isinstance(node, dict)}
    if len(names) != len(nodes) or None in names:
        return False
    for source, groups in connections.items():
        if source not in names or not isinstance(groups, dict):
            return False
        for outputs in groups.values():
            if not isinstance(outputs, list):
                return False
            for output_group in outputs:
                if not isinstance(output_group, list):
                    return False
                for edge in output_group:
                    if not isinstance(edge, dict) or edge.get("node") not in names:
                        return False
    return True


def _patch_applicable(output: dict[str, Any], prompt: str) -> bool | None:
    patch = output.get("patch")
    if not isinstance(patch, list):
        return False
    if not patch:
        return None if output.get("needs_human_review") is True else False
    case = _prompt_case(prompt)
    workflow = case.get("workflow") if case else None
    if workflow is None:
        return None
    try:
        updated = apply_patch(workflow, patch)
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return _workflow_is_structurally_sound(updated)


def _metrics(outputs: list[str], references: list[str], prompts: list[str]) -> dict[str, Any]:
    total = min(len(outputs), len(references), len(prompts))
    counts = {
        "valid_json": 0,
        "valid_schema": 0,
        "category_correct": 0,
        "diagnosis_code_correct": 0,
        "gold_patch_match": 0,
        "human_review_correct": 0,
        "patch_applicable": 0,
        "patch_applicability_tested": 0,
        "safety_violations": 0,
        "exact_reference_match": 0,
    }
    by_category: dict[str, dict[str, int]] = {}
    for output_text, reference_text, prompt_text in zip(outputs[:total], references[:total], prompts[:total]):
        output = parse_json_object(output_text)
        reference = parse_json_object(reference_text)
        if output is None:
            continue
        counts["valid_json"] += 1
        errors = schema_errors(output)
        if not errors:
            counts["valid_schema"] += 1
        if reference is None:
            continue
        category = reference.get("category", "unknown")
        group = by_category.setdefault(category, {"total": 0, "category_correct": 0, "diagnosis_correct": 0, "patch_match": 0})
        group["total"] += 1
        if output.get("category") == reference.get("category"):
            counts["category_correct"] += 1
            group["category_correct"] += 1
        if output.get("diagnosis_code") == reference.get("diagnosis_code"):
            counts["diagnosis_code_correct"] += 1
            group["diagnosis_correct"] += 1
        if output.get("patch") == reference.get("patch"):
            counts["gold_patch_match"] += 1
            group["patch_match"] += 1
        if output.get("needs_human_review") == reference.get("needs_human_review"):
            counts["human_review_correct"] += 1
        if output == reference:
            counts["exact_reference_match"] += 1
        try:
            if any(_is_sensitive(item.get("path", "")) for item in output.get("patch", []) if isinstance(item, dict)):
                counts["safety_violations"] += 1
        except (ValueError, TypeError):
            counts["safety_violations"] += 1
        applicable = _patch_applicable(output, prompt_text)
        if applicable is not None:
            counts["patch_applicability_tested"] += 1
            if applicable:
                counts["patch_applicable"] += 1

    def pct(key: str, denominator: int = total) -> str:
        if denominator == 0:
            return "n/a"
        return f"{100 * counts[key] / denominator:.1f}% ({counts[key]}/{denominator})"

    applicable_denominator = counts["patch_applicability_tested"]
    return {
        "examples_scored": total,
        "valid_json": pct("valid_json"),
        "valid_schema": pct("valid_schema"),
        "category_accuracy": pct("category_correct"),
        "diagnosis_code_accuracy": pct("diagnosis_code_correct"),
        "gold_patch_match": pct("gold_patch_match"),
        "safe_abstention_accuracy": pct("human_review_correct"),
        "exact_reference_match": pct("exact_reference_match"),
        "patch_applicability": pct("patch_applicable", applicable_denominator),
        "patch_applicability_tests": applicable_denominator,
        "safety_violations": counts["safety_violations"],
        "by_category": by_category,
        "interpretation": "Patch applicability is a structural fixture check, not execution in n8n. Gold patch match compares against the authored reference and may be strict when multiple repairs are valid.",
    }


def score_outputs(
    outputs: list[str],
    references: list[str],
    prompts: list[str],
) -> dict[str, Any]:
    """Return metrics for one model's outputs on the same held-out cases."""
    if not (len(outputs) == len(references) == len(prompts)):
        raise ValueError("outputs, references and prompts must have identical lengths")
    return _metrics(outputs, references, prompts)


def score_comparison_file(path: str) -> dict[str, Any]:
    """Score a starter-style comparison.json evidence file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("records", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("comparison file must contain a records array")
    references, prompts, base, tuned = [], [], [], []
    for row in rows:
        messages = row.get("messages", [])
        if not messages:
            raise ValueError("each comparison record must include prompt messages")
        prompts.append(messages[-1].get("content", ""))
        references.append(row.get("reference", ""))
        base.append(row.get("base_output", row.get("base", "")))
        tuned.append(row.get("finetuned_output", row.get("finetuned", "")))
    return {
        "base_model": score_outputs(base, references, prompts),
        "fine_tuned_model": score_outputs(tuned, references, prompts),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_json", help="Starter notebook comparison.json")
    args = parser.parse_args()
    print(json.dumps(score_comparison_file(args.comparison_json), indent=2))
