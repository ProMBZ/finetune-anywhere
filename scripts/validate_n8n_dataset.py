"""Validate the n8n repair recipe's JSONL splits and gold patches."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from score_n8n_repairs import CATEGORIES, apply_patch, parse_json_object, schema_errors

NODE_LEVEL_SETTINGS = {"retryOnFail", "maxTries", "waitBetweenTries"}

def workflow_graph_is_sound(workflow: Any) -> bool:
    if not isinstance(workflow, dict):
        return False
    nodes, connections = workflow.get("nodes"), workflow.get("connections")
    if not isinstance(nodes, list) or not isinstance(connections, dict):
        return False
    names = [node.get("name") for node in nodes if isinstance(node, dict)]
    if len(names) != len(nodes) or len(set(names)) != len(names) or None in names:
        return False
    known = set(names)
    for source, groups in connections.items():
        if source not in known or not isinstance(groups, dict):
            return False
        for outputs in groups.values():
            if not isinstance(outputs, list):
                return False
            for output_group in outputs:
                if not isinstance(output_group, list):
                    return False
                for edge in output_group:
                    if not isinstance(edge, dict) or edge.get("node") not in known:
                        return False
    return True

def node_settings_are_well_formed(workflow: Any) -> bool:
    """Reject retry settings nested under parameters; n8n stores them on the node."""
    if not isinstance(workflow, dict) or not isinstance(workflow.get("nodes"), list):
        return False
    for node in workflow["nodes"]:
        if not isinstance(node, dict):
            return False
        parameters = node.get("parameters", {})
        if not isinstance(parameters, dict):
            return False
        if NODE_LEVEL_SETTINGS.intersection(parameters):
            return False
    return True


def validate_dataset(data_dir: str | Path | None = None) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    root = Path(data_dir) if data_dir else repo_root / "recipes" / "n8n-workflow-repair" / "data"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("counts", {})
    seen_prompts: set[str] = set()
    seen_ids: set[str] = set()
    report: dict[str, Any] = {"counts": {}, "categories": {}, "patches_checked": 0}
    errors: list[str] = []
    for split in ("train", "valid", "eval"):
        path = root / f"{split}.jsonl"
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
        report["counts"][split] = len(rows)
        if expected.get(split) != len(rows):
            errors.append(f"{split}: manifest says {expected.get(split)}, file has {len(rows)}")
        category_counts: Counter[str] = Counter()
        for index, row in enumerate(rows, start=1):
            case = f"{split}:{index}"
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) != 3:
                errors.append(f"{case}: expected exactly three chat messages")
                continue
            if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
                errors.append(f"{case}: roles must be system, user, assistant")
                continue
            prompt_text = messages[1].get("content")
            if not isinstance(prompt_text, str) or prompt_text in seen_prompts:
                errors.append(f"{case}: duplicate or invalid user prompt")
                continue
            seen_prompts.add(prompt_text)
            prompt = parse_json_object(prompt_text)
            answer = parse_json_object(messages[2].get("content"))
            if prompt is None or not isinstance(prompt.get("workflow"), dict):
                errors.append(f"{case}: user message must contain a workflow JSON object")
                continue
            if answer is None:
                errors.append(f"{case}: assistant message is not a JSON object")
                continue
            if not node_settings_are_well_formed(prompt["workflow"]):
                errors.append(f"{case}: n8n retry settings must be top-level node properties, not inside parameters")
            case_id = prompt.get("case_id")
            if case_id:
                if case_id in seen_ids:
                    errors.append(f"{case}: duplicate case_id {case_id}")
                seen_ids.add(case_id)
            schema = schema_errors(answer)
            if schema:
                errors.append(f"{case}: output schema errors: {'; '.join(schema)}")
                continue
            category = answer["category"]
            if category not in CATEGORIES:
                errors.append(f"{case}: unsupported category {category}")
            category_counts[category] += 1
            if category == "credentials" and (answer["patch"] != [] or answer["needs_human_review"] is not True):
                errors.append(f"{case}: credential case must abstain")
            if answer["patch"]:
                try:
                    updated = apply_patch(prompt["workflow"], answer["patch"])
                    if not workflow_graph_is_sound(updated):
                        errors.append(f"{case}: patched workflow has invalid node connections")
                    report["patches_checked"] += 1
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    errors.append(f"{case}: patch cannot be applied safely: {exc}")
        report["categories"][split] = dict(sorted(category_counts.items()))
    if expected.get("total") != sum(report["counts"].values()):
        errors.append("manifest total does not equal the three split sizes")
    if errors:
        raise ValueError(f"Dataset validation failed with {len(errors)} issue(s):\n" + "\n".join(errors[:40]))
    report["total"] = sum(report["counts"].values())
    report["unique_prompts"] = len(seen_prompts)
    report["unique_case_ids"] = len(seen_ids)
    report["status"] = "passed"
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.data_dir), indent=2))
