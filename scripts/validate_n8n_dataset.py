"""Validate canonical n8n diagnosis data, safety rules and JSON Patch fixtures."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from score_n8n_repairs import CATEGORIES, CODES_BY_CATEGORY, apply_patch, parse_json_object, schema_errors

NODE_LEVEL_SETTINGS={"retryOnFail","maxTries","waitBetweenTries"}

def workflow_graph_is_sound(workflow: Any) -> bool:
    if not isinstance(workflow,dict): return False
    nodes,connections=workflow.get("nodes"),workflow.get("connections")
    if not isinstance(nodes,list) or not isinstance(connections,dict): return False
    names=[node.get("name") for node in nodes if isinstance(node,dict)]
    if len(names)!=len(nodes) or len(set(names))!=len(names) or None in names: return False
    known=set(names)
    for source,groups in connections.items():
        if source not in known or not isinstance(groups,dict): return False
        for outputs in groups.values():
            if not isinstance(outputs,list): return False
            for output_group in outputs:
                if not isinstance(output_group,list): return False
                for edge in output_group:
                    if not isinstance(edge,dict) or edge.get("node") not in known: return False
    return True

def node_settings_are_well_formed(workflow: Any) -> bool:
    if not isinstance(workflow,dict) or not isinstance(workflow.get("nodes"),list): return False
    for node in workflow["nodes"]:
        if not isinstance(node,dict): return False
        params=node.get("parameters",{})
        if not isinstance(params,dict) or NODE_LEVEL_SETTINGS.intersection(params): return False
    return True

def validate_dataset(data_dir: str | Path | None=None) -> dict[str,Any]:
    repo_root=Path(__file__).resolve().parents[1]
    root=Path(data_dir) if data_dir else repo_root/"recipes"/"n8n-workflow-repair"/"data"
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    expected=manifest.get("counts",{})
    expected_distribution=manifest.get("distribution",{})
    seen_prompts=set(); seen_ids=set()
    report={"counts":{},"categories":{},"codes":{},"patches_checked":0}
    errors=[]
    for split in ("train","valid","eval"):
        path=root/f"{split}.jsonl"
        rows=[]
        for line_number,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError as exc: errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
        report["counts"][split]=len(rows)
        if expected.get(split)!=len(rows): errors.append(f"{split}: manifest says {expected.get(split)}, file has {len(rows)}")
        category_counts=Counter(); code_counts=Counter()
        for index,row in enumerate(rows,1):
            case=f"{split}:{index}"
            messages=row.get("messages")
            if not isinstance(messages,list) or len(messages)!=3:
                errors.append(f"{case}: expected exactly three chat messages"); continue
            if [m.get("role") for m in messages]!=["system","user","assistant"]:
                errors.append(f"{case}: roles must be system, user, assistant"); continue
            prompt_text=messages[1].get("content")
            if not isinstance(prompt_text,str) or prompt_text in seen_prompts:
                errors.append(f"{case}: duplicate or invalid prompt"); continue
            seen_prompts.add(prompt_text)
            prompt=parse_json_object(prompt_text); answer=parse_json_object(messages[2].get("content"))
            if prompt is None or not isinstance(prompt.get("workflow"),dict):
                errors.append(f"{case}: user message must contain workflow JSON"); continue
            if answer is None:
                errors.append(f"{case}: assistant message is not JSON"); continue
            if not node_settings_are_well_formed(prompt["workflow"]):
                errors.append(f"{case}: retry settings must be node-level properties")
            case_id=prompt.get("case_id")
            if not isinstance(case_id,str) or not case_id: errors.append(f"{case}: missing case_id")
            elif case_id in seen_ids: errors.append(f"{case}: duplicate case_id {case_id}")
            else: seen_ids.add(case_id)
            output_errors=schema_errors(answer)
            if output_errors:
                errors.append(f"{case}: output schema errors: {'; '.join(output_errors)}"); continue
            category,code=answer["category"],answer["diagnosis_code"]
            category_counts[category]+=1; code_counts[code]+=1
            if category=="credentials" and (answer["patch"]!=[] or answer["needs_human_review"] is not True):
                errors.append(f"{case}: credentials must abstain")
            if category=="execution_api" and code in {"UPSTREAM_SERVICE_UNAVAILABLE","DNS_OR_CONNECTION_FAILURE"}:
                if answer["patch"]!=[] or answer["needs_human_review"] is not True: errors.append(f"{case}: outage/network must abstain")
            if category=="item_linking" and code=="ITEM_LINEAGE_UNAVAILABLE":
                evidence = " ".join([
                    str(prompt.get("execution_error", "")),
                    str(prompt.get("expected_behavior", "")),
                    str(prompt.get("evidence_note", "")),
                ]).lower()
                singleton_proven = any(
                    phrase in evidence for phrase in (
                        "exactly one item", "single result", "returns one item", "only one item"
                    )
                )
                if singleton_proven:
                    safe_first = any(
                        ".first()" in str(operation.get("value", ""))
                        for operation in answer["patch"]
                    )
                    if not answer["patch"] or answer["needs_human_review"] is True or not safe_first:
                        errors.append(f"{case}: proven singleton lineage needs a minimal .first() patch")
                elif answer["patch"] != [] or answer["needs_human_review"] is not True:
                    errors.append(f"{case}: unresolved item lineage must abstain without a patch")
            if answer["patch"]:
                try:
                    updated=apply_patch(prompt["workflow"],answer["patch"])
                    if not workflow_graph_is_sound(updated): errors.append(f"{case}: patched workflow graph is invalid")
                    report["patches_checked"]+=1
                except (KeyError,IndexError,TypeError,ValueError) as exc:
                    errors.append(f"{case}: patch cannot be applied: {exc}")
        report["categories"][split]=dict(sorted(category_counts.items()))
        report["codes"][split]=dict(sorted(code_counts.items()))
        desired=expected_distribution.get(split,{})
        if desired and dict(category_counts)!=desired: errors.append(f"{split}: category distribution differs from manifest")
        missing=[code for values in CODES_BY_CATEGORY.values() for code in values if not code_counts[code]]
        if missing: errors.append(f"{split}: missing diagnosis codes: {', '.join(missing)}")
    if expected.get("total")!=sum(report["counts"].values()): errors.append("manifest total mismatch")
    code_count=sum(len(codes) for codes in CODES_BY_CATEGORY.values())
    if code_count!=48: errors.append(f"codebook should contain 48 labels, found {code_count}")
    if errors: raise ValueError(f"Dataset validation failed with {len(errors)} issue(s):\n"+"\n".join(errors[:40]))
    report["total"]=sum(report["counts"].values())
    report["unique_prompts"]=len(seen_prompts)
    report["unique_case_ids"]=len(seen_ids)
    report["canonical_diagnosis_codes"]=code_count
    report["status"]="passed"
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=None)
    args=parser.parse_args()
    print(json.dumps(validate_dataset(args.data_dir),indent=2))
