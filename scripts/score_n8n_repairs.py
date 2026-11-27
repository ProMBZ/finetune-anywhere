"""Score n8n outputs against held-out references and workflow fixtures."""
from __future__ import annotations
import json, re
from typing import Any

CODES_BY_CATEGORY={
 "expression":{"EXPRESSION_PREFIX_MISSING","NODE_REFERENCE_INVALID","PROPERTY_PATH_INVALID","EXPRESSION_METHOD_INVALID","EXPRESSION_SYNTAX_INVALID","EXPRESSION_DATE_FORMAT_INVALID"},
 "data_mapping":{"FIELD_MISSING_OR_NULL","ARRAY_INDEX_INCORRECT","TYPE_COERCION_REQUIRED","DEFAULT_OPERATOR_LOSES_FALSY_VALUE","BOOLEAN_STRING_COERCION","NESTED_FIELD_PATH_MISMATCH"},
 "item_linking":{"ITEM_REFERENCE_AMBIGUOUS","PAIRED_ITEM_METADATA_MISSING","WRONG_ITEM_SELECTOR","FIRST_ITEM_USED_WHEN_ALL_REQUIRED","UPSTREAM_NODE_REFERENCE_WRONG","ITEM_LINEAGE_UNAVAILABLE"},
 "node_configuration":{"HTTP_METHOD_MISMATCH","HTTP_BODY_NOT_SENT","RESPONSE_FORMAT_MISMATCH","TIMEOUT_VALUE_INVALID","TIMEOUT_TOO_SHORT","WAIT_DURATION_MISMATCH"},
 "workflow_structure":{"CONNECTION_MISSING","CONNECTION_TARGET_INVALID","IF_BRANCH_MISROUTED","MERGE_INPUT_MISSING","NODE_DISCONNECTED","OUTPUT_INDEX_INVALID"},
 "code_node":{"RETURN_VALUE_NOT_ARRAY","ITEM_JSON_WRAPPER_MISSING","EXECUTION_MODE_MISMATCH","VARIABLE_UNDEFINED","PRIMITIVE_OUTPUT_RETURNED","PAIRED_ITEM_NOT_PRESERVED"},
 "execution_api":{"RATE_LIMIT_RETRY_POLICY_MISSING","API_PAYLOAD_FIELD_MISMATCH","API_PAYLOAD_TYPE_MISMATCH","UPSTREAM_SERVICE_UNAVAILABLE","DNS_OR_CONNECTION_FAILURE","REQUEST_TIMEOUT"},
 "credentials":{"AUTHENTICATION_REJECTED","PERMISSION_INSUFFICIENT","CREDENTIAL_REFERENCE_MISSING","OAUTH_CREDENTIAL_EXPIRED","API_KEY_REVOKED_OR_MISSING","SECRET_VALUE_REDACTED"},
}
CATEGORIES=set(CODES_BY_CATEGORY); PATCH_OPS={"add","replace","remove"}
SENSITIVE_SEGMENTS={"credential","credentials","password","token","apikey","api_key","secret","privatekey","private_key"}

def parse_json_object(text: str) -> dict[str,Any]|None:
    if not isinstance(text,str): return None
    candidate=text.strip(); fence=chr(96)*3
    candidate=re.sub(r"^"+re.escape(fence)+r"(?:json)?\s*","",candidate,flags=re.IGNORECASE)
    candidate=re.sub(r"\s*"+re.escape(fence)+r"$","",candidate)
    start=candidate.find("{")
    if start<0: return None
    try: value,_=json.JSONDecoder().raw_decode(candidate[start:])
    except (json.JSONDecodeError,ValueError): return None
    return value if isinstance(value,dict) else None

def _pointer_parts(pointer: str) -> list[str]:
    if pointer=="": return []
    if not pointer.startswith("/"): raise ValueError("JSON Pointer must start with '/'")
    return [p.replace("~1","/").replace("~0","~") for p in pointer[1:].split("/")]

def _is_sensitive(pointer: str) -> bool:
    parts=[p.lower() for p in _pointer_parts(pointer)]
    return any(p in SENSITIVE_SEGMENTS or any(s in p for s in ("password","token","secret","apikey","credential")) for p in parts)

def schema_errors(value: dict[str,Any]) -> list[str]:
    errors=[]; required={"category","diagnosis_code","faulty_node","root_cause","patch","needs_human_review"}
    if set(value)!=required: errors.append("wrong top-level keys")
    category=value.get("category")
    if not isinstance(category,str) or category not in CATEGORIES: errors.append("invalid category")
    valid=CODES_BY_CATEGORY.get(category,set()) if isinstance(category,str) else set()
    code=value.get("diagnosis_code")
    if not isinstance(code,str) or code not in valid: errors.append("diagnosis_code is not in the codebook for category")
    for key in ("diagnosis_code","faulty_node","root_cause"):
        if not isinstance(value.get(key),str) or not value[key].strip(): errors.append(f"{key} must be nonempty text")
    patch=value.get("patch")
    if not isinstance(patch,list): errors.append("patch must be an array")
    else:
        for i,op in enumerate(patch):
            if not isinstance(op,dict): errors.append(f"patch[{i}] must be an object"); continue
            if op.get("op") not in PATCH_OPS: errors.append(f"patch[{i}] has unsupported operation")
            path=op.get("path")
            if not isinstance(path,str) or not path.startswith("/"): errors.append(f"patch[{i}] has invalid JSON Pointer")
            else:
                try:
                    if _is_sensitive(path): errors.append(f"patch[{i}] targets sensitive path")
                except ValueError: errors.append(f"patch[{i}] has invalid JSON Pointer")
            if op.get("op") in {"add","replace"} and "value" not in op: errors.append(f"patch[{i}] is missing value")
            if op.get("op")=="remove" and "value" in op: errors.append(f"patch[{i}] remove must not include value")
    if not isinstance(value.get("needs_human_review"),bool): errors.append("needs_human_review must be boolean")
    if category=="credentials" and (patch!=[] or value.get("needs_human_review") is not True): errors.append("credential cases must abstain")
    if category=="execution_api" and code in {"UPSTREAM_SERVICE_UNAVAILABLE","DNS_OR_CONNECTION_FAILURE"} and (patch!=[] or value.get("needs_human_review") is not True): errors.append("external outage/network cases must abstain")
    return errors

def apply_patch(document: Any,operations: list[dict[str,Any]]) -> Any:
    root=json.loads(json.dumps(document))
    for operation in operations:
        op,path=operation["op"],operation["path"]
        if op not in PATCH_OPS: raise ValueError(f"unsupported patch operation: {op}")
        if not isinstance(path,str): raise ValueError("patch path must be a string")
        if _is_sensitive(path): raise ValueError("patch targets sensitive field")
        parts=_pointer_parts(path)
        if not parts:
            if op=="replace": root=operation["value"]; continue
            raise ValueError("root add/remove is not supported")
        parent=root
        for part in parts[:-1]:
            if isinstance(parent,list): parent=parent[int(part)]
            elif isinstance(parent,dict) and part in parent: parent=parent[part]
            else: raise KeyError(f"parent path does not exist: {path}")
        key=parts[-1]
        if isinstance(parent,list):
            if key=="-" and op=="add": parent.append(operation["value"])
            else:
                index=int(key)
                if op=="add":
                    if index<0 or index>len(parent): raise IndexError(path)
                    parent.insert(index,operation["value"])
                elif op=="replace": parent[index]=operation["value"]
                elif op=="remove": del parent[index]
        elif isinstance(parent,dict):
            if op=="add": parent[key]=operation["value"]
            elif op=="replace":
                if key not in parent: raise KeyError(f"replace target does not exist: {path}")
                parent[key]=operation["value"]
            elif op=="remove":
                if key not in parent: raise KeyError(f"remove target does not exist: {path}")
                del parent[key]
        else: raise TypeError(f"parent is not object or array: {path}")
    return root

def _workflow_is_structurally_sound(workflow: Any) -> bool:
    if not isinstance(workflow,dict): return False
    nodes,connections=workflow.get("nodes"),workflow.get("connections")
    if not isinstance(nodes,list) or not isinstance(connections,dict): return False
    names=[n.get("name") for n in nodes if isinstance(n,dict)]
    if len(names)!=len(nodes) or len(set(names))!=len(names) or None in names: return False
    known=set(names)
    for source,groups in connections.items():
        if source not in known or not isinstance(groups,dict): return False
        for outputs in groups.values():
            if not isinstance(outputs,list): return False
            for group in outputs:
                if not isinstance(group,list): return False
                for edge in group:
                    if not isinstance(edge,dict) or edge.get("node") not in known: return False
    return True

def _patch_applicable(output: dict[str,Any],prompt: str) -> bool|None:
    patch=output.get("patch")
    if not isinstance(patch,list): return False
    if not patch: return None if output.get("needs_human_review") is True else False
    try: updated=apply_patch(json.loads(prompt).get("workflow"),patch)
    except (KeyError,IndexError,TypeError,ValueError,json.JSONDecodeError): return False
    return _workflow_is_structurally_sound(updated)

def _pct(n: int,d: int) -> str:
    return "n/a" if d==0 else f"{100*n/d:.1f}% ({n}/{d})"

def _metrics(outputs: list[str],references: list[str],prompts: list[str]) -> dict[str,Any]:
    if not (len(outputs)==len(references)==len(prompts)): raise ValueError("outputs, references and prompts must have identical lengths")
    total=len(references)
    counts={"valid_json":0,"valid_schema":0,"category_correct":0,"diagnosis_code_correct":0,"diagnosis_code_valid":0,
      "code_correct_when_category_correct":0,"category_correct_denominator":0,"gold_patch_match":0,
      "human_review_correct":0,"review_tp":0,"review_fp":0,"review_fn":0,"patch_applicable":0,"patches_attempted":0,
      "safety_violations":0,"exact_reference_match":0}
    by_category={}; by_code={}; confusion={}; unexpected={}
    for output_text,reference_text,prompt_text in zip(outputs,references,prompts):
        ref=parse_json_object(reference_text)
        if ref is None: raise ValueError("A reference answer is not valid JSON")
        true_cat=ref.get("category","unknown"); true_code=ref.get("diagnosis_code","unknown")
        expected_review=ref.get("needs_human_review") is True
        cat=by_category.setdefault(true_cat,{"total":0,"category_correct":0,"diagnosis_code_correct":0,"human_review_correct":0,"valid_json":0})
        cat["total"]+=1
        code=by_code.setdefault(true_code,{"total":0,"correct":0}); code["total"]+=1
        out=parse_json_object(output_text)
        if out is None:
            if expected_review: counts["review_fn"]+=1
            continue
        counts["valid_json"]+=1; cat["valid_json"]+=1
        if not schema_errors(out): counts["valid_schema"]+=1
        pred_cat=out.get("category"); pred_code=out.get("diagnosis_code")
        pred_review=out.get("needs_human_review") is True
        if pred_review==expected_review:
            counts["human_review_correct"]+=1; cat["human_review_correct"]+=1
        if expected_review and pred_review: counts["review_tp"]+=1
        elif pred_review and not expected_review: counts["review_fp"]+=1
        elif expected_review and not pred_review: counts["review_fn"]+=1
        if pred_cat==true_cat:
            counts["category_correct"]+=1; counts["category_correct_denominator"]+=1; cat["category_correct"]+=1
        if isinstance(pred_cat,str) and pred_cat in CODES_BY_CATEGORY and isinstance(pred_code,str):
            if pred_code in CODES_BY_CATEGORY[pred_cat]: counts["diagnosis_code_valid"]+=1
            else: unexpected[pred_code]=unexpected.get(pred_code,0)+1
        if pred_code==true_code:
            counts["diagnosis_code_correct"]+=1; code["correct"]+=1
            if pred_cat==true_cat: counts["code_correct_when_category_correct"]+=1
        confusion.setdefault(true_code,{})
        label=pred_code if isinstance(pred_code,str) else "<missing>"
        confusion[true_code][label]=confusion[true_code].get(label,0)+1
        if out.get("patch")==ref.get("patch"): counts["gold_patch_match"]+=1
        if out==ref: counts["exact_reference_match"]+=1
        patch=out.get("patch")
        if isinstance(patch,list) and patch:
            counts["patches_attempted"]+=1
            if _patch_applicable(out,prompt_text): counts["patch_applicable"]+=1
        try:
            if isinstance(patch,list) and any(isinstance(x,dict) and isinstance(x.get("path"),str) and _is_sensitive(x["path"]) for x in patch):
                counts["safety_violations"]+=1
        except (ValueError,TypeError): counts["safety_violations"]+=1
    macro=sum(r["correct"]/r["total"] for r in by_code.values() if r["total"])/max(1,len(by_code))
    by_cat={}
    for category,row in by_category.items():
        codes=[c for c in by_code if c in CODES_BY_CATEGORY.get(category,set())]
        denom=sum(by_code[c]["total"] for c in codes); correct=sum(by_code[c]["correct"] for c in codes)
        by_cat[category]={**row,"category_accuracy":_pct(row["category_correct"],row["total"]),"diagnosis_code_accuracy":_pct(correct,denom)}
    f1_den=2*counts["review_tp"]+counts["review_fp"]+counts["review_fn"]
    return {
      "examples_scored":total,"valid_json":_pct(counts["valid_json"],total),"valid_schema":_pct(counts["valid_schema"],total),
      "category_accuracy":_pct(counts["category_correct"],total),"diagnosis_code_accuracy":_pct(counts["diagnosis_code_correct"],total),
      "diagnosis_code_macro_accuracy":f"{100*macro:.1f}%" if by_code else "n/a",
      "diagnosis_code_validity":_pct(counts["diagnosis_code_valid"],total),
      "diagnosis_code_accuracy_when_category_correct":_pct(counts["code_correct_when_category_correct"],counts["category_correct_denominator"]),
      "gold_patch_match":_pct(counts["gold_patch_match"],total),
      "human_review_decision_accuracy":_pct(counts["human_review_correct"],total),
      "safe_abstention_precision":_pct(counts["review_tp"],counts["review_tp"]+counts["review_fp"]),
      "safe_abstention_recall":_pct(counts["review_tp"],counts["review_tp"]+counts["review_fn"]),
      "safe_abstention_f1":"n/a" if f1_den==0 else f"{200*counts['review_tp']/f1_den:.1f}%",
      "exact_reference_match":_pct(counts["exact_reference_match"],total),
      "patch_applicability":_pct(counts["patch_applicable"],counts["patches_attempted"]),
      "patches_attempted":counts["patches_attempted"],"safety_violations":counts["safety_violations"],
      "by_category":by_cat,"by_diagnosis_code":by_code,"diagnosis_code_confusion":confusion,
      "unexpected_diagnosis_codes":unexpected,
      "interpretation":"Malformed outputs count as incorrect in full denominators, including diagnosis-code validity and safe-abstention recall. Human-review precision/recall measure agreement with the reference review flag, not a general safety guarantee. Patch applicability is a structural fixture check, not live execution in n8n; gold-patch match is strict."
    }

def score_outputs(outputs: list[str],references: list[str],prompts: list[str]) -> dict[str,Any]:
    return _metrics(outputs,references,prompts)

def score_comparison_file(path: str) -> dict[str,Any]:
    with open(path,encoding="utf-8") as f: data=json.load(f)
    rows=data.get("records",data) if isinstance(data,dict) else data
    if not isinstance(rows,list): raise ValueError("comparison file must contain records array")
    refs=[]; prompts=[]; base=[]; tuned=[]
    for row in rows:
        messages=row.get("messages",[])
        if len(messages)<2: raise ValueError("each record must include prompt messages")
        prompts.append(messages[-1].get("content","")); refs.append(row.get("reference",""))
        base.append(row.get("base_output",row.get("base",""))); tuned.append(row.get("finetuned_output",row.get("finetuned","")))
    return {"base_model":score_outputs(base,refs,prompts),"fine_tuned_model":score_outputs(tuned,refs,prompts)}

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("comparison_json"); args=parser.parse_args()
    print(json.dumps(score_comparison_file(args.comparison_json),indent=2))
