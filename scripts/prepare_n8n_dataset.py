"""Prepare a balanced, opaque-ID n8n repair dataset for Colab fine-tuning."""
from __future__ import annotations
import argparse, copy, hashlib, json, random, re, shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from score_n8n_repairs import CODES_BY_CATEGORY

PEOPLE={"Ari":"Riley","Bo":"Casey","Ria":"Jordan","Sam":"Taylor","Mina":"Morgan","Joe":"Avery","Ana":"Quinn","Lee":"Rowan"}

def _opaque_id(split: str, ordinal: int, origin: str, variant: int=0) -> str:
    raw=f"n8n-repair-v0.6|{split}|{ordinal}|{origin}|{variant}".encode()
    return "case-"+hashlib.sha256(raw).hexdigest()[:16]

def _rewrite_tree(value: Any, replacements: dict[str,str]) -> Any:
    """Replace each original node name once, without recursively rewriting new names."""
    ordered=sorted(replacements.items(),key=lambda pair:len(pair[0]),reverse=True)
    pattern=re.compile("|".join(re.escape(old) for old,_ in ordered)) if ordered else None
    def rewrite(item: Any) -> Any:
        if isinstance(item,dict):
            result={}
            for key,child in item.items():
                new_key=pattern.sub(lambda match:replacements[match.group(0)],key) if pattern else key
                if new_key in result: raise ValueError(f"node rename produced duplicate key: {new_key}")
                result[new_key]=rewrite(child)
            return result
        if isinstance(item,list): return [rewrite(child) for child in item]
        if isinstance(item,str) and pattern:
            return pattern.sub(lambda match:replacements[match.group(0)],item)
        return item
    return rewrite(value)

def _replace_domains(value: Any, variant: int) -> Any:
    if isinstance(value,dict): return {k:_replace_domains(v,variant) for k,v in value.items()}
    if isinstance(value,list): return [_replace_domains(v,variant) for v in value]
    if isinstance(value,str): return value.replace(".example.test",f".tenant-{variant:04d}.test")
    return value

def _mutate_input(value: Any, variant: int) -> Any:
    if isinstance(value,dict): return {k:_mutate_input(v,variant) for k,v in value.items()}
    if isinstance(value,list): return [_mutate_input(v,variant) for v in value]
    if not isinstance(value,str): return value
    suffix=f"v{variant:04d}"
    if value in PEOPLE: return PEOPLE[value]+suffix
    if "@" in value:
        local,_,domain=value.partition("@")
        domain=re.sub(r"\.example\.(?:test|com)$",f".tenant-{variant:04d}.test",domain)
        if domain==value.partition("@")[2]: domain=f"tenant-{variant:04d}.test"
        return f"{local}.{suffix}@{domain}"
    if re.fullmatch(r"[A-Za-z]{1,5}-\d{3,}",value): return f"{value}-{suffix.upper()}"
    return value.replace(".example.test",f".tenant-{variant:04d}.test")

def _make_variant(source_row: dict[str,Any], ordinal: int, variant: int) -> dict[str,Any]:
    row=copy.deepcopy(source_row)
    messages=row.get("messages")
    if not isinstance(messages,list) or len(messages)!=3: raise ValueError("expected three chat messages")
    source_prompt=json.loads(messages[1]["content"])
    replacements={}
    for node in source_prompt.get("workflow",{}).get("nodes",[]):
        name=node.get("name") if isinstance(node,dict) else None
        if isinstance(name,str) and name: replacements[name]=f"{name} [variant {variant:04d}]"
    row=_rewrite_tree(row,replacements)
    row=_replace_domains(row,variant)
    messages=row["messages"]; prompt=json.loads(messages[1]["content"]); answer=json.loads(messages[2]["content"])
    prompt["input_data"]=_mutate_input(prompt.get("input_data"),variant)
    for index,node in enumerate(prompt.get("workflow",{}).get("nodes",[])):
        pos=node.get("position") if isinstance(node,dict) else None
        if isinstance(pos,list) and len(pos)==2 and all(isinstance(x,(int,float)) for x in pos):
            pos[0]+=((variant%7)*31)+(index*19); pos[1]+=((variant%5)*23)
    prompt["case_id"]=_opaque_id("train",ordinal,str(source_prompt.get("case_id","missing")),variant)
    messages[1]["content"]=json.dumps(prompt,ensure_ascii=False,separators=(",",":"))
    messages[2]["content"]=json.dumps(answer,ensure_ascii=False,separators=(",",":"))
    return row

def _normalise_ids(rows: list[dict[str,Any]], split: str) -> list[dict[str,Any]]:
    result=[]
    for index,source in enumerate(rows,1):
        row=copy.deepcopy(source); prompt=json.loads(row["messages"][1]["content"])
        prompt["case_id"]=_opaque_id(split,index,str(prompt.get("case_id","missing")))
        row["messages"][1]["content"]=json.dumps(prompt,ensure_ascii=False,separators=(",",":"))
        result.append(row)
    return result

def _read(path: Path) -> list[dict[str,Any]]:
    rows=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if line.strip():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError as exc: raise ValueError(f"{path.name}:{n}: {exc}") from exc
    return rows

def _fingerprint(row: dict[str,Any]) -> str:
    prompt=json.loads(row["messages"][1]["content"]); prompt.pop("case_id",None)
    return json.dumps(prompt,sort_keys=True,ensure_ascii=False,separators=(",",":"))

def _counts(rows: list[dict[str,Any]]) -> tuple[Counter,Counter]:
    categories,codes=Counter(),Counter()
    for row in rows:
        answer=json.loads(row["messages"][2]["content"])
        categories[answer["category"]]+=1; codes[answer["diagnosis_code"]]+=1
    return categories,codes



def _lookup_input_path(value: Any, path: str) -> Any:
    """Read a simple n8n $json path from a supplied input fixture."""
    cleaned = path.replace("?.", ".").replace("[", ".").replace("]", "")
    current = value
    for part in (segment for segment in cleaned.split(".") if segment):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _normalise_legacy_labels(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], Counter]:
    """Correct audited legacy label collisions without changing the legacy held-out eval."""
    output = []
    corrections = Counter()
    for source_row in rows:
        row = copy.deepcopy(source_row)
        messages = row.get("messages", [])
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{split}: expected three chat messages during label normalization")
        prompt = json.loads(messages[1]["content"])
        answer = json.loads(messages[2]["content"])
        old_code = answer.get("diagnosis_code")

        if old_code == "FIELD_MISSING_OR_NULL":
            patch_values = [
                operation.get("value", "")
                for operation in answer.get("patch", [])
                if isinstance(operation, dict) and isinstance(operation.get("value"), str)
            ]
            candidate_paths = []
            for value in patch_values:
                candidate_paths.extend(re.findall(r"\$json\.([A-Za-z0-9_$?.\[\]]+)", value))
            existing_path = next(
                (
                    path for path in reversed(candidate_paths)
                    if _lookup_input_path(prompt.get("input_data", {}), path) is not None
                ),
                None,
            )
            if existing_path:
                answer["diagnosis_code"] = "NESTED_FIELD_PATH_MISMATCH"
                answer["category"] = "data_mapping"
                answer["root_cause"] = (
                    f"The requested value exists at {existing_path.replace('?.', '.')}; "
                    "the original expression reads a different source path. This is a "
                    "nested-path mismatch, not an absent source field."
                )
                corrections[f"{split}:FIELD_MISSING_OR_NULL->NESTED_FIELD_PATH_MISMATCH"] += 1

        elif old_code == "REQUEST_TIMEOUT":
            faulty = next(
                (node for node in prompt.get("workflow", {}).get("nodes", [])
                 if node.get("name") == answer.get("faulty_node")),
                {},
            )
            options = faulty.get("parameters", {}).get("options", {})
            timeout = options.get("timeout") if isinstance(options, dict) else None
            evidence = " ".join([
                str(prompt.get("execution_error", "")),
                str(prompt.get("expected_behavior", "")),
                str(prompt.get("evidence_note", "")),
            ]).lower()
            explicit_sla = (
                isinstance(timeout, (int, float))
                and (
                    "documentedmaxresponsems" in json.dumps(prompt.get("input_data", {})).lower()
                    or any(term in evidence for term in (
                        "endpoint contract allows",
                        "endpoint's supplied response-time contract",
                        "service response-time contract",
                        "documented service response time",
                        "documented maximum response",
                        "explicit endpoint response-time requirement",
                    ))
                )
            )
            if explicit_sla:
                answer["category"] = "node_configuration"
                answer["diagnosis_code"] = "TIMEOUT_TOO_SHORT"
                answer["root_cause"] = (
                    "The HTTP Request node has an explicit numeric timeout below the "
                    "supplied endpoint response-time requirement."
                )
                corrections[f"{split}:REQUEST_TIMEOUT->TIMEOUT_TOO_SHORT"] += 1
            else:
                answer["patch"] = []
                answer["needs_human_review"] = True
                answer["root_cause"] = (
                    "The request timed out, but no verified endpoint latency requirement or "
                    "approved timeout value supports a safe increase. Request human review."
                )
                corrections[f"{split}:REQUEST_TIMEOUT->safe_abstention"] += 1

        elif old_code == "PAIRED_ITEM_NOT_PRESERVED":
            faulty = next(
                (node for node in prompt.get("workflow", {}).get("nodes", [])
                 if node.get("name") == answer.get("faulty_node")),
                {},
            )
            js_code = faulty.get("parameters", {}).get("jsCode", "")
            if "pairedItem" not in js_code:
                answer["category"] = "item_linking"
                answer["diagnosis_code"] = "PAIRED_ITEM_METADATA_MISSING"
                answer["root_cause"] = (
                    "The Code node creates one output for each input but omits pairedItem "
                    "metadata; each output needs the matching source index."
                )
                corrections[f"{split}:PAIRED_ITEM_NOT_PRESERVED->PAIRED_ITEM_METADATA_MISSING"] += 1

        elif old_code == "WRONG_ITEM_SELECTOR":
            input_data = prompt.get("input_data", {})
            singleton_key = next(
                (key for key, value in input_data.items()
                 if isinstance(value, list) and len(value) == 1),
                None,
            )
            if singleton_key:
                second = copy.deepcopy(input_data[singleton_key][0])
                if isinstance(second, dict):
                    for key, value in list(second.items()):
                        if isinstance(value, str):
                            second[key] = value + "-LATER"
                            break
                input_data[singleton_key].append(second)
                input_data["ordering"] = "oldest_first"
                prompt["execution_error"] = (
                    "The lookup returns two records sorted oldest first, but last() selects "
                    "the newest record even though the earliest is required."
                )
                prompt["expected_behavior"] = (
                    "Select the first record from the explicitly oldest-first lookup output."
                )
                answer["root_cause"] = (
                    "The result is explicitly ordered oldest first, but last() selects the "
                    "newest item instead of the required earliest item."
                )
                corrections[f"{split}:WRONG_ITEM_SELECTOR_singleton_repaired"] += 1

        messages[1]["content"] = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
        messages[2]["content"] = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
        output.append(row)
    return output, corrections


def prepare_dataset(source_dir: str|Path, output_dir: str|Path, train_per_code: int=48) -> dict[str,Any]:
    if train_per_code<18: raise ValueError("train_per_code must be at least 18")
    source,out=Path(source_dir),Path(output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True,exist_ok=True)
    taxonomy=json.loads((source/"diagnosis-codes.json").read_text(encoding="utf-8"))
    allowed={entry["code"] for entries in taxonomy["categories"].values() for entry in entries}
    if allowed!={code for codes in CODES_BY_CATEGORY.values() for code in codes}: raise ValueError("taxonomy differs from scorer codebook")
    base_train,valid,evaluation=(_read(source/f"{name}.jsonl") for name in ("train","valid","eval"))
    supplemental_valid_path=source/"supplemental-valid-v0.6.jsonl"
    supplemental_valid=_read(supplemental_valid_path) if supplemental_valid_path.exists() else []
    for index,row in enumerate(supplemental_valid,1):
        messages=row.get("messages")
        if not isinstance(messages,list) or len(messages)!=3:
            raise ValueError(f"{supplemental_valid_path.name}:{index}: expected three chat messages")
        if [message.get("role") for message in messages]!=["system","user","assistant"]:
            raise ValueError(f"{supplemental_valid_path.name}:{index}: roles must be system, user, assistant")
    valid=valid+supplemental_valid
    supplemental_names=[
        "supplemental-train-v0.5.jsonl",
        "supplemental-train-v0.6-mapping.jsonl",
        "supplemental-train-v0.6-linking.jsonl",
        "supplemental-train-v0.6-node-config.jsonl",
        "supplemental-train-v0.6-code.jsonl",
        "supplemental-train-v0.6-api.jsonl",
        "supplemental-train-v0.6-expression.jsonl",
        "supplemental-train-v0.6-workflow.jsonl",
        "supplemental-train-v0.6-credentials.jsonl",
    ]
    supplemental_files=[source/name for name in supplemental_names if (source/name).exists()]
    supplemental=[]
    for path in supplemental_files:
        loaded=_read(path)
        for index,row in enumerate(loaded,1):
            messages=row.get("messages")
            if not isinstance(messages,list) or len(messages)!=3:
                raise ValueError(f"{path.name}:{index}: expected three chat messages")
            if [message.get("role") for message in messages]!=["system","user","assistant"]:
                raise ValueError(f"{path.name}:{index}: roles must be system, user, assistant")
            if not all(isinstance(message.get("content"),str) for message in messages):
                raise ValueError(f"{path.name}:{index}: message content must be text")
        supplemental.extend(loaded)

    base_train,base_corrections=_normalise_legacy_labels(base_train,"train")
    supplemental,supplemental_corrections=_normalise_legacy_labels(supplemental,"supplemental")
    valid,valid_corrections=_normalise_legacy_labels(valid,"valid")
    normalisation_report=Counter()
    normalisation_report.update(base_corrections)
    normalisation_report.update(supplemental_corrections)
    normalisation_report.update(valid_corrections)
    train=base_train+supplemental
    groups=defaultdict(list)
    for row in train:
        code=json.loads(row["messages"][2]["content"]).get("diagnosis_code")
        if code not in allowed: raise ValueError(f"unknown diagnosis code: {code}")
        groups[code].append(row)
    prepared_train=[]; generated=0
    for code in sorted(allowed):
        examples=groups.get(code,[])
        if not examples: raise ValueError(f"no training example for {code}")
        if len(examples)>train_per_code: raise ValueError(f"{code} has {len(examples)}, exceeding target {train_per_code}")
        prepared_train.extend(_normalise_ids(examples,"train"))
        for i in range(train_per_code-len(examples)):
            generated+=1
            prepared_train.append(_make_variant(examples[i%len(examples)],generated,generated))
    random.Random(42).shuffle(prepared_train)
    prepared_valid=_normalise_ids(valid,"valid"); prepared_eval=_normalise_ids(evaluation,"eval")
    splits={"train":prepared_train,"valid":prepared_valid,"eval":prepared_eval}
    seen_ids=set(); seen_prompts=set()
    for split,rows in splits.items():
        for row in rows:
            prompt=json.loads(row["messages"][1]["content"]); cid=prompt.get("case_id","")
            if not re.fullmatch(r"case-[0-9a-f]{16}",cid): raise ValueError(f"{split}: non-opaque case ID")
            if cid in seen_ids: raise ValueError(f"duplicate case ID: {cid}")
            seen_ids.add(cid)
            fp=_fingerprint(row)
            if fp in seen_prompts: raise ValueError(f"duplicate prompt after removing case_id: {cid}")
            seen_prompts.add(fp)
    distribution={}; code_distribution={}
    for split,rows in splits.items():
        cats,codes=_counts(rows)
        distribution[split]={cat:cats.get(cat,0) for cat in sorted(CODES_BY_CATEGORY)}
        code_distribution[split]={code:codes.get(code,0) for code in sorted(allowed)}
        if set(codes)!=allowed: raise ValueError(f"{split}: diagnosis code is missing")
    if set(code_distribution["train"].values())!={train_per_code}: raise AssertionError("training code counts are not balanced")
    source_manifest=json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    total=sum(len(rows) for rows in splits.values())
    manifest={"name":"n8n workflow diagnosis and repair dataset","version":"0.6.0-prepared",
      "provenance":"v0.4 source corpus plus independently authored v0.5/v0.6 supplemental synthetic training scenarios. Audited label collisions are corrected in train/validation; legacy held-out evaluation is left unchanged. Deterministic augmentation is applied only to training.",
      "license":source_manifest.get("license","MIT (repository license)"),"format":"JSONL chat messages: system, user, assistant",
      "counts":{"train":len(prepared_train),"valid":len(prepared_valid),"eval":len(prepared_eval),"total":total},
      "categories":sorted(CODES_BY_CATEGORY),"diagnosis_code_taxonomy":"diagnosis-codes.json",
      "distribution":distribution,"code_distribution":code_distribution,
      "label_normalization":{"corrections":dict(sorted(normalisation_report.items())),"scope":["train","supplemental_train","valid","supplemental_valid"],"heldout_eval_modified":False},
      "supplemental_validation":{"file":supplemental_valid_path.name if supplemental_valid else None,"cases":len(supplemental_valid),"validation_only":True},\n      "supplemental_training":{"files":[path.name for path in supplemental_files],
       "base_source_train_cases":len(base_train),"supplemental_train_cases":len(supplemental),
       "supplemental_cases_are_synthetic":True,"supplemental_cases_are_train_only":True},
      "augmentation":{"source_train_cases":len(train),"generated_train_variants":generated,"target_examples_per_diagnosis_code":train_per_code,
       "validation_augmented":False,"validation_supplemented":bool(supplemental_valid),"evaluation_augmented":False,
       "method":"Consistent node renaming, safe sample substitutions and position variations; prepared training examples are deterministically shuffled.",
       "warning":"Variants preserve the original label/patch; they are not independent new incidents or root-cause patterns."},
      "case_id_policy":"Opaque case-<16 hexadecimal characters>; label-bearing identifiers are rejected.",
      "archived_benchmark":source_manifest.get("archived_benchmark"),
      "evaluation_limitations":"Legacy held-out set has four cases per diagnosis code and remains unchanged; supplemental challenge evaluation is maintained separately.",
      "validation_policy":"Per-split/per-code counts recorded; duplicates checked after removing case_id."}
    for split,rows in splits.items():
        (out/f"{split}.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n" for row in rows),encoding="utf-8")
    (out/"diagnosis-codes.json").write_text(json.dumps(taxonomy,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (out/"DATASET_CARD.md").write_text(f"# Prepared dataset v0.6\n\n- Base source training cases: {len(base_train)}\n- Supplemental authored training cases: {len(supplemental)}\n- Training: {len(prepared_train)} ({train_per_code} per code)\n- Validation: {len(prepared_valid)} (unchanged)\n- Held-out evaluation: {len(prepared_eval)} (unchanged)\n- Generated training variants: {generated}\n\nTraining variants are semantic-preserving transformations, not newly observed incidents. The held-out set is not augmented; case IDs are opaque.\n",encoding="utf-8")
    if (source/"archive").exists(): shutil.copytree(source/"archive",out/"archive",dirs_exist_ok=True)
    return {"status":"prepared","version":manifest["version"],"counts":manifest["counts"],"base_source_train_cases":len(base_train),"supplemental_train_cases":len(supplemental),"train_per_diagnosis_code":train_per_code,"generated_training_variants":generated,"unique_case_ids":len(seen_ids),"unique_prompt_fingerprints":len(seen_prompts),"label_normalization":dict(sorted(normalisation_report.items())),"supplemental_validation_cases":len(supplemental_valid),"note":"Training/validation labels were audited; legacy held-out evaluation was untouched."}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--source-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--train-per-code",type=int,default=48)
    args=parser.parse_args(); print(json.dumps(prepare_dataset(args.source_dir,args.output_dir,args.train_per_code),indent=2))
