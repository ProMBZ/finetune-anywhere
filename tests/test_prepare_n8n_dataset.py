import json
import sys
from collections import Counter
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT/"scripts"))
from prepare_n8n_dataset import prepare_dataset
from validate_n8n_dataset import validate_dataset

def test_preparation_balances_training_but_keeps_heldout_splits(tmp_path):
    source=REPO_ROOT/"recipes"/"n8n-workflow-repair"/"data"; output=tmp_path/"prepared"
    report=prepare_dataset(source,output,train_per_code=40)
    assert report["status"]=="prepared"
    assert report["counts"]=={"train":1920,"valid":128,"eval":192,"total":2240}
    checked=validate_dataset(output)
    assert checked["status"]=="passed" and checked["unique_case_ids"]==2240
    assert checked["unique_case_fingerprints"]==2240
    manifest=__import__("json").loads((output/"manifest.json").read_text())
    assert set(manifest["code_distribution"]["train"].values())=={40}
    assert manifest["counts"]["eval"]==192
    assert manifest["supplemental_training"]["base_source_train_cases"]==640
    assert manifest["supplemental_training"]["supplemental_train_cases"]==216
    assert manifest["supplemental_training"]["supplemental_cases_are_train_only"] is True

def test_preparation_does_not_overwrite_existing_run(tmp_path):
    source=REPO_ROOT/"recipes"/"n8n-workflow-repair"/"data"; output=tmp_path/"prepared"; output.mkdir()
    (output/"keep.txt").write_text("keep")
    try: prepare_dataset(source,output,train_per_code=18)
    except FileExistsError: pass
    else: raise AssertionError("must not overwrite existing experiment data")

def test_node_renaming_does_not_rewrite_new_replacement_text():
    from prepare_n8n_dataset import _rewrite_tree
    value={"nodes":["HTTP Request","HTTP"],"expression":"$('HTTP Request').item + $('HTTP').item"}
    renamed=_rewrite_tree(value,{"HTTP Request":"HTTP Request [variant 1]","HTTP":"HTTP [variant 2]"})
    assert renamed["nodes"]==["HTTP Request [variant 1]","HTTP [variant 2]"]
    assert renamed["expression"]=="$('HTTP Request [variant 1]').item + $('HTTP [variant 2]').item"


def test_v05_supplemental_cases_expand_only_weak_categories():
    source=REPO_ROOT/"recipes"/"n8n-workflow-repair"/"data"
    rows=[json.loads(line) for line in (source/"supplemental-train-v0.5.jsonl").read_text().splitlines() if line]
    assert len(rows)==216
    category_counts=Counter()
    code_counts=Counter()
    for row in rows:
        messages=row["messages"]
        assert [message["role"] for message in messages]==["system","user","assistant"]
        answer=json.loads(messages[2]["content"])
        category_counts[answer["category"]]+=1
        code_counts[answer["diagnosis_code"]]+=1
    assert category_counts=={
        "data_mapping":72,
        "item_linking":36,
        "node_configuration":36,
        "code_node":36,
        "execution_api":36,
    }
    assert all(code_counts[code]==12 for code in (
        "FIELD_MISSING_OR_NULL","ARRAY_INDEX_INCORRECT","TYPE_COERCION_REQUIRED",
        "DEFAULT_OPERATOR_LOSES_FALSY_VALUE","BOOLEAN_STRING_COERCION","NESTED_FIELD_PATH_MISMATCH"
    ))
    assert len(code_counts)==30

