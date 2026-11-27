import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from prepare_n8n_dataset import prepare_dataset
from score_n8n_repairs import apply_patch
from validate_n8n_dataset import validate_dataset

DATA_DIR = REPO_ROOT / "recipes" / "n8n-workflow-repair" / "data"
SUPPLEMENTAL_FILES = [
    "supplemental-train-v0.5.jsonl",
    "supplemental-train-v0.6-mapping.jsonl",
    "supplemental-train-v0.6-linking.jsonl",
    "supplemental-train-v0.6-node-config.jsonl",
    "supplemental-train-v0.6-code.jsonl",
    "supplemental-train-v0.6-api.jsonl",
    "supplemental-train-v0.6-expression.jsonl",
    "supplemental-train-v0.6-workflow.jsonl",
    "supplemental-train-v0.6-credentials.jsonl",
    "supplemental-train-v0.7-mapping.jsonl",
    "supplemental-train-v0.7-linking.jsonl",
    "supplemental-train-v0.7-node-config.jsonl",
    "supplemental-train-v0.7-code.jsonl",
    "supplemental-train-v0.7-api.jsonl",
    "supplemental-train-v0.7-expression.jsonl",
    "supplemental-train-v0.7-workflow.jsonl",
    "supplemental-train-v0.7-credentials.jsonl",
]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_v07_preparation_balances_training_and_preserves_heldout_splits(tmp_path):
    output = tmp_path / "prepared"
    report = prepare_dataset(DATA_DIR, output)
    assert report["status"] == "prepared"
    assert report["version"] == "0.7.0-prepared"
    assert report["counts"] == {
        "train": 3072,
        "valid": 156,
        "eval": 192,
        "total": 3420,
    }

    checked = validate_dataset(output)
    assert checked["status"] == "passed"
    assert checked["unique_case_ids"] == 3420
    assert checked["unique_case_fingerprints"] == 3420

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["code_distribution"]["train"].values()) == {64}
    assert manifest["counts"]["eval"] == 192
    assert manifest["supplemental_training"]["base_source_train_cases"] == 640
    assert manifest["supplemental_training"]["supplemental_train_cases"] == 888
    assert len(manifest["supplemental_training"]["files"]) == 17
    assert manifest["supplemental_validation"]["cases"] == 28
    assert manifest["supplemental_validation"]["validation_only"] is True
    assert manifest["label_normalization"]["heldout_eval_modified"] is False
    assert manifest["augmentation"]["evaluation_augmented"] is False
    assert manifest["augmentation"]["validation_augmented"] is False


def test_v07_supplemental_training_coverage_and_schema():
    category_counts = Counter()
    diagnosis_counts = Counter()
    case_ids = set()
    fingerprints = set()

    for name in SUPPLEMENTAL_FILES:
        rows = read_jsonl(DATA_DIR / name)
        for row in rows:
            messages = row["messages"]
            assert [message["role"] for message in messages] == ["system", "user", "assistant"]
            prompt = json.loads(messages[1]["content"])
            answer = json.loads(messages[2]["content"])
            assert set(answer) == {
                "category", "diagnosis_code", "faulty_node", "root_cause", "patch",
                "needs_human_review",
            }
            assert prompt["workflow"]["nodes"]
            assert any(node["name"] == answer["faulty_node"] for node in prompt["workflow"]["nodes"])
            assert isinstance(answer["patch"], list)
            assert isinstance(answer["needs_human_review"], bool)
            assert prompt["case_id"].startswith("case-")
            assert prompt["case_id"] not in case_ids
            case_ids.add(prompt["case_id"])
            fingerprint_prompt = dict(prompt)
            fingerprint_prompt.pop("case_id", None)
            fingerprint = json.dumps(fingerprint_prompt, sort_keys=True, ensure_ascii=False)
            assert fingerprint not in fingerprints
            fingerprints.add(fingerprint)
            category_counts[answer["category"]] += 1
            diagnosis_counts[answer["diagnosis_code"]] += 1

    assert sum(category_counts.values()) == 888
    assert category_counts == {
        "data_mapping": 228,
        "item_linking": 144,
        "node_configuration": 126,
        "code_node": 108,
        "execution_api": 102,
        "expression": 60,
        "workflow_structure": 60,
        "credentials": 60,
    }
    assert all(diagnosis_counts[code] == 38 for code in (
        "FIELD_MISSING_OR_NULL", "ARRAY_INDEX_INCORRECT", "TYPE_COERCION_REQUIRED",
        "DEFAULT_OPERATOR_LOSES_FALSY_VALUE", "BOOLEAN_STRING_COERCION",
        "NESTED_FIELD_PATH_MISMATCH",
    ))
    assert len(diagnosis_counts) == 48


def test_validation_supplement_covers_timeout_abstention_and_paired_item_index():
    rows = read_jsonl(DATA_DIR / "supplemental-valid-v0.6.jsonl")
    assert len(rows) == 4
    counts = Counter()
    for row in rows:
        answer = json.loads(row["messages"][-1]["content"])
        counts[answer["diagnosis_code"]] += 1
        if answer["diagnosis_code"] == "REQUEST_TIMEOUT":
            assert answer["patch"] == []
            assert answer["needs_human_review"] is True
        else:
            assert answer["diagnosis_code"] == "PAIRED_ITEM_NOT_PRESERVED"
            assert answer["patch"]
            assert answer["needs_human_review"] is False
    assert counts == {"REQUEST_TIMEOUT": 2, "PAIRED_ITEM_NOT_PRESERVED": 2}




def test_v07_validation_supplement_is_separate_and_covers_weak_boundaries():
    rows = read_jsonl(DATA_DIR / "supplemental-valid-v0.7.jsonl")
    assert len(rows) == 24
    counts = Counter()
    fingerprints = set()
    for row in rows:
        prompt = json.loads(row["messages"][1]["content"])
        answer = json.loads(row["messages"][2]["content"])
        counts[(answer["category"], answer["diagnosis_code"])] += 1
        assert answer["category"] in {"data_mapping", "item_linking"}
        assert any(node["name"] == answer["faulty_node"] for node in prompt["workflow"]["nodes"])
        without_id = dict(prompt)
        without_id.pop("case_id", None)
        fingerprint = json.dumps(without_id, sort_keys=True, ensure_ascii=False)
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)
        if answer["needs_human_review"]:
            assert answer["patch"] == []
    assert len(counts) == 12
    assert set(counts.values()) == {2}


def test_v07_mapping_supplement_has_fourteen_diverse_cases_per_code():
    rows = read_jsonl(DATA_DIR / "supplemental-train-v0.7-mapping.jsonl")
    assert len(rows) == 84
    counts = Counter()
    distinct_inputs = defaultdict(set)
    for row in rows:
        prompt = json.loads(row["messages"][1]["content"])
        answer = json.loads(row["messages"][2]["content"])
        assert answer["category"] == "data_mapping"
        counts[answer["diagnosis_code"]] += 1
        distinct_inputs[answer["diagnosis_code"]].add(
            json.dumps(prompt["input_data"], sort_keys=True, ensure_ascii=False)
        )
    assert len(counts) == 6
    assert set(counts.values()) == {14}
    assert all(len(values) >= 10 for values in distinct_inputs.values())


def test_challenge_eval_has_one_heldout_case_per_canonical_code():
    rows = read_jsonl(DATA_DIR / "challenge-eval-v0.6.jsonl")
    assert len(rows) == 48
    counts = Counter()
    fingerprints = set()
    for row in rows:
        prompt = json.loads(row["messages"][1]["content"])
        answer = json.loads(row["messages"][2]["content"])
        counts[answer["diagnosis_code"]] += 1
        assert prompt.get("execution_error", "").strip()
        assert prompt.get("expected_behavior", "").strip()
        assert answer.get("faulty_node", "").strip()
        assert answer.get("root_cause", "").strip()
        assert isinstance(answer.get("patch"), list)
        assert any(node["name"] == answer["faulty_node"] for node in prompt["workflow"]["nodes"])
        if answer["diagnosis_code"] in {
            "REQUEST_TIMEOUT", "UPSTREAM_SERVICE_UNAVAILABLE", "DNS_OR_CONNECTION_FAILURE",
            "ITEM_LINEAGE_UNAVAILABLE", "AUTHENTICATION_REJECTED", "PERMISSION_INSUFFICIENT",
            "CREDENTIAL_REFERENCE_MISSING", "OAUTH_CREDENTIAL_EXPIRED",
            "API_KEY_REVOKED_OR_MISSING", "SECRET_VALUE_REDACTED",
        }:
            assert answer["patch"] == []
            assert answer["needs_human_review"] is True
        prompt.pop("case_id", None)
        fingerprint = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
        assert fingerprint not in fingerprints
        fingerprints.add(fingerprint)
    assert len(counts) == 48
    assert set(counts.values()) == {1}



def test_v07_if_branch_repairs_change_the_misrouted_outputs():
    rows = read_jsonl(DATA_DIR / "supplemental-train-v0.7-workflow.jsonl")
    cases = 0
    for row in rows:
        prompt = json.loads(row["messages"][1]["content"])
        answer = json.loads(row["messages"][2]["content"])
        if answer["diagnosis_code"] != "IF_BRANCH_MISROUTED":
            continue
        cases += 1
        before = prompt["workflow"]["connections"][answer["faulty_node"]]["main"]
        repaired = apply_patch(prompt["workflow"], answer["patch"])
        after = repaired["connections"][answer["faulty_node"]]["main"]
        assert before != after, "IF branch repair must not be a no-op"
        expected = prompt["expected_behavior"].removeprefix("Route true to ").removesuffix(".")
        true_target, false_target = expected.split(" and false to ", 1)
        assert after[0][0]["node"] == true_target
        assert after[1][0]["node"] == false_target
    assert cases == 4



def test_preparation_does_not_overwrite_existing_run(tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    try:
        prepare_dataset(DATA_DIR, output, train_per_code=64)
    except FileExistsError:
        pass
    else:
        raise AssertionError("must not overwrite existing experiment data")


def test_node_renaming_does_not_rewrite_new_replacement_text():
    from prepare_n8n_dataset import _rewrite_tree

    value = {
        "nodes": ["HTTP Request", "HTTP"],
        "expression": "$('HTTP Request').item + $('HTTP').item",
    }
    renamed = _rewrite_tree(
        value,
        {
            "HTTP Request": "HTTP Request [variant 1]",
            "HTTP": "HTTP [variant 2]",
        },
    )
    assert renamed["nodes"] == ["HTTP Request [variant 1]", "HTTP [variant 2]"]
    assert renamed["expression"] == (
        "$('HTTP Request [variant 1]').item + $('HTTP [variant 2]').item"
    )
