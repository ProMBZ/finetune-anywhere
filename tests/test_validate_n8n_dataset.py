import sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT/"scripts"))
from validate_n8n_dataset import case_id_is_opaque,node_settings_are_well_formed,validate_dataset

def test_complete_source_recipe_dataset_is_valid():
    report=validate_dataset(REPO_ROOT/"recipes"/"n8n-workflow-repair"/"data")
    assert report["status"]=="passed"
    assert report["counts"]=={"train":640,"valid":128,"eval":192}
    assert report["total"]==960 and report["unique_prompts"]==960
    assert report["unique_case_ids"]==960 and report["unique_case_fingerprints"]==960
    assert report["canonical_diagnosis_codes"]==48 and report["patches_checked"]>0

def test_case_ids_are_opaque_and_reject_embedded_answers():
    assert case_id_is_opaque("case-0123456789abcdef")
    assert not case_id_is_opaque("v03_eval_expression_expression_prefix_missing_5000")
    assert not case_id_is_opaque("case-expression00000000")

def test_retry_settings_are_top_level_node_properties():
    good={"nodes":[{"name":"HTTP","parameters":{"url":"https://api.example.test"},"retryOnFail":True,"maxTries":3,"waitBetweenTries":5000}],"connections":{}}
    bad={"nodes":[{"name":"HTTP","parameters":{"url":"https://api.example.test","retryOnFail":True,"maxTries":3,"waitBetweenTries":5000}}],"connections":{}}
    assert node_settings_are_well_formed(good) and not node_settings_are_well_formed(bad)
