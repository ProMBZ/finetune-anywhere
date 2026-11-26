import sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO_ROOT/"scripts"))
from validate_n8n_dataset import node_settings_are_well_formed,validate_dataset  # noqa: E402

def test_complete_v03_recipe_dataset_is_valid():
    report=validate_dataset(REPO_ROOT/"recipes"/"n8n-workflow-repair"/"data")
    assert report["status"]=="passed"
    assert report["counts"]=={"train":640,"valid":128,"eval":192}
    assert report["total"]==960
    assert report["unique_prompts"]==960
    assert report["unique_case_ids"]==960
    assert report["canonical_diagnosis_codes"]==48
    assert report["patches_checked"]>0
    assert all(report["codes"][split] for split in ("train","valid","eval"))
    assert all(len(report["codes"][split]) == 48 for split in ("train","valid","eval"))

def test_retry_settings_are_top_level_node_properties():
    good={"nodes":[{"name":"HTTP","parameters":{"url":"https://api.example.test"},
                   "retryOnFail":True,"maxTries":3,"waitBetweenTries":5000}],"connections":{}}
    bad={"nodes":[{"name":"HTTP","parameters":{"url":"https://api.example.test",
                "retryOnFail":True,"maxTries":3,"waitBetweenTries":5000}}],"connections":{}}
    assert node_settings_are_well_formed(good)
    assert not node_settings_are_well_formed(bad)
