from pathlib import Path
from validate_n8n_dataset import validate_dataset

def test_complete_n8n_recipe_dataset_is_valid():
    repo_root = Path(__file__).resolve().parents[1]
    report = validate_dataset(repo_root / "recipes" / "n8n-workflow-repair" / "data")
    assert report["status"] == "passed"
    assert report["counts"] == {"train": 200, "valid": 50, "eval": 50}
    assert report["total"] == 300
    assert report["unique_prompts"] == 300
    assert report["patches_checked"] > 0
