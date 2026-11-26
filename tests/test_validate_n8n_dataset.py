import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_n8n_dataset import validate_dataset  # noqa: E402


def test_complete_n8n_recipe_dataset_is_valid():
    report = validate_dataset(REPO_ROOT / "recipes" / "n8n-workflow-repair" / "data")
    assert report["status"] == "passed"
    assert report["counts"] == {"train": 200, "valid": 50, "eval": 50}
    assert report["total"] == 300
    assert report["unique_prompts"] == 300
    assert report["unique_case_ids"] == 300
    assert report["patches_checked"] > 0
