"""Regression checks for the Kaggle v0.7 n8n workflow repair notebook."""
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "paths/kaggle/n8n-workflow-repair.ipynb"


def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def cell_source(book, index):
    return "".join(book["cells"][index].get("source", []))


def test_kaggle_notebook_is_platform_port_not_a_recipe_change():
    book = notebook()
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in book["cells"]
        if cell.get("cell_type") == "code"
    )
    assert 'ROOT = Path("/kaggle/working/finetune-anywhere")' in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "from google.colab" not in source
    assert "google.colab" not in source
    assert "/content/" not in source
    assert "SAVE_TO_DRIVE" not in source
    assert "EXPORT_GGUF = False" in source


def test_kaggle_notebook_keeps_original_training_contract():
    book = notebook()
    recipe = yaml.safe_load(
        (ROOT / "recipes/n8n-workflow-repair/config.yaml").read_text(encoding="utf-8")
    )
    install = cell_source(book, 4)
    dataset = cell_source(book, 16)
    training = cell_source(book, 20)

    assert recipe["model"] == "starter"
    assert recipe["rank"] == 16
    assert recipe["alpha"] == 32
    assert recipe["learning_rate"] == 1e-5
    assert recipe["epochs"] == 2
    assert recipe["max_seq_length"] == 2048
    assert recipe["batch_size"] == 1
    assert recipe["gradient_accumulation_steps"] == 8

    assert "transformers==4.57.6" in install
    assert "trl==0.22.2" in install
    assert "completion_only_loss=True" in training
    assert "num_train_epochs=config['epochs']" in training
    assert "per_device_train_batch_size=config['batch_size']" in training
    assert "gradient_accumulation_steps=config['gradient_accumulation_steps']" in training
    assert "train_per_code=64" in dataset
    assert "challenge-eval-v0.6.jsonl" in dataset


def test_kaggle_outputs_are_isolated_and_downloadable():
    book = notebook()
    setup = cell_source(book, 8)
    artifacts = cell_source(book, 32)
    assert "Path('/kaggle/working/finetune-anywhere-runs')" in setup
    assert "mkdir(parents=True, exist_ok=False)" in setup
    assert "run-evidence.zip" in artifacts
    assert "adapter.zip" in artifacts
    assert "FileLink" in artifacts
    assert "from google.colab" not in artifacts
