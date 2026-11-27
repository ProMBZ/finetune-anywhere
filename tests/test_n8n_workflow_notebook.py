"""Guard the Qwen3 n8n notebook's dataset and loss-mask contract."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return json.loads((ROOT / "paths/colab/n8n-workflow-repair.ipynb").read_text(encoding="utf-8"))


def cell_text(book, index):
    return "".join(book["cells"][index].get("source", []))


def code_cells(book):
    return [
        "".join(cell.get("source", []))
        for cell in book["cells"]
        if cell.get("cell_type") == "code"
    ]


def unique_code_cell_containing(book, marker):
    matches = [source for source in code_cells(book) if marker in source]
    assert len(matches) == 1, f"Expected exactly one code cell containing {marker!r}, found {len(matches)}"
    return matches[0]


def test_notebook_uses_completion_only_prompt_completion_training():
    book = notebook()
    install = cell_text(book, 4)
    dataset = cell_text(book, 16)
    training = cell_text(book, 20)
    report = next(
        cell_text(book, index)
        for index, cell in enumerate(book["cells"])
        if "'training_dataset_format': 'conversational_prompt_completion'" in cell_text(book, index)
    )

    assert "trl==0.22.2" in install
    assert "def as_prompt_completion(row):" in dataset
    assert "'prompt': messages[:-1]" in dataset
    assert "'completion': [messages[-1]]" in dataset
    assert "completion_only_loss=True" in training
    assert "assistant_only_loss=True" not in training
    assert "'completion_only_loss': True" in report
    assert "'assistant_only_loss': False" in report


def test_step_8_preview_uses_prompt_completion_columns():
    dataset = cell_text(notebook(), 16)
    assert "train_data[0]['text']" not in dataset
    assert "train_data[0]['prompt'] + train_data[0]['completion']" in dataset


def test_heldout_reference_is_not_in_generation_prompt():
    dataset = cell_text(notebook(), 16)
    evaluate = cell_text(notebook(), 18)
    assert "eval_prompts = [row['messages'][:-1] for row in augmented['eval']]" in dataset
    assert "answer(messages)" in evaluate


def test_gguf_export_remains_disabled():
    book = notebook()
    export = unique_code_cell_containing(book, "EXPORT_GGUF")
    download = unique_code_cell_containing(book, "DOWNLOAD_GGUF")

    # Locate settings by content, not absolute notebook cell indexes: adding a
    # Markdown section must not make this safety regression test fail.
    assert any(line.strip() == "EXPORT_GGUF = False" for line in export.splitlines())
    assert "if EXPORT_GGUF:" in export
    assert "model.save_pretrained_gguf(" in export

    assert any(line.strip() == "DOWNLOAD_GGUF = False" for line in download.splitlines())
    assert "if DOWNLOAD_GGUF:" in download
    assert "files.download(str(file))" in download


def test_settings_cell_defines_all_step_4_and_training_inputs():
    book = notebook()
    settings = cell_text(book, 6)
    drive_backup = cell_text(book, 8)

    assert 'RECIPE = "n8n-workflow-repair"' in settings
    assert 'DATA_DIR = recipe_dir / "data"' in settings
    assert 'config = yaml.safe_load(' in settings
    assert 'registry = yaml.safe_load(' in settings
    assert 'MODEL_ALIAS = config["model"]' in settings
    assert 'model_entry = registry["models"][MODEL_ALIAS]' in settings
    assert 'RUN_NAME = RECIPE +' in settings
    assert 'OUTPUT = OUTPUT_ROOT / RUN_NAME' in drive_backup

def test_step_8_uses_compact_codebook_and_adaptive_context_budget():
    dataset = cell_text(notebook(), 16)

    assert "COMPACT_CODE_DEFINITIONS" in dataset
    assert "missing_definitions = allowed_codes - set(COMPACT_CODE_DEFINITIONS)" in dataset
    assert "retrieval_attempts = [(2, 280), (1, 120), (1, 60), (0, 0)]" in dataset
    assert "token_length(row['messages'], add_generation_prompt=False)" in dataset
    assert "add_generation_prompt=(split == 'eval')" not in dataset
    assert "Inspect the longest source prompt/target rather than silently truncating it." in dataset

def test_step_8_replaces_duplicate_taxonomy_instead_of_appending_it():
    dataset = cell_text(notebook(), 16)
    assert "BASE_SYSTEM_INSTRUCTION = (" in dataset
    assert "row['messages'][0]['content'] = BASE_SYSTEM_INSTRUCTION" in dataset
    assert "row['messages'][0]['content'] += " not in dataset
    assert "Combined system instruction characters:" in dataset


def test_notebook_uses_v05_supplemental_training_and_records_its_hash():
    book = notebook()
    dataset = cell_text(book, 16)
    report = next(
        cell_text(book, index)
        for index, cell in enumerate(book["cells"])
        if "'dataset_preparation': preparation_report" in cell_text(book, index)
    )
    config = (ROOT / "recipes/n8n-workflow-repair/config.yaml").read_text(encoding="utf-8")

    assert "prepared-dataset-v0.5" in dataset
    assert "train_per_code=40" in dataset
    assert "learning_rate: 0.000015" in config
    assert "epochs: 2" in config
    assert "'training_data_version': preparation_report['version']" in report
    assert "'supplemental_train_sha256'" in report

