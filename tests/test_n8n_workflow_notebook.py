"""Guard the Qwen3 n8n notebook's dataset and loss-mask contract."""
import json
import re
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
    assert re.search(r"(?m)^EXPORT_GGUF\\s*=\\s*False\\s*$", export)
    assert "if EXPORT_GGUF:" in export
    assert "model.save_pretrained_gguf(" in export

    assert re.search(r"(?m)^DOWNLOAD_GGUF\\s*=\\s*False\\s*$", download)
    assert "if DOWNLOAD_GGUF:" in download
    assert "files.download(str(file))" in download
