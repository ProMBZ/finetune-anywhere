# n8n workflow diagnosis and repair

Fine-tune Qwen3-1.7B to classify n8n failures, identify evidence-supported root causes and propose minimal JSON Patch repairs.

## Dataset v0.4

The committed source corpus contains 640 training, 128 validation and 192 held-out evaluation cases. Colab prepares a balanced training set of **1,536 cases (32 per diagnosis code)** using 896 deterministic, meaning-preserving variants. Validation and evaluation are not augmented.

| Prepared split | Cases | Distribution |
|---|---:|---|
| Training | 1,536 | 32 per code; 192 per category |
| Validation | 128 | 16 per category |
| Held-out evaluation | 192 | 24 per category; 4 per code |
| **Total** | **1,856** | |

There are 48 canonical diagnosis codes, six per category, described in `data/diagnosis-codes.json`.

## Critical v0.3 benchmark issue fixed

Earlier validation/evaluation `case_id` values contained their target category and diagnosis-code labels. Because the ID is part of the user prompt, that leaked the expected answer. v0.4 uses opaque IDs and the validator rejects identifiers containing category/code names. It also checks prompt uniqueness after removing the ID field.

## Training augmentation

The preparation script keeps all 640 source cases and adds 896 controlled variants. It consistently renames workflow nodes throughout expressions, connections and patches; changes safe sample identifiers/domains; and varies positions where present. The original root-cause label and patch are preserved.

These are augmentation variants, **not 896 independent new incidents**. They balance labels and expose the model to different surface forms; they do not add 896 new root-cause patterns. The 192-case held-out test set is kept unchanged apart from opaque IDs.

## Evaluation

- Malformed outputs count as wrong in full metric denominators.
- Diagnosis-code reporting includes exact accuracy, macro per-code accuracy, code validity, conditional accuracy, per-code counts and a confusion matrix.
- Human-review decision agreement is separated from safe-abstention precision and recall.
- Gold-patch match is strict and can penalize alternative valid repairs.
- Patch applicability checks JSON Patch against synthetic fixtures; it does not prove execution inside n8n.
- Both base and fine-tuned models receive the same retrieved documentation excerpts and held-out prompts.

The notebook trains on assistant responses only using TRL 0.22.2's conversational dataset support. The best adapter checkpoint is selected by validation loss.

## Fine-tuning configuration

Qwen3-1.7B; 4-bit QLoRA; LoRA rank 16 / alpha 32; learning rate 2e-5; three epochs; 2048-token context; batch size 1 with gradient accumulation 8. GGUF/Ollama export is disabled.

## Colab

[Open the updated n8n repair Colab notebook](https://colab.research.google.com/github/ProMBZ/finetune-anywhere/blob/project-01-n8n-workflow-repair/paths/colab/n8n-workflow-repair.ipynb).

Select a T4 GPU when available and run top to bottom. It prepares the balanced dataset, validates patches, runs project tests, fetches official n8n documentation, scores the base model, fine-tunes Qwen3 and evaluates all 192 held-out prompts. The evidence ZIP records data preparation, hashes, settings and complete base/fine-tuned outputs.

Keep `eval.jsonl` untouched. Do not train against held-out references or tune repeatedly against individual test misses.
