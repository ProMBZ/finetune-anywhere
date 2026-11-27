# n8n workflow diagnosis and repair

Fine-tune Qwen3-1.7B to classify n8n failures, identify evidence-supported root causes and propose minimal JSON Patch repairs.

## Dataset v0.5 training corpus

The v0.4 source corpus contains 640 training, 128 validation and 192 held-out evaluation cases. v0.5 adds **216 independently authored synthetic training scenarios** in `data/supplemental-train-v0.5.jsonl`: 72 data-mapping cases and 36 each for item linking, node configuration, Code-node behavior and execution/API failures. Colab retains all 856 source training cases and adds deterministic variants to reach **1,920 training examples (40 per diagnosis code)**. The new scenarios are synthetic, not observed incidents; generated variants are not independent root-cause cases. Validation and held-out evaluation remain unchanged and are never augmented.

| Prepared split | Examples | Distribution |
|---|---:|---|
| Training | 1,920 | 40 per code; 240 per category |
| Validation | 128 | 16 per category |
| Held-out evaluation | 192 | 24 per category; 4 per code |
| **Total** | **2,240** | |

There are 48 canonical diagnosis codes, six per category, described in `data/diagnosis-codes.json`.

## Critical v0.3 benchmark issue fixed

Earlier validation/evaluation `case_id` values contained their target category and diagnosis-code labels. Because the ID is part of the user prompt, that leaked the expected answer. v0.4 uses opaque IDs and the validator rejects identifiers containing category/code names. It also checks prompt uniqueness after removing the ID field.

## Training augmentation

v0.5 retains the original 640 training cases plus 216 new train-only scenarios. Data-mapping additions cover missing/null fields, zero-based array indexing, type conversion, falsy-value defaults, string-to-boolean parsing and incorrect nested paths. The other additions target ambiguous item selection and lineage, HTTP configuration, Code-node output/mode errors, and API payload/retry/timeout/outage distinctions.

The remaining **1,064 generated examples** are controlled variants: consistent node renaming across expressions/connections/patches, safe identifier/domain substitutions and position changes. These variants preserve the original label and patch; they are not independent incidents. The 128 validation and 192 held-out evaluation cases remain unchanged apart from opaque-ID normalization.

## Evaluation

- Malformed outputs count as wrong in full metric denominators.
- Diagnosis-code reporting includes exact accuracy, macro per-code accuracy, code validity, conditional accuracy, per-code counts and a confusion matrix.
- Human-review decision agreement is separated from safe-abstention precision and recall.
- Gold-patch match is strict and can penalize alternative valid repairs.
- Patch applicability checks JSON Patch against synthetic fixtures; it does not prove execution inside n8n.
- Both base and fine-tuned models receive the same retrieved documentation excerpts and held-out prompts.

The notebook uses TRL 0.22.2's conversational prompt/completion format with `completion_only_loss=True`, so loss is applied to assistant diagnosis/repair targets without relying on assistant-token masks in Qwen3's stock chat template. The best adapter checkpoint is selected by validation loss.

## Fine-tuning configuration

Qwen3-1.7B; 4-bit QLoRA; LoRA rank 16 / alpha 32; learning rate 1.5e-5; two epochs; 2048-token context; batch size 1 with gradient accumulation 8 (480 optimizer steps). The best checkpoint is selected by validation loss. GGUF/Ollama export and download remain disabled.

## Colab

[Open the updated n8n repair Colab notebook](https://colab.research.google.com/github/ProMBZ/finetune-anywhere/blob/project-01-n8n-workflow-repair/paths/colab/n8n-workflow-repair.ipynb).

Select a T4 GPU when available and run top to bottom. It prepares the balanced dataset, validates patches, runs project tests, fetches official n8n documentation, scores the base model, fine-tunes Qwen3 and evaluates all 192 held-out prompts. The evidence ZIP records data preparation, hashes, settings and complete base/fine-tuned outputs.

Keep `eval.jsonl` untouched. Do not train against held-out references or tune repeatedly against individual test misses.
