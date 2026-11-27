# n8n workflow diagnosis and repair

Fine-tune Qwen3-1.7B to classify n8n failures, identify evidence-supported root causes and propose minimal JSON Patch repairs.

## Dataset v0.6 training and evaluation

The v0.4 source corpus contains 640 training, 128 validation and 192 held-out evaluation cases. v0.5/v0.6 add **540 independently authored synthetic training scenarios** across all eight categories. Data mapping receives 144 new cases; item linking and Code-node behavior receive 72 each; node configuration receives 78 and execution/API failures receive 66; expression, workflow structure and credentials receive 36 each. Colab audits legacy label collisions, retains the 1,180 source training cases and balances the result to **2,304 training examples (48 per diagnosis code)**. Deterministic variants are generated only to fill remaining per-code slots and are not independent incidents.

| Prepared split | Examples | Distribution |
|---|---:|---|
| Training | 2,304 | 48 per code; 288 per category |
| Validation | 132 | 128 base cases plus 2 timeout-abstention and 2 paired-item index cases |
| Held-out evaluation | 192 | 24 per category; 4 per code |
| Legacy prepared total | 2,628 | Training + validation + original held-out evaluation |

There are 48 canonical diagnosis codes, six per category, described in `data/diagnosis-codes.json`.

## Critical v0.3 benchmark issue fixed

Earlier validation/evaluation `case_id` values contained their target category and diagnosis-code labels. Because the ID is part of the user prompt, that leaked the expected answer. v0.4 uses opaque IDs and the validator rejects identifiers containing category/code names. It also checks prompt uniqueness after removing the ID field.

## Training augmentation

v0.6 retains all original source cases, adds 540 train-only scenarios and corrects audited label collisions in training/validation. In particular, a value at a different nested path is not labelled missing; missing pairedItem metadata is distinguished from an incorrect pairedItem index; and an unexplained request timeout is taught as a safe abstention rather than a guessed timeout increase. Four additional validation cases teach two boundaries: unexplained timeouts must abstain, and incorrect pairedItem indexes must be corrected without confusing them with missing metadata.

The pipeline fills remaining per-code slots with controlled node-renaming and safe identifier/domain variations, then deterministically shuffles the training rows. These variants preserve their source label and patch; they are not independent incidents. The original 192-case held-out evaluation is not modified. A separate **48-case challenge evaluation** has one fresh synthetic case per diagnosis code and is never used for training or validation.

## Evaluation

- Malformed outputs count as wrong in full metric denominators.
- Diagnosis-code reporting includes exact accuracy, macro per-code accuracy, code validity, conditional accuracy, per-code counts and a confusion matrix.
- Human-review decision agreement is separated from safe-abstention precision and recall.
- Gold-patch match is strict and can penalize alternative valid repairs.
- Patch applicability checks JSON Patch against synthetic fixtures; it does not prove execution inside n8n.
- Both base and fine-tuned models receive the same retrieved documentation excerpts and generation settings. The separate challenge set has one case per diagnosis code, so it is a broad smoke test, not a precise category-level statistical estimate.

The notebook uses TRL 0.22.2's conversational prompt/completion format with `completion_only_loss=True`, so loss is applied to assistant diagnosis/repair targets without relying on assistant-token masks in Qwen3's stock chat template. The best adapter checkpoint is selected by validation loss.

## Fine-tuning configuration

Qwen3-1.7B; 4-bit QLoRA; LoRA rank 16 / alpha 32; learning rate 1.5e-5; two epochs; 2048-token context; batch size 1 with gradient accumulation 8 (576 optimizer steps). The best checkpoint is selected by validation loss. GGUF/Ollama export and download remain disabled.

## Colab

[Open the updated n8n repair Colab notebook](https://colab.research.google.com/github/ProMBZ/finetune-anywhere/blob/project-01-n8n-workflow-repair/paths/colab/n8n-workflow-repair.ipynb).

Select a T4 GPU when available and run top to bottom. It prepares the balanced dataset, validates patches, runs project tests, fetches official n8n documentation, scores the base model, fine-tunes Qwen3 and evaluates the original 192 held-out prompts and the separate 48-case challenge set. The evidence ZIP records data preparation, label corrections, source hashes, settings and complete base/fine-tuned outputs for both evaluations.

Keep `eval.jsonl` untouched. Do not train against held-out references or tune repeatedly against individual test misses.
