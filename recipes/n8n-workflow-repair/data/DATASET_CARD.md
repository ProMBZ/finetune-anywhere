# Dataset card: n8n workflow diagnosis and repair v0.4

The committed source corpus contains 640 training, 128 validation and 192 held-out evaluation cases. Colab prepares **1,536 training examples (32 per canonical diagnosis code)** by retaining all 640 source cases and adding 896 deterministic, meaning-preserving variants. These are not 1,536 independent incidents. Validation and held-out evaluation are not augmented.

| Prepared split | Examples | Distribution |
|---|---:|---|
| Training | 1,536 | 32 per diagnosis code; 192 per category |
| Validation | 128 | 16 per category; 2–3 per diagnosis code |
| Held-out evaluation | 192 | 24 per category; 4 per diagnosis code |
| **Total** | **1,856** | |

There are 48 canonical diagnosis codes: six in each of eight categories. Exact values and definitions are documented in `diagnosis-codes.json`.

## Critical v0.3 correction: case-ID target leakage

The earlier validation/evaluation identifiers encoded the target category and diagnosis code (for example, an ID containing `expression_prefix_missing`). Since `case_id` is part of the user prompt, this disclosed the answer. v0.4 replaces identifiers in every committed split with opaque IDs and adds validator checks to reject semantic IDs. Prompt uniqueness is also checked after removing the ID field.

## Training augmentation

The preparation script retains all 640 committed training cases and adds 896 controlled variations to balance the 48 diagnosis codes at 32 cases each. It consistently renames workflow nodes across expressions, connections and patches; changes safe sample identifiers/domains; and varies node positions where present. The original label and patch are preserved.

These are **augmentation variants, not 896 newly discovered root causes or independent real-world incidents**. They improve representation and surface-form variation, but do not replace independently authored cases.

## Evaluation, retrieval and training

Validation and evaluation remain unaugmented apart from opaque IDs. The evaluation contains four cases per diagnosis code, so per-code estimates remain noisy. The notebook uses TRL 0.22.2's conversational prompt/completion format with `completion_only_loss=True`, so loss is applied to assistant diagnosis/repair targets without requiring assistant-token generation masks in the stock Qwen3 template. The same retrieved official documentation context is used for each split and for both base/fine-tuned evaluation.

Malformed outputs count as incorrect in full diagnosis-code validity and abstention-recall denominators. Reports include exact and macro code accuracy, category accuracy, confusion matrix, human-review decision accuracy and safe-abstention precision/recall. Patch applicability is a structural JSON fixture check only; it does not execute a live n8n workflow.

All examples are synthetic. No dataset size or synthetic score guarantees 100% real-world accuracy. Never modify credentials or apply generated patches automatically to production.
