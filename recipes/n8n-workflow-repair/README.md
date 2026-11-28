# n8n workflow diagnosis and repair

Fine-tune Qwen3-1.7B to classify n8n workflow failures, identify evidence-supported root causes and propose minimal JSON Patch repairs.

## Dataset v0.7 training and evaluation

The v0.4 source corpus contains 640 training, 128 validation and 192 held-out evaluation cases. v0.5/v0.6/v0.7 add **888 independently authored synthetic training scenarios** across the eight canonical categories.

| Category | Authored supplemental cases |
|---|---:|
| Data mapping | 228 |
| Item linking | 144 |
| Node configuration | 126 |
| Code node | 108 |
| Execution API | 102 |
| Expression | 60 |
| Workflow structure | 60 |
| Credentials | 60 |
| **Total** | **888** |

The v0.7 increment contributes 348 cases: 84 mapping, 72 linking, 48 node configuration, 36 Code-node, 36 execution/API, 24 expression, 24 workflow-structure and 24 credential-safety cases. The additional 24 v0.7 validation-only examples focus on mapping and item-linking diagnosis boundaries.

The authored source pool is 1,528 examples (640 original + 888 supplemental). Colab balances the 48 diagnosis codes to **3,072 training examples (64 per code)**. Remaining slots are filled by deterministic, controlled variants. Those variants preserve their source labels and patches; they are not independent incidents. The prepared split is:

| Split | Cases | Notes |
|---|---:|---|
| Training | 3,072 | 64 per diagnosis code; 384 per category |
| Validation | 156 | 128 original + 4 v0.6 + 24 v0.7 validation-only cases |
| Original held-out evaluation | 192 | 24 per category; 4 per diagnosis code |
| **Prepared total** | **3,420** | Training + validation + original held-out evaluation |

The separate `challenge-eval-v0.6.jsonl` contains 48 fresh synthetic cases, one per canonical diagnosis code. It remains outside training and validation.

## What v0.7 is designed to improve

- **Data mapping:** absent/null versus nested-path mismatch; zero-based array index errors; local type conversion; preserving valid `0`, `false` and empty-string values with nullish fallbacks; textual boolean parsing.
- **Item linking:** missing `pairedItem` metadata versus incorrect source indexes; wrong first/last selection under explicit ordering; wrong upstream node; all-items versus first-item mistakes; ambiguity versus irrecoverably lost lineage.
- **Node configuration and API:** request method/body/response-format, timeout units and SLA evidence, wait durations, bounded 429 retries, external payload field/type contracts, provider outages and network failures.
- **Code nodes and workflow graphs:** n8n output item shape, execution mode, undefined variables, preserved source indexes, missing/invalid connections, IF branch routing and Merge inputs.
- **Safety:** credentials, secrets, unresolved item lineage, confirmed outages, network-layer faults and unexplained runtime timeouts should abstain rather than invent a patch.

This is a deliberate increase in reasoning-pattern diversity, not a claim that simply adding more rows guarantees accuracy.

## Benchmark integrity and limitations

- The original `data/eval.jsonl` is preserved byte-for-byte and is not used for training or validation.
- The separate 48-case challenge set is also held out.
- Dataset preparation checks opaque case IDs, canonical diagnosis codes, exact prompt duplicates and cross-split prompt fingerprints.
- JSON Patch applicability is checked against synthetic workflow fixtures. It does **not** prove execution inside a live n8n instance.
- All authored additions are synthetic. There is no production incident corpus in this dataset.
- The legacy evaluation has only four cases per diagnosis code. Scores at category/code level are noisy; 90% is a target to test, not a guaranteed outcome.
- Strict gold-patch match can penalize an alternative valid repair. Review it alongside diagnosis accuracy, patch applicability and safe-abstention metrics.

## Fine-tuning configuration

Qwen3-1.7B, 4-bit QLoRA, LoRA rank 16 / alpha 32, **learning rate 1.0e-5**, two epochs, 2,048-token context, batch size 1 and gradient accumulation 8. The prepared 3,072-row training split means **768 planned optimizer steps**. Validation and checkpoint saves occur every 128 steps; the best checkpoint is selected by validation loss. This is intentionally more conservative than the earlier 1.5e-5 run after the observed validation-loss increase.

Training uses TRL 0.22.2 conversational prompt/completion data with `completion_only_loss=True`. GGUF/Ollama export and model download remain disabled until adapter quality is accepted.

## Colab

[Open the Qwen n8n Workflow Repair v0.7 Colab notebook](https://colab.research.google.com/github/ProMBZ/finetune-anywhere/blob/project-01-n8n-workflow-repair/paths/colab/n8n-workflow-repair.ipynb).

Select a T4 GPU when available and run from a **fresh runtime**, top to bottom. Step 8 prepares the v0.7 data, checks tokenizer lengths, validates the prepared recipe and runs the project tests before any GPU training. The notebook then evaluates the base model, trains the adapter, evaluates the original 192 held-out cases and the separate 48-case challenge set, and saves the adapter and full evidence. Review the validation curve and all category/code confusion before deciding whether the model is submission-ready.

Keep both held-out evaluation sets untouched. Do not tune repeatedly against individual test misses.

## Kaggle

[Import the dedicated Kaggle v0.7 notebook](https://www.kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/ProMBZ/finetune-anywhere/project-01-n8n-workflow-repair/paths/kaggle/n8n-workflow-repair.ipynb). Select **GPU T4 x2** and turn **Internet On**. The port changes only runtime paths, storage and artifact downloads; it reads the same `config.yaml`, source data, model registry, documentation and evaluation code as Colab. Python uses one T4 (GPU 0) to preserve the original single-GPU batch semantics. It saves a new adapter and full evidence in a unique folder under `/kaggle/working/finetune-anywhere-runs/`.

See the [Kaggle run guide](../../paths/kaggle/README.md#hackathon-notebook-n8n-workflow-repair-v07) for import, settings and download instructions. The run does not touch an existing adapter or Google Drive files.

## Kaggle

[Import the dedicated Kaggle v0.7 notebook](https://www.kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/ProMBZ/finetune-anywhere/project-01-n8n-workflow-repair/paths/kaggle/n8n-workflow-repair.ipynb). Select **GPU T4 x2** and turn **Internet On**. The port changes only runtime paths, storage and artifact downloads; it reads the same `config.yaml`, source data, model registry, documentation and evaluation code as Colab. Python uses one T4 (GPU 0) to preserve the original single-GPU batch semantics. It saves a new adapter and full evidence in a unique folder under `/kaggle/working/finetune-anywhere-runs/`.

See the [Kaggle run guide](../../paths/kaggle/README.md#hackathon-notebook-n8n-workflow-repair-v07) for import, settings and download instructions. The run does not touch an existing adapter or Google Drive files.

