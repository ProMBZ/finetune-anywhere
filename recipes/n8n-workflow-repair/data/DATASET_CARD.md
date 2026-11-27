# Dataset card: n8n workflow diagnosis and repair v0.6

## Prepared training and validation data

The v0.4 source corpus contains 640 training, 128 validation and 192 legacy held-out evaluation cases. v0.5/v0.6 add 540 independently authored synthetic training scenarios across the eight canonical categories.

| Supplemental category | New cases |
|---|---:|
| Data mapping | 144 |
| Item linking | 72 |
| Node configuration | 78 |
| Code node | 72 |
| Execution API | 66 |
| Expression | 36 |
| Workflow structure | 36 |
| Credentials | 36 |
| **Total supplemental training** | **540** |

The preparation script normalizes a small number of legacy label-boundary errors, then retains the 1,180 source/supplemental training cases and balances all 48 diagnosis codes to 48 examples each. It creates deterministic variants only to fill remaining per-code slots. These variants are not independent incidents or new root causes.

| Prepared split | Examples | Distribution |
|---|---:|---|
| Training | 2,304 | 48 per diagnosis code; 288 per category |
| Validation | 132 | 128 original + 4 supplemental |
| Legacy held-out evaluation | 192 | 24 per category; 4 per diagnosis code |
| **Prepared total** | **2,628** | Excludes separate challenge evaluation |

## Separate challenge evaluation

`challenge-eval-v0.6.jsonl` contains 48 independently authored synthetic cases, one for each canonical diagnosis code. It is kept out of training and validation. It is evaluated separately from the original 192-case held-out benchmark. One case per code is a useful challenge smoke test, but it is not a statistically precise per-code or per-category estimate.

The original `eval.jsonl` is preserved unchanged. Neither held-out evaluation set is augmented or used for training.

## Data quality and limitations

- Every prompt uses an opaque case ID; labels are not encoded in IDs.
- Prompt fingerprints are checked after removing the case ID to catch exact duplicates and cross-split leakage.
- Diagnosis code and category are checked against the canonical taxonomy.
- JSON Patch paths are checked against synthetic workflow fixtures; this does not execute a workflow in a live n8n instance.
- Credential/secret, unresolved lineage, network/DNS failure, confirmed outage and unsupported runtime-timeout cases require human review rather than speculative patches.
- All examples are synthetic. The generated augmentation variants preserve their source labels and patches; they do not represent independent real-world incidents.
- The held-out sets are small and synthetic. Their scores do not establish production reliability.

The notebook uses conversational prompt/completion training with `completion_only_loss=True`; validation loss selects the best checkpoint. Base and fine-tuned evaluation use the same retrieved official-document context and generation settings. GGUF export/download remain disabled.
