# Dataset card: n8n workflow diagnosis and repair v0.7

## Authored data

The v0.4 source corpus contains 640 training, 128 validation and 192 held-out evaluation cases. v0.5/v0.6/v0.7 add 888 independently authored synthetic training examples.

| Supplemental category | Cases |
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

The latest v0.7 increment contains 348 train-only examples (84 mapping, 72 linking, 48 node configuration, 36 Code node, 36 execution/API, 24 expression, 24 workflow structure and 24 credentials) and 24 validation-only examples (12 mapping and 12 item linking).

## Prepared splits

The 1,528 authored synthetic training scenarios (640 original + 888 supplemental) are balanced to 64 examples for each of the 48 canonical diagnosis codes. The resulting prepared training split contains 3,072 examples. The remaining 1,544 examples are deterministic node-name, safe identifier/domain and position variants; they preserve the source labels/patches and are **not** independent incidents.

| Split | Count | Distribution |
|---|---:|---|
| Training | 3,072 | 64 per code; 384 per category |
| Validation | 156 | 128 original + 4 v0.6 + 24 v0.7 supplemental |
| Original held-out evaluation | 192 | 24 per category; 4 per code |
| **Prepared total** | **3,420** | Excludes separate challenge evaluation |

The separate `challenge-eval-v0.6.jsonl` contains 48 held-out synthetic cases, one per diagnosis code. It is not included in training, validation or the prepared split.

## Data quality and safety

- The original `eval.jsonl` is preserved byte-for-byte and is not used for training or validation.
- Prompt fingerprints are checked after removing case IDs to detect exact duplicates and cross-split leakage.
- Category and diagnosis labels are checked against the canonical 48-code taxonomy.
- Patch paths are validated against the supplied synthetic workflow fixture. This does not execute the repaired workflow in n8n.
- Credentials and secrets, ambiguous or unavailable item lineage, network/DNS failures, confirmed provider outages and unexplained runtime timeouts require human review rather than speculative patches.
- All authored additions are synthetic. V0.7 improves coverage of distinct reasoning patterns; it does not establish production incident coverage.
- The legacy test split provides only four examples per diagnosis code. Its code-level percentages are highly sensitive to one case and should not be read as a reliability guarantee.

The notebook uses Qwen3-1.7B 4-bit QLoRA, rank 16 / alpha 32, a 1.0e-5 learning rate and two epochs. TRL conversational prompt/completion training uses `completion_only_loss=True`; validation loss selects the best checkpoint with evaluation/checkpoint saves every 128 steps. GGUF export/download remain disabled.
