# n8n workflow diagnosis and repair

Fine-tune Qwen3-1.7B with the starter repository's Unsloth QLoRA Colab pipeline to classify n8n failures, identify root causes and propose minimal JSON Patch repairs.

## Dataset v0.3

| Split | Cases | Per category |
|---|---:|---:|
| Training | 640 | 80 |
| Validation | 128 | 16 |
| Held-out evaluation | 192 | 24 |
| **Total** | **960** | |

Eight categories are covered: expression, data_mapping, item_linking, node_configuration, workflow_structure, code_node, execution_api, credentials.

There are 48 canonical diagnosis codes (six per category). The codebook is in `data/diagnosis-codes.json`; Colab adds its concise code definitions to the shared system prompt for train, validation and evaluation. v0.2 contained inconsistent near-synonymous labels such as `AMBIGUOUS_ITEM_LINK` and `AMBIGUOUS_ITEM_LINKING`, which made exact code scoring unnecessarily noisy.

The training split is balanced at 80 cases per category. It retains 200 v0.2 training cases and 50 former validation cases after normalizing their diagnosis labels, plus 390 newly authored synthetic examples. Validation and evaluation are new. The held-out evaluation split contains 192 cases: 24 per category and four per diagnosis code. Its scenario templates are separate from those used in training. The original v0.2 evaluation set is archived unchanged at `data/archive/eval-v0.2.jsonl` and is not used for v0.3 training.

## Evaluation improvements

- Diagnosis-code output is checked against the controlled category-specific codebook.
- Per-category totals include malformed model outputs as incorrect, rather than silently omitting them.
- The report includes codebook validity, exact diagnosis-code accuracy, macro per-code accuracy, accuracy conditional on correct category, per-code results and a confusion matrix.
- Patch applicability remains a structural JSON fixture check only, not a live n8n execution test.
- The same retrieved official documentation excerpts are supplied to training, validation and evaluation prompts to avoid a documentation-context mismatch.
- Best adapter checkpoint is selected by validation loss.

## Fine-tuning configuration

Qwen3-1.7B; 4-bit QLoRA; LoRA rank 16 / alpha 32; learning rate 5e-5; four epochs; 2048-token context; batch size 1 with gradient accumulation 8. The notebook validates actual token lengths before training and reduces retrieval excerpts if necessary.

## Scope and limitations

The model proposes patches; it never applies them to a production workflow. Credential cases and evidenced external outages require abstention. Missing item lineage can use `.first()` only when exactly one upstream result is explicitly guaranteed; otherwise it must abstain. Structural patch applicability does not prove successful live execution. Synthetic data is useful for controlled iteration but cannot guarantee 100% real-world accuracy. A separate real-workflow test set is still required.

## Colab

[Open the updated n8n repair Colab notebook](https://colab.research.google.com/github/ProMBZ/finetune-anywhere/blob/project-01-n8n-workflow-repair/paths/colab/n8n-workflow-repair.ipynb).

Select a T4 GPU when available, then run the notebook top to bottom. It validates the dataset, checks project tests, fetches official n8n documentation, applies the same retrieval step to all splits, scores the base model, fine-tunes Qwen3, and evaluates all 192 held-out prompts. GGUF/Ollama export is intentionally disabled.

Keep `eval.jsonl` untouched. Do not train against the held-out references or tune repeatedly against individual test misses.
