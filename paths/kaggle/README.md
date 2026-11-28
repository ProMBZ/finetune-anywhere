# Fine-tune on Kaggle

Train a Qwen3 text model, compare real before/after answers and export a local model. You need a Kaggle account and a GPU runtime. Plan an afternoon for the first run.

[Open the notebook](https://www.kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/junaiddshaukat/finetune-anywhere/main/paths/kaggle/finetune_unsloth_kaggle.ipynb).

## Run the notebook

1. Import the notebook, select a GPU accelerator, and enable Internet in settings.
2. Run **Check the GPU**. Confirm the NVIDIA GPU is listed.
3. Run **Install Unsloth**. The cell follows Unsloth's recommended Kaggle installation sequence and displays its logs. If prompted to restart, restart and continue at Settings.
4. In **Settings and recipe**, keep `RECIPE = 'json-extractor'` and `MODEL_ALIAS = 'starter'` for the first run. This selects the Apache-2.0 Qwen3 1.7B text model.
5. The optional authentication cell reads HF_TOKEN from Kaggle Secrets for later uploads. Leave it off for the first run.
6. Run cells **Load the model** through **Download evidence and model**, in order.

The notebook shows model loading, every LoRA setting, the non-thinking chat template, data validation, base answers, training, a test prompt, held-out evaluation, saving and GGUF export. Training uses 40 original examples; 10 validation examples guide the run and 10 held-out examples supply the final comparison.

## Send back real results

Download **`run-evidence.zip`** from the final results cell. It contains:

- `comparison.md`: all held-out prompts and actual base and fine-tuned answers.
- `comparison.json`: the full system/user messages, references and raw answers.
- `run.json`: model, software versions, GPU, settings, data hashes, elapsed training time and export status.
- `training-log.json`: training and validation losses.

Send that ZIP to the maintainer to add real before/after examples. If a cell fails, send its full error output and the section title. Run the evidence cell after an export failure to collect the already-saved training results.

Download the GGUF from the Files panel. Use Save Version with outputs to retain the run.

## Run the exported model locally

Install [Ollama](https://ollama.com/download), start it and save the downloaded GGUF as `outputs/finetuned-q4_k_m.gguf` in your local checkout. Create `outputs/Modelfile`:

```dockerfile
FROM ./finetuned-q4_k_m.gguf
PARAMETER temperature 0
PARAMETER num_ctx 2048
```

```bash
ollama create recipe-local -f outputs/Modelfile
ollama run recipe-local
```

Use the recipe's system instruction when testing the extraction task. See [local inference](../../guides/06-run-locally.md) and [evaluation](../../guides/04-evaluate.md) for API calls and a quantized base-versus-tuned comparison.

## Train on your own data

The last notebook section accepts `train.jsonl`, `valid.jsonl` and `eval.jsonl` in [chat format](../../guides/02-make-a-dataset.md). Turn on `USE_OWN_DATA`, attach your dataset through the Input panel and choose it, and run validation. Restart the session, rerun Install and Settings, apply the printed data path and `RECIPE = 'my-task'`, then load a fresh base and train again.

[Troubleshooting](../../guides/faq.md) · [Share your model](../../guides/07-share-your-model.md)

## Hackathon notebook: n8n Workflow Repair v0.7

**[Import the dedicated Kaggle notebook](https://www.kaggle.com/kernels/welcome?src=https://raw.githubusercontent.com/ProMBZ/finetune-anywhere/project-01-n8n-workflow-repair/paths/kaggle/n8n-workflow-repair.ipynb).**

Choose **GPU T4 x2** and turn **Internet On**. This port uses the same Qwen3-1.7B 4-bit QLoRA recipe, v0.7 data preparation, retrieval, held-out evaluation and scoring as the Colab notebook. It exposes one T4 to Python to preserve the original single-GPU effective batch semantics. It saves a new adapter and evidence under `/kaggle/working` and does not overwrite an existing model.

See [the step-by-step n8n Kaggle guide](README.md#hackathon-notebook-n8n-workflow-repair-v07).

