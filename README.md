# NuosuGPT Inference and Evaluation

This release provides end-to-end inference and task evaluation code for NuosuGPT. Model weights are not included.

## Required artifacts

Users must independently provide compatible local paths for the following artifacts. The repository does not contain datasets, LoRA adapters, router checkpoints, or base-model weights.

- Qwen2.5-VL base model;
- NuosuGPT router checkpoint and router metadata JSON;
- task-specific LoRA adapters for Yi OCR, Chinese-Yi translation, and Yi dialogue;
- a JSON evaluation dataset with `instruction`, optional `input`/`image`, and `output` fields.

## 1. End-to-end inference

The router calculates the cosine similarity between the input feature and each task prototype. It first selects the prototype with the highest cosine similarity, then activates its LoRA expert only when that score exceeds the selected expert's learned threshold. Otherwise, inference follows the frozen base-model path. The script writes the reference answer, routing diagnostics, base-model answer, and NuosuGPT answer to one CSV.

```bash
pip install -r requirements.txt

python run_nuosugpt_inference.py \
  --base-model /path/to/your/Qwen2.5-VL-7B \
  --router-checkpoint /path/to/your/router_checkpoint.pth \
  --router-metadata /path/to/your/router_metadata.json \
  --ocr-lora /path/to/your/ocr_lora_adapter \
  --translation-lora /path/to/your/translation_lora_adapter \
  --dialogue-lora /path/to/your/dialogue_lora_adapter \
  --dataset /path/to/your/evaluation_dataset.json \
  --output outputs/nuosugpt_predictions.csv \
  --device cuda:0
```

## 2. Task metrics

```bash
python evaluate_task_metrics.py outputs/nuosugpt_predictions.csv
```

The default 891-sample split is fixed as 303 Yi OCR samples, 305 Chinese-Yi translation samples, and 283 Yi dialogue samples. Metrics are Exact Accuracy and CER for OCR; character-level BLEU-4, Exact Accuracy, and CER for translation; and character-level ROUGE-L F1 and CER for dialogue.

`metrics_output/metrics_summary.json` contains the final summary, while `metrics_output/per_sample_metrics.csv` stores per-sample scores.
