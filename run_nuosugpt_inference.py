"""Run NuosuGPT routing, LoRA loading, generation, and CSV export.

No model weights are bundled with this code. Provide the base model, the router
checkpoint/metadata, and the three task-specific LoRA adapter directories at runtime.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


EXPERT_LABELS = {
    0: "Yi_OCR_Expert",
    1: "Yi_Translation_Expert",
    2: "Yi_Dialogue_Expert",
}


class NuosuRouter(nn.Module):
    """Router architecture for the learnable cosine-threshold checkpoint."""

    def __init__(
        self,
        vision_tower: nn.Module,
        token_embedding: nn.Module,
        qwen_hidden_size: int,
        vision_hidden_size: int,
        d_model: int,
        feature_dim: int,
        temperature: float,
        gate_alpha: float,
        threshold_min: float,
        threshold_max: float,
        known_class_count: int,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.gate_alpha = gate_alpha
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.vision_tower = vision_tower
        self.embedding = token_embedding
        for parameter in self.vision_tower.parameters():
            parameter.requires_grad = False
        for parameter in self.embedding.parameters():
            parameter.requires_grad = False

        self.text_reducer = nn.Linear(qwen_hidden_size, d_model)
        self.vis_reducer = nn.Linear(vision_hidden_size, d_model)
        self.modality_embedding = nn.Parameter(torch.randn(2, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.att_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.feature_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.centers = nn.Parameter(torch.randn(known_class_count, feature_dim))
        self.threshold_logits = nn.Parameter(torch.zeros(known_class_count))

    def get_thresholds(self) -> torch.Tensor:
        return self.threshold_min + (self.threshold_max - self.threshold_min) * torch.sigmoid(self.threshold_logits)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        vision_indices: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = input_ids.size(0)
        dtype = self.text_reducer.weight.dtype
        token_embeddings = self.embedding(input_ids).to(dtype)
        text_states = self.text_reducer(token_embeddings) + self.modality_embedding[0]
        vision_state = torch.zeros(
            (batch_size, 1, text_states.size(-1)),
            device=text_states.device,
            dtype=text_states.dtype,
        )

        if pixel_values is not None and image_grid_thw is not None and vision_indices:
            vision_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw)
            if not isinstance(vision_outputs, torch.Tensor):
                vision_outputs = getattr(vision_outputs, "last_hidden_state", vision_outputs[0])
            vision_outputs = vision_outputs.to(self.vis_reducer.weight.dtype)
            if vision_outputs.dim() == 3:
                vision_outputs = vision_outputs.squeeze(0)
            split_sizes = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
            pooled = torch.stack([part.mean(dim=0) for part in torch.split(vision_outputs, split_sizes)])
            vision_state[vision_indices] = self.vis_reducer(pooled).unsqueeze(1) + self.modality_embedding[1]

        states = torch.cat([text_states, vision_state], dim=1)
        mask = torch.cat(
            [attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=1,
        )
        encoded = self.transformer(states, src_key_padding_mask=(mask == 0))
        attention = torch.matmul(self.att_query, encoded.transpose(-1, -2))
        attention = attention.masked_fill((mask == 0).unsqueeze(1), -1e9)
        pooled = torch.matmul(F.softmax(attention, dim=-1), encoded).squeeze(1)
        features = self.feature_head(pooled)
        normalized_features = F.normalize(features.float(), p=2, dim=1)
        normalized_centers = F.normalize(self.centers.float(), p=2, dim=1)
        cosine_similarity = torch.matmul(normalized_features, normalized_centers.t())
        cosine_logits = cosine_similarity * self.temperature
        thresholds = self.get_thresholds()
        gate_scores = torch.sigmoid(self.gate_alpha * (cosine_similarity - thresholds.unsqueeze(0)))
        return {
            "cosine_similarity": cosine_similarity,
            "cosine_logits": cosine_logits,
            "gate_scores": gate_scores,
            "thresholds": thresholds,
            "features": normalized_features,
            "centers": normalized_centers,
        }


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_sample(sample: dict[str, Any], dataset_dir: Path) -> dict[str, str | Path | None]:
    system = str(sample.get("system", "") or "")
    instruction = str(sample.get("instruction", "") or "")
    user_messages = [message for message in sample.get("messages", []) if message.get("role") == "user"]
    if not instruction and user_messages:
        instruction = str(user_messages[0].get("content", "") or "")
    image = sample.get("image") or (sample.get("images") or [None])[0]
    image_path = None
    if image:
        candidate = Path(str(image))
        image_path = candidate if candidate.is_absolute() else dataset_dir / candidate
    return {
        "system": system,
        "instruction": instruction,
        "input": str(sample.get("input", "") or ""),
        "target": str(sample.get("output", sample.get("target", "")) or ""),
        "image": image_path,
    }


def build_messages(sample: dict[str, str | Path | None]) -> tuple[list[dict[str, Any]], str]:
    parts = []
    if sample["instruction"]:
        parts.append(f"Instruction: {sample['instruction']}")
    if sample["input"]:
        parts.append(f"Input: {sample['input']}")
    if sample["system"]:
        parts.append(f"Context: {sample['system']}")
    question = "\n".join(parts)
    content: list[dict[str, Any]] = []
    image_path = sample["image"]
    if isinstance(image_path, Path) and image_path.is_file():
        content.append({"type": "image", "image": str(image_path)})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}], question


def find_vision_tower(base_model: nn.Module) -> nn.Module:
    for candidate in (base_model, getattr(base_model, "model", None), getattr(base_model, "transformer", None)):
        if candidate is not None and hasattr(candidate, "visual"):
            return candidate.visual
    raise AttributeError("Unable to locate the Qwen2.5-VL vision tower.")


def generate(model: PeftModel, processor: AutoProcessor, inputs: dict[str, torch.Tensor], input_length: int, max_new_tokens: int) -> str:
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    return processor.decode(output[0][input_length:], skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NuosuGPT end-to-end inference.")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--router-metadata", type=Path, required=True)
    parser.add_argument("--ocr-lora", type=Path, required=True, help="Path to the user-provided Yi OCR LoRA adapter")
    parser.add_argument(
        "--translation-lora", type=Path, required=True, help="Path to the user-provided Chinese-Yi translation LoRA adapter"
    )
    parser.add_argument("--dialogue-lora", type=Path, required=True, help="Path to the user-provided Yi dialogue LoRA adapter")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the user-provided JSON evaluation dataset")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.router_metadata)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    vision_tower = find_vision_tower(base_model)
    if hasattr(base_model.config, "text_config"):
        text_dim = base_model.config.text_config.hidden_size
    else:
        text_dim = base_model.config.hidden_size
    router = NuosuRouter(
        vision_tower=vision_tower,
        token_embedding=base_model.get_input_embeddings(),
        qwen_hidden_size=text_dim,
        vision_hidden_size=getattr(vision_tower, "hidden_size", 1280),
        d_model=int(metadata.get("d_model", 256)),
        feature_dim=int(metadata.get("feature_dim", 128)),
        temperature=float(metadata.get("temperature", 16.0)),
        gate_alpha=float(metadata["gate_alpha"]),
        threshold_min=float(metadata["threshold_min"]),
        threshold_max=float(metadata["threshold_max"]),
        known_class_count=int(metadata.get("known_class_count", 3)),
    ).to(args.device)
    router.load_state_dict(torch.load(args.router_checkpoint, map_location="cpu"))
    router.eval()

    adapters = [args.ocr_lora, args.translation_lora, args.dialogue_lora]
    model = PeftModel.from_pretrained(base_model, adapters[0], adapter_name="expert_0")
    for expert_id, adapter_path in enumerate(adapters[1:], start=1):
        model.load_adapter(adapter_path, adapter_name=f"expert_{expert_id}")
    model.eval()

    with args.dataset.open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    if not isinstance(dataset, list):
        raise ValueError("The evaluation dataset must be a JSON array.")

    rows = []
    for item in tqdm(dataset, desc="NuosuGPT inference"):
        sample = normalize_sample(item, args.dataset.parent)
        messages, question = build_messages(sample)
        try:
            vision_inputs, _ = process_vision_info(messages)
        except Exception:
            vision_inputs = None
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = processor(text=[prompt], images=vision_inputs, padding=True, return_tensors="pt").to(args.device)
        input_length = model_inputs["input_ids"].shape[1]
        has_image = isinstance(sample["image"], Path) and sample["image"].is_file()

        with torch.inference_mode():
            router_output = router(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs.get("pixel_values"),
                image_grid_thw=model_inputs.get("image_grid_thw"),
                vision_indices=[0] if has_image else None,
            )
            cosine_similarity = router_output["cosine_similarity"]
            thresholds = router_output["thresholds"]
            cosine_scores = cosine_similarity[0]
            best_score, pred_id = cosine_similarity.max(dim=1)
            best_score = best_score.item()
            pred_id = pred_id.item()
            class_threshold = thresholds[pred_id].item()
            final_route = EXPERT_LABELS[pred_id] if best_score >= class_threshold else "Base_Clean"
            with model.disable_adapter():
                base_answer = generate(model, processor, model_inputs, input_length, args.max_new_tokens)
            if final_route != "Base_Clean":
                model.set_adapter(f"expert_{pred_id}")
                answer = generate(model, processor, model_inputs, input_length, args.max_new_tokens)
            else:
                answer = base_answer

        rows.append({
            "Question": question,
            "Target": sample["target"],
            "Predicted_Expert": EXPERT_LABELS[pred_id],
            "Cosine_OCR": round(float(cosine_scores[0].item()), 6),
            "Cosine_Translation": round(float(cosine_scores[1].item()), 6),
            "Cosine_Dialogue": round(float(cosine_scores[2].item()), 6),
            "Cosine_Score": round(best_score, 6),
            "Class_Threshold": round(class_threshold, 6),
            "Threshold_Pass": int(final_route != "Base_Clean"),
            "Final_Route": final_route,
            "Base_Answer": base_answer,
            "Model_Answer": answer,
        })
        clear_memory()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
