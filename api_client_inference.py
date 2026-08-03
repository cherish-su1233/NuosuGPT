"""Call a hosted NuosuGPT inference API and export results for evaluation.

The API server keeps the base model, Router checkpoint, and LoRA adapters private.
This client only sends evaluation inputs and writes a CSV compatible with
evaluate_task_metrics.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm


def normalize_sample(sample: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    system = str(sample.get("system", "") or "")
    instruction = str(sample.get("instruction", "") or "")
    if not instruction:
        user_messages = [m for m in sample.get("messages", []) if m.get("role") == "user"]
        if user_messages:
            instruction = str(user_messages[0].get("content", "") or "")

    image = sample.get("image") or (sample.get("images") or [None])[0]
    image_path = None
    if image:
        candidate = Path(str(image))
        image_path = candidate if candidate.is_absolute() else dataset_dir / candidate

    parts = []
    if instruction.strip():
        parts.append(f"Instruction: {instruction.strip()}")
    if str(sample.get("input", "") or "").strip():
        parts.append(f"Input: {str(sample['input']).strip()}")
    if system.strip():
        parts.append(f"Context: {system.strip()}")

    return {
        "question": "\n".join(parts),
        "target": str(sample.get("output", sample.get("target", "")) or ""),
        "image_path": image_path,
    }


def image_data_uri(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept the documented response keys and a common nested `result` form."""
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    answer = result.get("answer", result.get("model_answer", result.get("output", "")))
    route = result.get("route", result.get("final_route", ""))
    return {
        "answer": "" if answer is None else str(answer),
        "route": "" if route is None else str(route),
        "cosine_score": result.get("cosine_score", ""),
        "class_threshold": result.get("class_threshold", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NuosuGPT inference through a hosted API.")
    parser.add_argument("--api-url", required=True, help="Hosted NuosuGPT inference endpoint")
    parser.add_argument("--api-key", default=None, help="Optional bearer token")
    parser.add_argument("--dataset", type=Path, required=True, help="Local JSON evaluation dataset")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("The evaluation dataset must be a JSON array.")

    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    rows = []
    with requests.Session() as session:
        for index, item in enumerate(tqdm(dataset, desc="NuosuGPT API inference")):
            sample = normalize_sample(item, args.dataset.parent)
            payload = {
                "question": sample["question"],
                "image": image_data_uri(sample["image_path"]),
                "max_new_tokens": args.max_new_tokens,
            }
            response = session.post(args.api_url, headers=headers, json=payload, timeout=args.timeout)
            try:
                response.raise_for_status()
            except requests.HTTPError as error:
                raise RuntimeError(f"API request failed at sample {index}: {response.text[:500]}") from error
            result = extract_response(response.json())
            rows.append(
                {
                    "Sample_Index": index,
                    "Question": sample["question"],
                    "Target": sample["target"],
                    "Final_Route": result["route"],
                    "Cosine_Score": result["cosine_score"],
                    "Class_Threshold": result["class_threshold"],
                    "Model_Answer": result["answer"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(rows)} API predictions to {args.output}")


if __name__ == "__main__":
    main()
