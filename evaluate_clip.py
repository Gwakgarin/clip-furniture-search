#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

from lora_utils import DEFAULT_LORA_WEIGHTS, load_lora


PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_CSV = DATA_DIR / "valid_ids.csv"

CATEGORY_PROMPTS = {
    "SOFA": "a product photo of a sofa",
    "CHAIR": "a product photo of a chair",
    "TABLE": "a product photo of a table",
    "BED": "a product photo of a bed",
    "DESK": "a product photo of a desk",
    "CABINET": "a product photo of a cabinet",
    "SHELF": "a product photo of a shelf",
    "OTTOMAN": "a product photo of an ottoman",
    "BENCH": "a product photo of a bench",
    "DRESSER": "a product photo of a dresser",
}


def resolve_image_path(row):
    candidates = []
    image_path = row.get("image_path", "")
    abo_image_path = row.get("abo_image_path", "")

    if image_path:
        candidates.append(Path(image_path))
        candidates.append(PROJECT_DIR / image_path)
    if abo_image_path:
        candidates.append(DATA_DIR / "abo-images" / abo_image_path)
        candidates.append(DATA_DIR / "images" / abo_image_path)

    for path in candidates:
        if path.exists():
            return path
    return None


def load_rows(csv_path, max_samples=0):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_path = resolve_image_path(row)
            caption = row.get("caption", "")
            product_type = row.get("product_type", "").upper()
            if image_path and caption and product_type in CATEGORY_PROMPTS:
                row["resolved_image_path"] = str(image_path)
                rows.append(row)
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def encode_images(model, preprocess, rows, device, batch_size):
    features = []
    for start in tqdm(range(0, len(rows), batch_size), desc="Encoding images"):
        batch_rows = rows[start:start + batch_size]
        images = [
            preprocess(Image.open(row["resolved_image_path"]).convert("RGB"))
            for row in batch_rows
        ]
        images = torch.stack(images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        features.append(image_features.cpu())
    return torch.cat(features, dim=0)


def encode_texts(model, tokenizer, texts, device, batch_size):
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch = texts[start:start + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        features.append(text_features.cpu())
    return torch.cat(features, dim=0)


def zero_shot_accuracy(image_features, rows, model, tokenizer, device, text_batch_size):
    labels = list(CATEGORY_PROMPTS.keys())
    prompts = [CATEGORY_PROMPTS[label] for label in labels]
    text_features = encode_texts(model, tokenizer, prompts, device, text_batch_size)
    sims = image_features @ text_features.T
    pred_indices = sims.argmax(dim=1).numpy()
    correct = 0
    for idx, row in enumerate(rows):
        if labels[pred_indices[idx]] == row["product_type"].upper():
            correct += 1
    return correct / max(len(rows), 1)


def retrieval_recall(image_features, text_features):
    sims = text_features @ image_features.T
    top5 = torch.topk(sims, k=min(5, sims.shape[1]), dim=1).indices
    labels = torch.arange(sims.shape[0]).unsqueeze(1)
    r1 = (top5[:, :1] == labels).any(dim=1).float().mean().item()
    r5 = (top5 == labels).any(dim=1).float().mean().item()
    return r1, r5


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP furniture retrieval metrics")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--lora", action="store_true", help="evaluate with lora_weights/clip_lora.pt")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all valid rows")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(args.csv, max_samples=args.max_samples)
    if not rows:
        raise RuntimeError("No valid image-text rows found")

    print(f"device: {device}")
    print(f"rows: {len(rows):,}")
    print(f"model: {'CLIP + LoRA' if args.lora else 'Zero-shot CLIP'}")

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    if args.lora:
        if not load_lora(model, DEFAULT_LORA_WEIGHTS, map_location="cpu"):
            raise FileNotFoundError(f"LoRA weights not found: {DEFAULT_LORA_WEIGHTS}")
        print(f"LoRA weights loaded: {DEFAULT_LORA_WEIGHTS}")
    model = model.to(device).eval()

    image_features = encode_images(model, preprocess, rows, device, args.batch_size)
    captions = [row["caption"] for row in rows]
    text_features = encode_texts(model, tokenizer, captions, device, args.text_batch_size)

    zs_acc = zero_shot_accuracy(
        image_features, rows, model, tokenizer, device, args.text_batch_size
    )
    r1, r5 = retrieval_recall(image_features, text_features)

    print("\n=== Metrics ===")
    print(f"Zero-shot category accuracy: {zs_acc:.4f} ({zs_acc * 100:.2f}%)")
    print(f"Image retrieval R@1:        {r1:.4f} ({r1 * 100:.2f}%)")
    print(f"Image retrieval R@5:        {r5:.4f} ({r5 * 100:.2f}%)")


if __name__ == "__main__":
    main()
