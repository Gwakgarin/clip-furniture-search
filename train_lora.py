#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from lora_utils import DEFAULT_LORA_WEIGHTS, apply_lora, save_lora


DATA_DIR = Path(__file__).parent / "data"
FURNITURE_IMAGES_FILE = DATA_DIR / "furniture_captions.csv"


class FurniturePairDataset(Dataset):
    def __init__(self, csv_path, preprocess):
        self.rows = []
        self.preprocess = preprocess
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                caption = row.get("caption", "")
                image_path = self._resolve_image_path(row)
                if image_path and caption:
                    row["resolved_image_path"] = str(image_path)
                    self.rows.append(row)

    @staticmethod
    def _resolve_image_path(row):
        candidates = []
        if row.get("image_path"):
            candidates.append(Path(row["image_path"]))
        if row.get("abo_image_path"):
            candidates.append(DATA_DIR / "abo-images" / row["abo_image_path"])
            candidates.append(DATA_DIR / "images" / row["abo_image_path"])

        for path in candidates:
            if path.exists():
                return path
        return None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["resolved_image_path"]).convert("RGB")
        return self.preprocess(image), row["caption"]


def contrastive_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(image_features.shape[0], device=image_features.device)
    loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def evaluate_retrieval(model, tokenizer, dataloader, device):
    model.eval()
    total = 0
    r1 = 0
    r5 = 0
    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            tokens = tokenizer(list(captions)).to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = text_features @ image_features.t()
            top5 = sims.topk(k=min(5, sims.shape[1]), dim=1).indices
            labels = torch.arange(sims.shape[0], device=device).unsqueeze(1)
            r1 += (top5[:, :1] == labels).any(dim=1).sum().item()
            r5 += (top5 == labels).any(dim=1).sum().item()
            total += sims.shape[0]
    model.train()
    return r1 / max(total, 1), r5 / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for CLIP furniture retrieval")
    parser.add_argument("--csv", default=str(FURNITURE_IMAGES_FILE), help="image-text pair CSV")
    parser.add_argument("--output", default=str(DEFAULT_LORA_WEIGHTS), help="LoRA weight output path")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    for param in model.parameters():
        param.requires_grad = False
    replaced = apply_lora(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA modules: {len(replaced)}")
    print(f"trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")

    dataset = FurniturePairDataset(args.csv, preprocess_train)
    if len(dataset) < 2:
        raise RuntimeError("Not enough valid image-text pairs for training")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    best_r5 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for images, captions in progress:
            images = images.to(device)
            tokens = tokenizer(list(captions)).to(device)

            image_features = model.encode_image(images)
            text_features = model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            loss = contrastive_loss(image_features, text_features, model.logit_scale.exp())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        r1, r5 = evaluate_retrieval(model, tokenizer, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"epoch {epoch}: loss={avg_loss:.4f}, val_R@1={r1:.4f}, val_R@5={r5:.4f}")

        if r5 > best_r5:
            best_r5 = r5
            save_lora(
                model,
                args.output,
                rank=args.rank,
                alpha=args.alpha,
                dropout=args.dropout,
            )
            print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
