"""
train_classifier.py — обучение голов классификатора BGClassifier на
dataset/train.jsonl (см. gen_dataset.py -> labels.py -> features.py).

Обучает ТОЛЬКО головы (энкодер заморожен) — быстро и на CPU: сотни примеров,
несколько эпох, минуты. Оценка — holdout accuracy по каждой голове.

Запуск:
    python train_classifier.py                # всё дообучение + holdout
    python train_classifier.py --epochs 5 --holdout 0.2
Чекпоинт: checkpoints/bg_clf/model.pt
"""

import argparse
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset

from classifier import (BUY_CLASSES, COMP_CLASSES, LAYA_REPO,
                        MAX_LEN, build_model)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train.jsonl")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "bg_clf")

HEADS = ("buy", "upgrade", "reroll", "freeze", "comp")


def labels_to_vector(questions: list[dict]) -> dict:
    """Ответы из train.jsonl -> вектор меток {-1 = нет метки}."""
    ans = {q["question"]: q.get("answer") for q in questions}
    v = {"buy": -1, "upgrade": -1, "reroll": -1, "freeze": -1, "comp": -1}

    def yn(a):
        return {"да": 1, "нет": 0}.get(a, -1)

    b = ans.get("Что купить?")
    if b is not None:
        if b == "ничего":
            v["buy"] = 0
        else:
            m = b.split("—")[0].strip()
            v["buy"] = int(m.replace("слот", "").strip()) if m.replace("слот", "").strip().isdigit() else -1
    v["upgrade"] = yn(ans.get("Апнуть таверну?"))
    v["reroll"] = yn(ans.get("Рероллить остаток золота?"))
    v["freeze"] = yn(ans.get("Заморозить магазин?"))
    c = ans.get("Цель сборки?")
    if c:
        v["comp"] = COMP_CLASSES.index(c) if c in COMP_CLASSES else -1
    return v


class BGDataset(Dataset):
    def __init__(self, rows: list[dict], tok):
        self.items = []
        for r in rows:
            vec = labels_to_vector(r.get("questions", []))
            if all(vec[h] < 0 for h in HEADS):
                continue
            enc = tok(r["text"], truncation=True, max_length=MAX_LEN)
            self.items.append((enc, vec))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        enc, vec = self.items[i]
        return (torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"]),
                {h: torch.tensor(vec[h]) for h in HEADS})


def collate(batch):
    maxlen = max(len(ids) for ids, _, _ in batch)
    pad_id = 0
    ids, masks, vecs = [], [], {h: [] for h in HEADS}
    for i, m, v in batch:
        pad = maxlen - len(i)
        ids.append(torch.cat([i, torch.full((pad,), pad_id)]))
        masks.append(torch.cat([m, torch.zeros(pad, dtype=m.dtype)]))
        for h in HEADS:
            vecs[h].append(v[h])
    return (torch.stack(ids), torch.stack(masks),
            {h: torch.stack(vecs[h]) for h in HEADS})


def run_epoch(model, loader, device, opt=None):
    train_mode = opt is not None
    model.train(train_mode)
    tot_loss, n_batches = 0.0, 0
    for ids, mask, vec in loader:
        ids, mask = ids.to(device), mask.to(device)
        vec = {h: vec[h].to(device) for h in HEADS}
        logits = model(input_ids=ids, attention_mask=mask)
        loss = 0
        for h in HEADS:
            valid = vec[h] >= 0
            if valid.any():
                loss = loss + torch.nn.functional.cross_entropy(
                    logits[h][valid], vec[h][valid])
        if train_mode:
            opt.zero_grad()
            loss.backward()
            opt.step()
        tot_loss += float(loss)
        n_batches += 1
    return tot_loss / max(n_batches, 1)


@torch.no_grad()
def holdout_acc(model, rows, tok, device):
    model.eval()
    correct = {h: [0, 0] for h in HEADS}   # [верно, всего]
    for r in rows:
        vec = labels_to_vector(r.get("questions", []))
        enc = tok(r["text"], truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(device)
        logits = model(**enc)
        for h in HEADS:
            if vec[h] < 0:
                continue
            pred = logits[h].argmax(-1).item()
            correct[h][1] += 1
            correct[h][0] += int(pred == vec[h])
    return {h: (f"{c}/{n}", round(c / max(n, 1), 2)) for h, (c, n) in correct.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(TRAIN_PATH):
        print(f"Не найден {TRAIN_PATH}")
        return 1
    rows = [json.loads(ln) for ln in open(TRAIN_PATH, encoding="utf-8")
            if ln.strip()]
    random.Random(args.seed).shuffle(rows)
    n_hold = max(1, int(len(rows) * args.holdout))
    hold, train = rows[:n_hold], rows[n_hold:]
    print(f"примеров: {len(train)} train / {len(hold)} holdout")

    model, tok, device = build_model(LAYA_REPO)
    for p in model.base.parameters():
        p.requires_grad = False

    ds = BGDataset(train, tok)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    for ep in range(1, args.epochs + 1):
        loss = run_epoch(model, loader, device, opt)
        print(f"эпоха {ep}: loss {loss:.3f}")

    print("holdout:", holdout_acc(model, hold, tok, device))

    os.makedirs(CKPT_DIR, exist_ok=True)
    heads_sd = {k: v for k, v in model.state_dict().items()
                if not k.startswith("base.")}
    torch.save(heads_sd, os.path.join(CKPT_DIR, "model.pt"))
    print(f"-> {os.path.join(CKPT_DIR, 'model.pt')} (только головы, "
          f"{len(heads_sd)} тензоров)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
