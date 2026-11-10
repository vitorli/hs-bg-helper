"""
eval_clf.py — оценка обученного BGClassifier на размеченном датасете.

Два режима:
- holdout-прогон train_classifier.py уже даёт честную оценку;
- этот скрипт меряет качество ТЕКУЩЕГО чекпоинта на всём dataset/labels.jsonl
  (включая те примеры, на которых он учился) + среднюю уверенность —
  ориентир для clf_threshold в каскаде.

Запуск:
    python eval_clf.py [--ckpt checkpoints/bg_clf]
"""

import argparse
import json
import os

from classifier import BGClassifier, COMP_CLASSES
from features import encode_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(BASE_DIR, "dataset", "labels.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(BASE_DIR, "checkpoints", "bg_clf"))
    args = ap.parse_args()

    if not os.path.exists(LABELS_PATH):
        print(f"Не найден {LABELS_PATH}")
        return 1

    clf = BGClassifier(args.ckpt)
    correct = {"buy": [0, 0], "comp": [0, 0]}
    conf_sum, conf_n = 0.0, 0
    for ln in open(LABELS_PATH, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        snap, labels = r["snapshot"], r["labels"]
        text = encode_text(snap)
        p = clf.predict(text, shop_size=len(snap.get("shop", [])))
        if labels.get("buy_slot") is not None:
            correct["buy"][1] += 1
            correct["buy"][0] += int(p["buy"][0] == labels["buy_slot"])
            conf_sum += p["buy"][1]
            conf_n += 1
        if labels.get("comp"):
            correct["comp"][1] += 1
            correct["comp"][0] += int(COMP_CLASSES[p["comp"][0]] == labels["comp"])
    b, bn = correct["buy"]
    c, cn = correct["comp"]
    print(f"buy (9 классов):   {b}/{bn} = {b / max(bn, 1):.2f}"
          f" | средняя уверенность {conf_sum / max(conf_n, 1):.2f}")
    print(f"comp (12 классов): {c}/{cn} = {c / max(cn, 1):.2f}")
    print("заметь: оценка на всех данных, включая обучающие — ориентир, не holdout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
