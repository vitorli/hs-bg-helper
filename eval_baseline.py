"""
eval_baseline.py — sanity-check обучаемости датасета: TF-IDF + логистическая
регрессия по каждому вопросу классификатора.

Даёт первую цифру качества (accuracy с holdout) до возни с Laya: если
baseline не учится — проблема в признаках/метках, а не в модели.

Запуск:
    python eval_baseline.py            # по dataset/train.jsonl
    python eval_baseline.py --cv 5     # кросс-валидация (для маленьких данных)
"""

import argparse
import json
import os
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "dataset", "train.jsonl")

MIN_SAMPLES = 8    # меньше — только кросс-валидация, цифры не стабильны


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
    return rows


def eval_question(rows: list[dict], question: str, cv: int) -> None:
    X, y = [], []
    for r in rows:
        for q in r.get("questions", []):
            if q.get("question") == question and q.get("answer"):
                X.append(r["text"])
                y.append(q["answer"])
    if len(set(y)) < 2:
        print(f"{question}: пропуск — один класс ({len(y)} ответов)")
        return
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    Xv = vec.fit_transform(X)
    if len(y) >= MIN_SAMPLES and len(y) >= cv * 2:
        scores = cross_val_score(LogisticRegression(max_iter=2000), Xv, y,
                                 cv=min(cv, len(y) // 2))
        maj = max(sum(1 for a in y if a == c) for c in set(y)) / len(y)
        print(f"{question}: cv-acc {scores.mean():.2f} ± {scores.std():.2f} "
              f"(n={len(y)}, классов {len(set(y))}, большинство {maj:.2f})")
    else:
        print(f"{question}: мало данных (n={len(y)}) — пропуск holdout")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=int, default=5)
    args = ap.parse_args()

    if not os.path.exists(TRAIN_PATH):
        print(f"Не найден {TRAIN_PATH} — сначала gen_dataset.py && labels.py && features.py")
        return 1
    rows = load_rows(TRAIN_PATH)
    if not rows:
        print("train.jsonl пуст")
        return 1
    print(f"записей: {len(rows)}")
    for q in ("Что купить?", "Апнуть таверну?", "Рероллить остаток золота?",
              "Заморозить магазин?", "Цель сборки?"):
        eval_question(rows, q, args.cv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
