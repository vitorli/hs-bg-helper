"""
features.py — кодирование снапшота игры в признаки для обучения классификатора.

Два представления:
- encode_text()  — компактная русская текстовая сводка состояния (для
                   энкодерных decision-моделей типа Laya multilingual);
- encode_questions() — вопросы с динамическим списком вариантов (для
                   router/pointer-стиля: варианты покупок = реальные слоты).

Экспорт размеченного датасета:
    python features.py              # dataset/labels.jsonl -> dataset/train.jsonl
    python features.py --stats      # сводка: длина текста, баланс ответов

Формат train.jsonl (на строку):
    {"text": <сводка состояния>, "answers": {"Что купить?": "слот 2 — ...",
     "Апнуть таверну?": "да", ...}}
"""

import argparse
import json
import os

from advisor import _tribe_ru, load_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bg_db.json")
IN_PATH = os.path.join(BASE_DIR, "dataset", "labels.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "dataset", "train.jsonl")

COMP_TARGETS = [
    "Нежить", "Механизм", "Демон", "Дракон", "Зверь", "Пират",
    "Мурлок", "Элементаль", "Свинобраз", "Нага", "Аберрация",
]

MAX_SHOP_ITEMS = 8    # обрезка длинных магазинов (заклинания боба и т.п.)
MAX_BOARD_ITEMS = 7


def _fmt_card(m: dict, with_stats: bool = True) -> str:
    name = m.get("name") or m.get("card_id") or "?"
    parts = [name]
    tribe = _tribe_ru(m.get("tribe", ""))
    if tribe and tribe not in ("0", "Все племена"):
        parts.append(tribe)
    if m.get("kind") == "spell":
        parts.append("заклинание")
    elif with_stats:
        parts.append(f"{m.get('atk', 0)}/{m.get('hp', 0)}")
    if m.get("tier"):
        parts.append(f"т{m['tier']}")
    if m.get("golden"):
        parts.append("золот.")
    return " ".join(parts)


def encode_text(snap: dict, history: str | None = None,
                card_db: dict | None = None, max_chars: int = 1200) -> str:
    """Компактная русская сводка состояния для энкодерной модели."""
    p = []
    p.append(f"Ход {snap.get('turn', '?')}. Герой {snap.get('hero') or '?'}. "
             f"Тир {snap.get('tier', '?')}, золото {snap.get('gold', '?')}, "
             f"HP {snap.get('health', '?')}+{snap.get('armor', 0)}.")

    if snap.get("shop_frozen"):
        p.append("Магазин заморожен.")

    pool = [t for t in (_tribe_ru(x) for x in snap.get("tribes_seen", []))
            if t and t != "Все племена"]
    if pool:
        p.append("Племена в пуле: " + ", ".join(pool) + ".")

    shop = snap.get("shop", [])[:MAX_SHOP_ITEMS]
    if shop:
        p.append("Магазин: " + "; ".join(
            f"{i}) {_fmt_card(m)}" for i, m in enumerate(shop, 1)) + ".")

    board = snap.get("board", [])[:MAX_BOARD_ITEMS]
    if board:
        tribes: dict[str, int] = {}
        for m in board:
            t = _tribe_ru(m.get("tribe", "")) or "без племени"
            tribes[t] = tribes.get(t, 0) + 1
        dominant = max(tribes.items(), key=lambda kv: kv[1])
        p.append(f"Доска ({len(board)}): " + ", ".join(f"{t}×{n}" for t, n in
                sorted(tribes.items(), key=lambda kv: -kv[1])) + " — "
                + ", ".join(_fmt_card(m, with_stats=False) for m in board) + ".")
        if dominant[1] * 2 >= len(board) and dominant[0] != "без племени":
            p.append(f"Основа доски: {dominant[0]}.")

    hand = snap.get("hand", [])[:4]
    if hand:
        p.append("Рука: " + ", ".join(_fmt_card(m, with_stats=False) for m in hand) + ".")

    lc = snap.get("last_combat")
    if lc:
        lt = [t for t in (_tribe_ru(x) for x in (lc.get("tribes") or {}))
              if t and t not in ("0", "Все племена")]
        p.append(f"Прошлый бой: {lc.get('board_size', '?')} существ, "
                 f"{lc.get('total_atk', '?')}/{lc.get('total_hp', '?')}"
                 + (", " + ", ".join(lt) if lt else "") + ".")

    if history:
        # в текстовые признаки — только строки без разметки-заголовков
        hist_lines = [ln for ln in history.splitlines() if ln.strip()][1:]
        if hist_lines:
            p.append("История: " + " | ".join(hist_lines[-3:]) + ".")

    text = " ".join(p)
    return text[:max_chars]


def encode_questions(snap: dict, labels: dict | None = None) -> list[dict]:
    """Вопросы с вариантами; если даны метки — приклеиваем правильные ответы."""
    shop = snap.get("shop", [])[:MAX_SHOP_ITEMS]
    buy_options = ["ничего"] + [
        f"слот {i} — {(m.get('name') or m.get('card_id'))}" for i, m in enumerate(shop, 1)]
    questions = [
        {"question": "Что купить?", "options": buy_options},
        {"question": "Апнуть таверну?", "options": ["нет", "да"]},
        {"question": "Рероллить остаток золота?", "options": ["нет", "да"]},
        {"question": "Заморозить магазин?", "options": ["нет", "да"]},
        {"question": "Цель сборки?", "options": ["держать текущую"] + COMP_TARGETS},
    ]
    if labels:
        answers = {}
        bs = labels.get("buy_slot")
        if bs is not None:
            answers["Что купить?"] = buy_options[bs] if 0 <= bs < len(buy_options) else None
        if labels.get("upgrade") is not None:
            answers["Апнуть таверну?"] = "да" if labels["upgrade"] else "нет"
        if labels.get("reroll") is not None:
            answers["Рероллить остаток золота?"] = "да" if labels["reroll"] else "нет"
        if labels.get("freeze") is not None:
            answers["Заморозить магазин?"] = "да" if labels["freeze"] else "нет"
        if labels.get("comp"):
            answers["Цель сборки?"] = labels["comp"]
        for q in questions:
            q["answer"] = answers.get(q["question"])
    return questions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(IN_PATH):
        print(f"Не найден {IN_PATH} — сначала python gen_dataset.py && python labels.py")
        return 1

    db = load_db(DB_PATH)
    card_db = db.get("minions", {})
    rows = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            snap = rec["snapshot"]
            text = encode_text(snap, card_db=card_db)
            questions = encode_questions(snap, rec.get("labels"))
            rows.append({"text": text, "questions": questions})

    lens = [len(r["text"]) for r in rows]
    answered = sum(1 for r in rows for q in r["questions"] if q.get("answer"))
    total_q = sum(len(r["questions"]) for r in rows)
    print(f"записей: {len(rows)}")
    if lens:
        print(f"длина текста: min {min(lens)}, средн {sum(lens)//len(lens)}, "
              f"max {max(lens)} символов")
    print(f"ответов на вопросы: {answered}/{total_q}")

    if not args.stats:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"-> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
