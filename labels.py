"""
labels.py — конвертация текстовых советов LLM в типизированные метки
для обучения decision-классификатора (Laya и аналоги).

Из каждой записи dataset/advice.jsonl извлекает:
    buy_slot  — какой слот магазина купить (0 = ничего не покупать)
    reroll    — рероллить ли остаток золота
    upgrade   — апнуть таверну
    freeze    — заморозить магазин
    comp      — целевое племя сборки (русское название или '')

Эвристики осознанно консервативны: сомнительное не размечаем (None),
чтобы не учить классификатор на шуме.

Запуск:
    python labels.py              # dataset/advice.jsonl -> dataset/labels.jsonl
    python labels.py --stats      # сводка по покрытию меток
"""

import argparse
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(BASE_DIR, "dataset", "advice.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "dataset", "labels.jsonl")

# племена, которые классификатор может предсказывать как цель сборки
COMP_TARGETS = [
    "Нежить", "Механизм", "Демон", "Дракон", "Зверь", "Пират",
    "Мурлок", "Элементаль", "Свинобраз", "Нага", "Аберрация",
]

SLOT_RE = re.compile(r"[№#]\s*(\d)\b|слот\D{0,5}(\d)\b|купить\D{0,10}(\d)\b")
NO_BUY_RE = re.compile(r"ничего\s*не\s*куп|не\s+покуп|без\s+покуп|ничего,|ничего\.")


SECTION_NAMES = ("купить", "сборка", "почему", "дальше")
HEADER_RE = re.compile(
    r"^\s*(?:\d+\s*[).]?\s*)?(купить|сборка|почему|дальше)\s*:?\s*(.*)$",
    re.IGNORECASE)


def _section(text: str, name: str) -> str:
    """Текст пункта совета. Терпим к формату: «1) КУПИТЬ: ...», «КУПИТЬ 2»,
    «Купить: ...» — заголовок с номером/двоеточием или без."""
    name = name.lower()
    parts = []
    in_section = False
    for ln in text.splitlines():
        m = HEADER_RE.match(ln)
        if m:
            if m.group(1).lower() == name:
                in_section = True
                parts.append(m.group(2))
            elif in_section:
                break
        elif in_section:
            parts.append(ln)
    return "\n".join(parts).strip()


def extract_labels(rec: dict) -> dict | None:
    snap, advice = rec.get("snapshot", {}), rec.get("advice", "")
    if not advice or snap.get("phase") != "recruit":
        return None
    low = advice.lower()
    shop_size = len(snap.get("shop", []))
    gold = snap.get("gold", 0) or 0
    tier = snap.get("tier", 1) or 1

    # --- покупка: слоты из пункта КУПИТЬ, отсекаем невозможное по золоту
    buy_text = _section(advice, "купить").lower()
    slots = set()
    if buy_text and not NO_BUY_RE.search(buy_text):
        for m in SLOT_RE.finditer(buy_text):
            n = int(next(g for g in m.groups() if g))
            if 1 <= n <= max(shop_size, 7):
                slots.add(n)
        # модель иногда пишет просто «КУПИТЬ 2» без слова «слот»
        bare = re.fullmatch(r"(\d)\s*[.)]?", buy_text.strip())
        if bare and 1 <= int(bare.group(1)) <= max(shop_size, 7):
            slots.add(int(bare.group(1)))
    max_affordable = gold // 3
    slots = {s for s in slots if s <= max_affordable or max_affordable == 0 and False}
    if not slots and buy_text and not NO_BUY_RE.search(buy_text) and shop_size:
        # упомянул карту без номера — метку слота не ставим (шум)
        pass
    buy_slot = min(slots) if len(slots) == 1 else (0 if not slots else None)

    # --- ап таверны
    up_re = re.compile(r"\bап\b|апнуть|апгрейд|повысить\s+таверн|поднять\s+таверн")
    upgrade = None
    if tier < 6:
        upgrade = bool(up_re.search(low))
        # ап «на следующий ход» — не ап прямо сейчас
        if upgrade and re.search(r"ап\w*\D{0,15}(на\s+след|потом|позже|на\s+будущ)", low):
            upgrade = False

    # --- реролл остатка (не «не рероллить»)
    reroll = None
    if re.search(r"реролл|перекрут", low):
        reroll = not bool(re.search(r"не\s+реролл|без\s+реролл|не\s+перекрут", low))

    # --- заморозка
    freeze = None
    if re.search(r"замороз", low):
        freeze = not bool(re.search(r"не\s+замороз|размороз", low))

    # --- цель сборки: племя из пункта СБОРКА
    comp_text = _section(advice, "сборка")
    comp = ""
    for tribe in COMP_TARGETS:
        if re.search(rf"\b{re.escape(tribe.lower())}", comp_text.lower()):
            comp = tribe
            break

    return {"buy_slot": buy_slot, "reroll": reroll,
            "upgrade": upgrade, "freeze": freeze, "comp": comp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="только сводка, файл не писать")
    args = ap.parse_args()

    if not os.path.exists(IN_PATH):
        print(f"Не найден {IN_PATH} — сначала python gen_dataset.py")
        return 1

    n = labeled = 0
    coverage = {"buy_slot": 0, "reroll": 0, "upgrade": 0, "freeze": 0, "comp": 0}
    rows = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            n += 1
            labels = extract_labels(rec)
            if labels is None:
                continue
            labeled += 1
            for k, v in labels.items():
                if v not in (None, "", 0) or k == "buy_slot" and v == 0:
                    if v is not None and v != "":
                        coverage[k] += 1
            rows.append({"snapshot": rec["snapshot"], "labels": labels})

    print(f"записей: {n}, размечено: {labeled}")
    for k, c in coverage.items():
        print(f"  {k}: {c} ({100 * c / max(labeled, 1):.0f}%)")

    if not args.stats:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"-> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
