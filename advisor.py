"""
advisor.py — формирует промпт из состояния игры и спрашивает локальную LLM через Ollama.
"""

import json
import os
import re

import requests

TRIBE_RU = {
    "QUILBOAR": "Свинобраз",
    "DRAGON": "Дракон",
    "ABERRATION": "Аберрация",
    "PIRATE": "Пират",
    "ELEMENTAL": "Элементаль",
    "DEMON": "Демон",
    "MURLOC": "Мурлок",
    "MECH": "Механизм",
    "BEAST": "Зверь",
    "UNDEAD": "Нежить",
    "NAGA": "Нага",
    "ALL": "Все племена",
}

TAVERN_MAX_TIER = 6
# Стоимость повышения таверны до тира (2→6), золота. Можно переопределить
# ключом "tavern_upgrade_costs" в config.json.
UPGRADE_COSTS = {2: 4, 3: 5, 4: 7, 5: 8, 6: 9}
MAX_RETRIES = 2  # повторные запросы при ошибках в ответе

SYSTEM_PROMPT_RU = (
    "Ты — тренер по режиму «Поля сражений» (Battlegrounds) в Hearthstone, знаешь актуальную мету "
    "(патч 37.0.2: тип Аберрации с Божествами, безделушки, Тёмный парадокс, наги выведены из пула). "
    "Тебе дают текущее состояние игры: таверну, золото, здоровье, магазин Боба, доску и руку игрока. "
    "Ниже в промпте даны справка по мете и правила стратегии — следуй им, но адаптируй под конкретное состояние. "
    "Отвечай на русском, коротко и конкретно, строго в формате:\n"
    "1) КУПИТЬ — что взять из магазина прямо сейчас: ОБЯЗАТЕЛЬНО номер слота и полное имя карты "
    "(например: «слот №2 — Гроза кабанов»), включая заклинания; если покупать нечего — "
    "напиши «ничего, апнуть таверну / заморозить / рероллить»;\n"
    "2) СБОРКА — какую комбинацию собирать под текущую доску (племя/синергия и 2-3 ключевых существа цели);\n"
    "3) ПОЧЕМУ — одно-два предложения обоснования;\n"
    "4) ДАЛЬШЕ — что делать с оставшимся золотом и на следующий ход.\n"
    "СТРОГО: в пункте КУПИТЬ можно называть ТОЛЬКО карты из списка «МАГАЗИН БОБА» ниже. "
    "Заклинания в магазине — такие же покупки, учитывай их наравне с существами. "
    "ВАЖНО: сборку выбирай по РЕАЛЬНОЙ доске — какое племя уже собрано большинством существ "
    "и что предлагает магазин. Не советуй сборку из справки, если доска ей не соответствует; "
    "перестройку на другое племя предлагай только если доска слабая и есть конкретная причина. "
    "Правила экономики: покупка существа = 3 золота, реролл = 1, продажа = 1 золото; "
    "не советуй апнуть таверну, если после апа не останется золота на покупку и доска слабее магазина; "
    "два одинаковых существа — приоритет: триплет делает золотое существо и открывает существо на тир выше; "
    "не советуй покупать существо, которое слабее любого на доске и не усиливает синергию племени. "
    "Не выдумывай карты, которых нет в состоянии или справке. Если данных мало — скажи об этом честно. "
    "ФОРМАТ ОТВЕТА — ЖЁСТКО: ровно четыре пункта, каждый одной строкой:\n"
    "1) КУПИТЬ: ...\n2) СБОРКА: ...\n3) ПОЧЕМУ: ...\n4) ДАЛЬШЕ: ...\n"
    "ЗАПРЕЩЕНО: markdown (** или ##), эмодзи, заголовки, списки внутри пунктов, "
    "вводные фразы, вопросы игроку, итоги. Только четыре строки, ничего больше.\n"
    "\n"
    "ПРИМЕР правильного ответа (другая партия — формат образец, содержание не копируй):\n"
    "1) КУПИТЬ: слот №2 — Певица шипов (тир 3).\n"
    "2) СБОРКА: Свинобразы через кровавые самоцветы: Певица шипов + Пророк кабана.\n"
    "3) ПОЧЕМУ: На доске уже два свинобраза, а слот №2 закрывает нехватку генератора самоцветов.\n"
    "4) ДАЛЬШЕ: оставшиеся 4 з. — реролл в поисках Пророка кабана, ап таверны не нужен."
)


def load_db(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _fmt_minion(m: dict, card_db: dict | None = None,
                texts_by_name: dict | None = None) -> str:
    parts = [m.get("name") or m.get("card_id") or "?"]
    if m.get("kind") == "spell":
        parts.append("[заклинание]")
    else:
        tribe = TRIBE_RU.get(m.get("tribe", ""), m.get("tribe") or "")
        if tribe and tribe != "0":
            parts.append(f"[{tribe}]")
        if m.get("golden"):
            parts.append("(золотое)")
        parts.append(f"{m.get('atk', 0)}/{m.get('hp', 0)}")
    if m.get("tier"):
        parts.append(f"тир {m['tier']}")
    # описание карты из базы — модель должна понимать, что карта делает
    text = None
    if card_db:
        info = card_db.get(m.get("card_id", ""))
        if info and info.get("text"):
            text = info["text"]
    if not text and texts_by_name:
        text = texts_by_name.get(m.get("name", ""))
    if text:
        parts.append(f"— {text}")
    return " ".join(parts)


def build_prompt(snapshot: dict, db: dict) -> str:
    ref = []
    notes = db.get("meta_notes_ru", [])
    if notes:
        ref.append("СПРАВКА ПО СТРАТЕГИИ И МЕТЕ:")
        for n in notes:
            ref.append(f"- {n}")

    strategy = db.get("strategy_ru", [])
    if strategy:
        ref.append("\nПРАВИЛА: КАК ВЫБИРАТЬ СТРАТЕГИЮ И ДЕЙСТВОВАТЬ:")
        for n in strategy:
            ref.append(f"- {n}")

    comps = db.get("comps_ru", {})
    if comps:
        ref.append("\nСИЛЬНЫЕ СБОРКИ МЕТЫ (это справочник, а НЕ указание — выбирай по доске):")
        for name, desc in comps.items():
            ref.append(f"- {name}: {desc}")

    heroes = db.get("heroes_ru", {})
    if heroes:
        ref.append("\nГЕРОИ МЕТЫ:")
        for name, desc in heroes.items():
            ref.append(f"- {name}: {desc}")

    lines = []
    if ref:
        lines.append("\n".join(ref))
        lines.append("\n===== ТЕКУЩЕЕ СОСТОЯНИЕ ИГРЫ (советуй ТОЛЬКО по нему) =====")

    phase = "ЗАКУПКА (фаза магазина)" if snapshot.get("phase") == "recruit" else "БОЙ"
    lines.append(f"Фаза: {phase}. Ход №{snapshot.get('turn', '?')}.")
    lines.append(
        f"Герой: {snapshot.get('hero') or 'неизвестен'} | "
        f"HP: {snapshot.get('health', '?')}+{snapshot.get('armor', 0)} брони | "
        f"Таверна: тир {snapshot.get('tier', '?')} | Золото: {snapshot.get('gold', '?')}"
    )

    minion_db = db.get("minions", {})
    texts_by_name = db.get("texts_by_name_ru", {})

    shop = snapshot.get("shop", [])
    if shop:
        lines.append("МАГАЗИН БОБА (в ответе называй номер слота):")
        for i, m in enumerate(shop, 1):
            lines.append(f"  {i}. {_fmt_minion(m, minion_db, texts_by_name)}")
    else:
        lines.append("МАГАЗИН БОБА: пусто или не виден.")

    board = snapshot.get("board", [])
    if board:
        # сводка по племенам доски — чтобы модель видела, что УЖЕ собрано
        tribes = {}
        for m in board:
            t = TRIBE_RU.get(m.get("tribe", ""), m.get("tribe") or "без племени")
            if t == "0":
                t = "без племени"
            tribes[t] = tribes.get(t, 0) + 1
        top = sorted(tribes.items(), key=lambda kv: -kv[1])
        lines.append("Племена моей доски: " + ", ".join(f"{t}×{n}" for t, n in top))
        dom_t, dom_n = top[0]
        if dom_t not in ("без племени", "Все племена") and dom_n * 2 >= len(board):
            lines.append(
                f"ВАЖНО: основа моей доски — «{dom_t}» ({dom_n} из {len(board)} существ). "
                f"По умолчанию продолжай эту сборку; перестройку на другое племя "
                f"предлагай только при очень веской причине и назови её."
            )
        lines.append("МОЯ ДОСКА (слева направо):")
        for i, m in enumerate(board, 1):
            lines.append(f"  {i}. {_fmt_minion(m, minion_db, texts_by_name)}")
    else:
        lines.append("МОЯ ДОСКА: пуста.")

    hand = snapshot.get("hand", [])
    if hand:
        lines.append("РУКА:")
        for i, m in enumerate(hand, 1):
            lines.append(f"  {i}. {_fmt_minion(m)}")

    lines.append("\nДай совет по текущей фазе. Строго четыре пункта: КУПИТЬ / СБОРКА / ПОЧЕМУ / ДАЛЬШЕ, "
                 "каждый одной строкой, без markdown и вводных слов.")
    return "\n".join(lines)


class OllamaAdvisor:
    def __init__(self, url: str, model: str, system_prompt: str = SYSTEM_PROMPT_RU,
                 timeout: int = 120, upgrade_costs: dict | None = None):
        self.url = url.rstrip("/")
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.upgrade_costs = {int(k): v for k, v in (upgrade_costs or UPGRADE_COSTS).items()}

    def available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/api/version", timeout=3)
            return r.ok
        except requests.RequestException:
            return False

    def advise(self, snapshot: dict, db: dict) -> str:
        prompt = build_prompt(snapshot, db)
        text = self._generate(prompt)

        # валидация ответа: ловим жёсткие ошибки и просим модель исправиться
        errors = []
        for _ in range(MAX_RETRIES):
            errors = self._validate(text, snapshot)
            if not errors:
                break
            fix = (
                "\n\n===== ТВОЙ ПРЕДЫДУЩИЙ ОТВЕТ =====\n" + text +
                "\n===== ОШИБКИ В ОТВЕТЕ (обязательно исправь) =====\n"
                + "\n".join(f"- {e}" for e in errors) +
                "\nИсправь ошибки и дай финальный ответ заново, строго в том же формате. "
                "Невозможные действия (ап на несуществующий/максимальный тир, покупки без "
                "золота) просто замени на допустимые (покупка по золоту, реролл, заморозка) "
                "и не упоминай их в ответе."
            )
            text = self._generate(prompt + fix)

        # модель так и не исправилась — покажем ошибку рядом с советом,
        # чтобы пользователь не принял невозможное действие за чистую монету
        if errors:
            text += "\n\n⚠ " + " ".join(errors)

        # страховка от выдумок: секция КУПИТЬ должна ссылаться на карты магазина
        shop_names = [m.get("name", "") for m in snapshot.get("shop", []) if m.get("name")]
        if shop_names and snapshot.get("phase") == "recruit":
            lines = text.splitlines()
            buy_section = []
            for i, ln in enumerate(lines):
                if "КУПИТЬ" in ln.upper():
                    for ln2 in lines[i:]:
                        if ln2.strip().startswith(("2)", "2.")):
                            break
                        buy_section.append(ln2)
                    break
            buy_text = " ".join(buy_section).lower()
            mentioned = any(n.lower() in buy_text for n in shop_names)
            soft = any(w in buy_text for w in
                       ("апн", "замороз", "реролл", "ничего", "не покуп"))
            has_slot = bool(re.search(r"[№#]\s*\d|слот\D{0,5}\d|купить\D{0,10}\d", buy_text))
            if buy_section and not mentioned and not soft and not has_slot:
                text += ("\n\n⚠ Совет упоминает карту не из текущего магазина — "
                         "сверьтесь со списком слотов выше.")
        return text

    # ---------- генерация и валидация ----------

    def _generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
            # отключить thinking-режим у моделей типа qwen3 — нам нужен быстрый ответ
            "think": False,
            "options": {"temperature": 0.2, "num_ctx": 4096},
        }
        r = requests.post(f"{self.url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _validate(self, text: str, snapshot: dict) -> list[str]:
        """Жёсткие ошибки, после которых ответ надо перегенерировать."""
        errors = []
        low = text.lower()
        tier = snapshot.get("tier", 1)
        gold = snapshot.get("gold", 0)

        # 1) Таверна не бывает выше 6-го тира
        if re.search(r"тир\s*7|тира\s*7|7-го\s+тира|таверн[а-я]*\s+до\s+7\b|\bдо\s+7\s*тир", low):
            errors.append(f"Таверна максимум {TAVERN_MAX_TIER}-го тира — «тир 7» не существует.")
        elif tier >= TAVERN_MAX_TIER and re.search(
                r"\bап\b|апнуть|апгрейд|повысить\s+таверн|поднять\s+таверн", low):
            errors.append(
                f"Таверна уже максимального ({TAVERN_MAX_TIER}) тира — апнуть нельзя; "
                f"советуй покупку, реролл или заморозку.")

        # 2) Арифметика золота (только в фазе закупки; если советуешь продажу —
        #    золото от неё непредсказуемо, проверку пропускаем)
        if snapshot.get("phase") == "recruit" and "продаж" not in low and "продать" not in low:
            buy_section = []
            lines = text.splitlines()
            for i, ln in enumerate(lines):
                if "купить" in ln.lower():
                    for ln2 in lines[i:]:
                        if ln2.strip().startswith(("2)", "2.")):
                            break
                        buy_section.append(ln2)
                    break
            buy_text = " ".join(buy_section).lower()

            slots = set()
            for m in re.finditer(r"[№#]\s*(\d)\b|слот\D{0,5}(\d)\b|купить\D{0,10}(\d)\b", buy_text):
                n = int(next(g for g in m.groups() if g))
                if 1 <= n <= 7:
                    slots.add(n)
            if re.search(r"ничего\s*не\s*куп|не\s+покупать|без\s+покуп", buy_text):
                slots.clear()

            rerolls = 0
            for m in re.finditer(r"реролл", low):
                if not re.search(r"не\s+реролл|без\s+реролл", low[max(0, m.start() - 10):m.end()]):
                    rerolls += 1

            wants_upgrade = bool(re.search(
                r"\bап\b|апнуть|апгрейд|повысить\s+таверн|поднять\s+таверн", low))

            cost = 3 * len(slots)
            parts = []
            if slots:
                parts.append(f"покупки: {len(slots)} × 3 = {3 * len(slots)} з.")
            if wants_upgrade:
                if tier < TAVERN_MAX_TIER:
                    up = self.upgrade_costs.get(tier + 1, 0)
                    cost += up
                    parts.append(f"ап таверны: {up} з.")
                else:
                    pass  # ап при макс. тире уже пойман выше
            # реролл «на оставшееся золото» — это слив остатка, он всегда
            # доступен; учитываем рероллы только в пределах остатка
            remainder = gold - cost
            if remainder > 0 and rerolls:
                paid = min(rerolls, remainder)
                cost += paid
                parts.append(f"рероллы: {paid} з.")
            if cost > gold:
                errors.append(
                    f"Арифметика не сходится: план стоит {cost} з., а золота {gold} "
                    f"({'; '.join(parts)}). Сократи план или предложи продажу существа."
                )
        return errors
