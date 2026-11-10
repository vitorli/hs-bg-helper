"""
history.py — краткая память текущего матча: золото/тир/HP по ходам и что
появлялось на доске.

Нужна советчику для контекста («добери третьего X», «ты два хода подряд
рероллишь в пустую») и датасету классификатора. Сброс — по смене героя
или убыванию номера хода (признак новой партии).
"""


class MatchHistory:
    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns   # сколько последних ходов попадает в промпт
        self._hero = None
        self._last_turn = 0
        self.turns = []              # dict: turn, gold, tier, hp, armor, board_ids, gained

    def _reset(self, hero):
        self._hero = hero
        self._last_turn = 0
        self.turns = []

    def update(self, snap: dict) -> None:
        hero = snap.get("hero") or None
        turn = snap.get("turn") or 0
        # новая партия — только по убыванию хода: имя героя в логе может
        # мигать (конец матча, наблюдение) и не должно стирать историю
        if turn and self._last_turn and turn < self._last_turn:
            self._reset(hero)
        elif self._hero is None and hero:
            self._hero = hero
        if snap.get("phase") != "recruit":
            return

        board = snap.get("board", [])
        shop_fp = tuple(m["card_id"] for m in snap.get("shop", []))
        gained = []
        if self.turns:
            prev_ids = set(self.turns[-1].get("board_ids", []))
            gained = [m["name"] for m in board
                      if m["card_id"] not in prev_ids and not m["card_id"].startswith("TB_")][:3]
        entry = {
            "turn": turn,
            "gold": snap.get("gold"),
            "tier": snap.get("tier"),
            "hp": snap.get("health"),
            "armor": snap.get("armor"),
            "shop_fp": shop_fp,
            "board_ids": [m["card_id"] for m in board],
            "gained": gained,
        }
        if self.turns and self.turns[-1].get("shop_fp") == shop_fp:
            # тот же магазин — обновляем запись (покупки/продажи внутри хода)
            self.turns[-1].update(entry)
        else:
            self.turns.append(entry)
        if turn:
            self._last_turn = max(self._last_turn, turn)
        if len(self.turns) > 12:
            self.turns = self.turns[-12:]

    def summary(self) -> str:
        """Компактная сводка последних ходов для промпта (или '')."""
        if len(self.turns) < 2:
            return ""
        # схлопываем серии одного хода (рероллы) — оставляем последнюю запись
        collapsed = []
        for t in self.turns:
            if collapsed and collapsed[-1]["turn"] == t["turn"]:
                collapsed[-1] = t
            else:
                collapsed.append(t)
        lines = ["ИСТОРИЯ МАТЧА (последние ходы):"]
        for t in collapsed[-self.max_turns:]:
            g = f"; на доске появились: {', '.join(t['gained'])}" if t["gained"] else ""
            lines.append(
                f"- ход {t['turn']}: тир {t['tier']}, {t['gold']} з., "
                f"HP {t['hp']}+{t['armor']}{g}"
            )
        return "\n".join(lines)
