"""
test_e2e.py — сквозной тест: парсер Power.log → промпт → совет Ollama.

Требует запущенный `ollama serve`.
"""

import glob
import json
import os
import sys

from log_watcher import GameState
from advisor import OllamaAdvisor, load_db, build_prompt

BASE = os.path.dirname(os.path.abspath(__file__))


def _load_cfg() -> dict:
    with open(os.path.join(BASE, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def latest_recruit_snapshot(path: str, tail_mb: float = 70) -> dict:
    """Парсит хвост лога от последнего CREATE_GAME и возвращает снапшот
    в середине последней фазы закупки."""
    import re as _re
    size = os.path.getsize(path)
    offset = max(0, size - int(tail_mb * 1024 * 1024))
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    starts = [m.start() for m in _re.finditer(rb"CREATE_GAME", data)]
    if starts:
        idx = starts[-2] if len(starts) >= 2 else starts[0]
        nl = data.rfind(b"\n", 0, idx)
        offset = offset + (nl + 1 if nl >= 0 else idx)

    st = GameState()
    best = None  # последний снапшот фазы закупки с непустым магазином
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        for i, line in enumerate(f, 1):
            st.feed_line(line)
            if i % 2000 == 0:
                snap = st.snapshot()
                if snap["phase"] == "recruit" and len(snap["shop"]) >= 3 and snap["gold"]:
                    best = snap
    return best or st.snapshot()


def main():
    cfg = _load_cfg()
    logs = glob.glob(os.path.join(cfg["hearthstone_logs_dir"], "Hearthstone_*", "Power.log"))
    path = max(logs, key=os.path.getmtime)
    print("Лог:", path)

    snap = latest_recruit_snapshot(path)
    print(f"\n=== СОСТОЯНИЕ ===")
    print(f"Ход {snap['turn']}, фаза {snap['phase']}, таверна {snap['tier']}, "
          f"золото {snap['gold']}, HP {snap['health']}+{snap['armor']}, герой {snap['hero']}")
    print("Магазин:", ", ".join(f"{m['name']} {m['atk']}/{m['hp']}" for m in snap["shop"]))
    print("Доска:", ", ".join(f"{m['name']} {m['atk']}/{m['hp']}" for m in snap["board"]))

    db = load_db(os.path.join(BASE, "bg_db.json"))
    advisor = OllamaAdvisor(cfg["ollama_url"], cfg["model"])
    if not advisor.available():
        print("\nOllama не отвечает — запустите ollama serve")
        sys.exit(1)

    print("\n=== ПРОМПТ (сокр.) ===")
    print(build_prompt(snap, db)[:800])
    print("\n=== СОВЕТ LLM ===")
    print(advisor.advise(snap, db))


if __name__ == "__main__":
    main()
