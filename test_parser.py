"""
test_parser.py — офлайн-проверка парсера на хвосте существующего Power.log.

Запуск:  python test_parser.py [путь_к_Power.log] [размер_хвоста_в_МБ]
"""

import os
import re
import sys

from log_watcher import GameState


def main():
    default_log = None
    import glob
    candidates = glob.glob(r"G:\Hearthstone\Logs\Hearthstone_*\Power.log")
    if candidates:
        default_log = max(candidates, key=os.path.getmtime)

    path = sys.argv[1] if len(sys.argv) > 1 else default_log
    tail_mb = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    stop_line = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if not path or not os.path.exists(path):
        print("Power.log не найден"); sys.exit(1)

    size = os.path.getsize(path)
    offset = max(0, size - int(tail_mb * 1024 * 1024))

    # argv[4]: какой CREATE_GAME с конца брать (1 = последний, 2 = предпоследний,
    #          0 = НЕ отматывать — сырой хвост, вход посреди партии)
    cg_from_end = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    if cg_from_end > 0:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
        starts = [m.start() for m in re.finditer(rb"CREATE_GAME", data)]
        if starts:
            idx = starts[-cg_from_end] if len(starts) >= cg_from_end else starts[0]
            nl = data.rfind(b"\n", 0, idx)
            offset = offset + (nl + 1 if nl >= 0 else idx)
            print(f"CREATE_GAME #{cg_from_end} с конца, всего их в окне: {len(starts)}")
    else:
        with open(path, "rb") as f:
            f.seek(offset)
            f.readline()  # выкинуть обрыв строки
            offset = f.tell()
        print("Режим сырого хвоста (вход посреди партии)")

    state = GameState()
    best = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        n = 0
        for line in f:
            state.feed_line(line)
            n += 1
            if stop_line and n >= stop_line:
                break
            if n % 2000 == 0:
                s = state.snapshot()
                if s["phase"] == "recruit" and len(s["shop"]) >= 3:
                    best = s
    print(f"Строк обработано: {n}")

    snap = state.snapshot()
    if best:
        print("\n=== ЛУЧШИЙ СНАПШОТ ЗАКУПКИ (промежуточный) ===")
        print(f"turn={best['turn']} gold={best['gold']} tier={best['tier']} "
              f"hero={best['hero']} hp={best['health']}+{best['armor']} player={best['player']}")
        print("SHOP:", ", ".join(f"{m['name']}({m['tier']})" for m in best["shop"]))
        print("BOARD:", ", ".join(f"{m['name']}({m['tier']})" for m in best["board"]))
        print("HAND:", ", ".join(m['name'] for m in best["hand"]))
    print("\n=== СНАПШОТ ===")
    for k in ("turn", "phase", "gold", "tier", "hero", "health", "armor"):
        print(f"{k}: {snap[k]}")
    print("local_pid:", state.local_pid, "bob_pid:", state.bob_pid, "local_name:", state.local_name)
    for section in ("shop", "board", "hand"):
        print(f"\n{section.upper()} ({len(snap[section])}):")
        for m in snap[section]:
            print("  ", m)
    print(f"\nИзвестных cardId: {len(state.card_names)}")


if __name__ == "__main__":
    main()
