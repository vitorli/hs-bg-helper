"""
gen_dataset.py — офлайн-генератор датасета «снапшот → совет» из старых Power.log.

Прогоняет сохранённые логи через GameState, собирает снапшоты закупки
(с полным контекстом: история матча, племена пула, сводка боя) и спрашивает
локальную LLM совет на каждый — ровно то, что в живой игре делает main.py.
Результат дозаписывается в dataset/advice.jsonl для будущей дистилляции
в классификатор (Laya).

Запуск:
    python gen_dataset.py                    # все логи из config.hearthstone_logs_dir
    python gen_dataset.py --max-per-log 20   # по N снапшотов с лога (пробный)
    python gen_dataset.py --log <путь>       # конкретный Power.log

Требует запущенной Ollama (ollama serve).
"""

import argparse
import glob
import json
import os
import sys
import time

from advisor import OllamaAdvisor, load_db
from history import MatchHistory
from log_watcher import GameState

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DB_PATH = os.path.join(BASE_DIR, "bg_db.json")
OUT_PATH = os.path.join(BASE_DIR, "dataset", "advice.jsonl")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_snapshots(path: str, every_lines: int = 400):
    """Прогоняет лог, возвращает уникальные снапшоты закупки с полным магазином."""
    state, history = GameState(), MatchHistory()
    seen = set()
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            state.feed_line(line)
            if i % every_lines:
                continue
            snap = state.snapshot()
            history.update(snap)
            if snap.get("phase") != "recruit" or len(snap.get("shop", [])) < 3:
                continue
            key = (snap.get("hero"), snap.get("turn"), snap.get("tier"),
                   snap.get("gold"),
                   tuple(m["card_id"] for m in snap.get("shop", [])))
            if key in seen:
                continue
            seen.add(key)
            out.append({"snapshot": snap, "history": history.summary()})
    return out


def snap_key(snap: dict) -> str:
    return json.dumps({
        "hero": snap.get("hero"), "turn": snap.get("turn"),
        "tier": snap.get("tier"), "gold": snap.get("gold"),
        "shop": sorted(m["card_id"] for m in snap.get("shop", [])),
    }, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="конкретный Power.log")
    ap.add_argument("--max-per-log", type=int, default=0,
                    help="ограничить число снапшотов с лога (0 = без лимита)")
    ap.add_argument("--every-lines", type=int, default=400,
                    help="как часто опрашивать состояние (строк лога)")
    args = ap.parse_args()

    config = load_config()
    db = load_db(DB_PATH)
    advisor = OllamaAdvisor(
        config["ollama_url"], config["model"],
        upgrade_costs=config.get("tavern_upgrade_costs"),
    )
    if not advisor.available():
        print(f"Ollama не отвечает на {config['ollama_url']} — запустите ollama serve.")
        return 1

    if args.log:
        logs = [args.log]
    else:
        logs = sorted(glob.glob(os.path.join(
            config["hearthstone_logs_dir"], "Hearthstone_*", "Power.log")))
    if not logs:
        print("Логи не найдены.")
        return 1

    # уже размеченные — не повторяем при повторном запуске
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    done.add(snap_key(json.loads(ln)["snapshot"]))
                except (ValueError, KeyError):
                    pass

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    total = ok = fail = 0
    t0 = time.time()
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for path in logs:
            items = collect_snapshots(path, every_lines=args.every_lines)
            if args.max_per_log:
                items = items[:args.max_per_log]
            print(f"{os.path.basename(os.path.dirname(path))}: {len(items)} снапшотов")
            for item in items:
                total += 1
                snap = item["snapshot"]
                if snap_key(snap) in done:
                    continue
                try:
                    advice = advisor.advise(snap, db, history=item["history"])
                except Exception as e:
                    fail += 1
                    print(f"  ошибка: {e}", file=sys.stderr)
                    continue
                out.write(json.dumps(
                    {"snapshot": snap, "history": item["history"], "advice": advice},
                    ensure_ascii=False) + "\n")
                out.flush()
                ok += 1
                done.add(snap_key(snap))
                print(f"  [{ok}/{total}] ход {snap.get('turn')} тир {snap.get('tier')} "
                      f"— {advice.splitlines()[0][:60] if advice else '(пусто)'}")
    print(f"\nГотово: {ok} советов, {fail} ошибок, {time.time() - t0:.0f} сек. -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
