"""
main.py — точка входа: лог-наблюдатель + Ollama + оверлей.

Запуск:  python main.py
Перед запуском: ollama serve (обычно уже запущен) и ollama pull qwen2.5:7b-instruct
"""

import json
import os
import sys
import threading

from PySide6.QtWidgets import QApplication

from log_watcher import LogWatcher
from advisor import OllamaAdvisor, load_db
from overlay import Overlay
from history import MatchHistory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DB_PATH = os.path.join(BASE_DIR, "bg_db.json")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    config = load_config()
    db = load_db(DB_PATH)

    advisor = OllamaAdvisor(
        config["ollama_url"], config["model"],
        upgrade_costs=config.get("tavern_upgrade_costs"),
    )

    app = QApplication(sys.argv)
    overlay = Overlay(config)
    overlay.show()

    history = MatchHistory(max_turns=config.get("history_turns", 4))

    # датасет «снапшот → совет» для будущей дистилляции в классификатор
    dataset_lock = threading.Lock()
    dataset_dir = os.path.join(BASE_DIR, "dataset")
    _logged_shops = {"key": None}

    def log_dataset(filename: str, obj: dict):
        if not config.get("dataset_log", True):
            return
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            with dataset_lock, open(os.path.join(dataset_dir, filename),
                                    "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except OSError:
            pass

    last_snapshot = {"snap": None}
    last_advice = {"key": None, "text": None}
    busy = {"flag": False}

    def _snap_key(snap: dict):
        shop_ids = tuple(sorted(m["card_id"] for m in snap.get("shop", [])))
        return (snap.get("turn"), snap.get("phase"), snap.get("gold"),
                snap.get("tier"), shop_ids, snap.get("shop_frozen"),
                tuple(m["card_id"] for m in snap.get("board", [])))

    def request_advice(auto: bool = False):
        snap = last_snapshot["snap"]
        if not snap:
            overlay.signals.status_updated.emit("Нет данных из лога — начните партию BG.")
            return
        if busy["flag"]:
            return
        key = _snap_key(snap)
        # тот же стейт, что в прошлый раз — не дёргаем LLM, показываем кэш
        if auto and key == last_advice["key"] and last_advice["text"]:
            overlay.signals.advice_updated.emit(last_advice["text"])
            return
        if not advisor.available():
            overlay.signals.status_updated.emit(
                "Ollama не отвечает на " + config["ollama_url"] + " — запустите её."
            )
            return

        def worker():
            busy["flag"] = True
            overlay.signals.status_updated.emit(
                ("Авто-совет… " if auto else "Думаю… ") + f"({config['model']})"
            )
            overlay.signals.advice_started.emit()
            try:
                text = advisor.advise(
                    snap, db, history=history.summary(),
                    on_token=overlay.signals.advice_chunk.emit,
                )
                last_advice["key"] = key
                last_advice["text"] = text
                overlay.signals.advice_updated.emit(text)
                overlay.signals.status_updated.emit("")
                log_dataset("advice.jsonl", {"snapshot": snap, "advice": text})
            except Exception as e:
                overlay.signals.status_updated.emit(f"Ошибка Ollama: {e}")
            finally:
                busy["flag"] = False

        threading.Thread(target=worker, daemon=True).start()

    overlay.on_advise_click = lambda: request_advice(auto=False)

    def on_log_update(snap: dict):
        prev = last_snapshot["snap"]
        last_snapshot["snap"] = snap
        history.update(snap)
        overlay.signals.state_updated.emit(snap)
        # логируем снапшоты закупки (дедуп по магазину/ходу) в датасет
        if snap.get("phase") == "recruit" and snap.get("shop"):
            shop_key = (snap.get("turn"), snap.get("gold"),
                        tuple(m["card_id"] for m in snap.get("shop", [])))
            if shop_key != _logged_shops["key"]:
                _logged_shops["key"] = shop_key
                log_dataset("snapshots.jsonl", snap)
        # авто-совет при начале фазы закупки / смене магазина
        if config.get("auto_advice") and snap.get("phase") == "recruit":
            new_shop = snap.get("phase") == "recruit" and (
                prev is None or prev.get("phase") != "recruit"
            )
            shop_changed = prev and prev.get("phase") == "recruit" and (
                [m["card_id"] for m in prev.get("shop", [])]
                != [m["card_id"] for m in snap.get("shop", [])]
            )
            if new_shop or shop_changed:
                request_advice(auto=True)

    watcher = LogWatcher(
        config["hearthstone_logs_dir"], on_log_update,
        hydrate_mb=config.get("hydrate_mb", 0),
    )
    watcher.start()

    app.aboutToQuit.connect(watcher.stop)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
