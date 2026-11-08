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

    last_snapshot = {"snap": None}
    busy = {"flag": False}

    def request_advice(auto: bool = False):
        snap = last_snapshot["snap"]
        if not snap:
            overlay.signals.status_updated.emit("Нет данных из лога — начните партию BG.")
            return
        if busy["flag"]:
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
            try:
                text = advisor.advise(snap, db)
                overlay.signals.advice_updated.emit(text)
                overlay.signals.status_updated.emit("")
            except Exception as e:
                overlay.signals.status_updated.emit(f"Ошибка Ollama: {e}")
            finally:
                busy["flag"] = False

        threading.Thread(target=worker, daemon=True).start()

    overlay.on_advise_click = lambda: request_advice(auto=False)

    def on_log_update(snap: dict):
        prev = last_snapshot["snap"]
        last_snapshot["snap"] = snap
        overlay.signals.state_updated.emit(snap)
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
