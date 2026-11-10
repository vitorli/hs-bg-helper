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

# каскад «классификатор -> LLM» — импорты опциональны: без них советчик
# работает в режиме LLM, как раньше
try:
    from classifier import BGClassifier, COMP_CLASSES
    from features import encode_text
    _CLF_AVAILABLE = True
except Exception:
    _CLF_AVAILABLE = False


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

    # классификатор грузим в фоне: инициализация энкодера ~полминуты,
    # оверлей и лог-наблюдатель работают сразу
    clf_holder = {"clf": None}
    clf_conf = config.get("clf_threshold", 0.6)

    def _load_clf():
        if not (_CLF_AVAILABLE and config.get("fast_mode", True)):
            return
        try:
            clf_holder["clf"] = BGClassifier(os.path.join(
                BASE_DIR, config.get("clf_path", "checkpoints/bg_clf")))
        except Exception as e:
            overlay.signals.status_updated.emit(f"Классификатор недоступен: {e}")

    threading.Thread(target=_load_clf, daemon=True).start()

    def fast_advice(snap: dict, hist_txt: str):
        """Каскад, шаг 1: мгновенный совет классификатора при высокой
        уверенности. Возвращает (текст, уверенность) или None."""
        clf = clf_holder["clf"]
        if clf is None:
            return None
        text = encode_text(snap, history=hist_txt)
        p = clf.predict(text, shop_size=len(snap.get("shop", [])))
        buy, buy_conf = p["buy"]
        comp, comp_conf = p["comp"]
        if buy_conf < clf_conf:
            return None
        lines = ["⚡ Быстрый совет (классификатор):"]
        if buy:
            name = snap["shop"][buy - 1]["name"] if buy <= len(snap.get("shop", [])) else f"слот №{buy}"
            lines.append(f"1) КУПИТЬ: слот №{buy} — {name}.")
        else:
            lines.append("1) КУПИТЬ: ничего.")
        if p["upgrade"][0] == 1:
            lines.append("2) Апнуть таверну.")
        if p["reroll"][0] == 1:
            lines.append("3) Рероллить остаток золота.")
        if p["freeze"][0] == 1:
            lines.append("4) Заморозить магазин.")
        if comp:
            lines.append(f"Сборка: {COMP_CLASSES[comp]}.")
        lines.append(f"(уверенность {buy_conf:.2f}; 💡 Совет — глубокий разбор LLM)")
        return "\n".join(lines), buy_conf

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
            hist_txt = history.summary()
            # каскад: сначала классификатор (~0.1 с), LLM — если неуверенен
            try:
                fast = fast_advice(snap, hist_txt)
            except Exception:
                fast = None
            if fast:
                text, conf = fast
                last_advice["key"] = key
                last_advice["text"] = text
                overlay.signals.advice_updated.emit(text)
                overlay.signals.status_updated.emit(
                    f"⚡ классификатор, уверенность {conf:.2f}")
                busy["flag"] = False
                log_dataset("advice.jsonl",
                            {"snapshot": snap, "history": hist_txt,
                             "advice": text, "source": "classifier"})
                return
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
                log_dataset("advice.jsonl", {"snapshot": snap, "history": hist_txt,
                                             "advice": text, "source": "llm"})
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
