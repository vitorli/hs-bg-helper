"""
art_cache.py — фоновая загрузка иконок существ по cardId с art.hearthstonejson.com.

Иконки складываются в папку art/ рядом со скриптом и кэшируются навсегда.
"""

import os
import threading

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "art")
URL = "https://art.hearthstonejson.com/v1/tiles/{card_id}.png"

_lock = threading.Lock()
_pending = set()
_listeners = []


def _notify(card_id: str):
    for cb in _listeners:
        try:
            cb(card_id)
        except Exception:
            pass


def _download(card_id: str):
    path = os.path.join(ART_DIR, f"{card_id}.png")
    try:
        r = requests.get(URL.format(card_id=card_id), timeout=15)
        if r.ok and len(r.content) > 1000:
            with open(path, "wb") as f:
                f.write(r.content)
            _crop(path)
    except requests.RequestException:
        pass
    finally:
        with _lock:
            _pending.discard(card_id)
        _notify(card_id)


def _crop(path: str):
    """Вырезает из широкого тайла квадрат с наиболее детализированным участком
    (по насыщенности и контрасту) — так в квадрат попадает морда существа."""
    try:
        from PIL import Image, ImageStat, ImageEnhance
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if w <= h:
            return
        side = h
        best_x, best_score = 0, -1.0
        step = max(1, (w - side) // 24)
        for x in range(0, w - side + 1, step):
            win = im.crop((x, 0, x + side, side))
            sat = ImageEnhance.Color(win).enhance(1.0).convert("HSV")
            s_chan = sat.getchannel("S")
            score = ImageStat.Stat(s_chan).mean[0] + ImageStat.Stat(win.convert("L")).stddev[0]
            if score > best_score:
                best_score, best_x = score, x
        im.crop((best_x, 0, best_x + side, side)).save(path)
    except Exception:
        pass  # обрезка необязательна — оставляем как есть


def ensure_art(card_id: str):
    """Гарантирует загрузку иконки (асинхронно). Возвращает путь, если уже скачана."""
    if not card_id:
        return None
    path = os.path.join(ART_DIR, f"{card_id}.png")
    if os.path.exists(path):
        return path
    with _lock:
        if card_id in _pending:
            return None
        _pending.add(card_id)
    os.makedirs(ART_DIR, exist_ok=True)
    threading.Thread(target=_download, args=(card_id,), daemon=True).start()
    return None


def add_listener(cb):
    """callback(card_id) вызывается (из фонового потока!) после загрузки."""
    _listeners.append(cb)
