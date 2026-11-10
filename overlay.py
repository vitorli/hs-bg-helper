"""
overlay.py — полупрозрачное окно-оверлей поверх Hearthstone (PySide6).

Показывает состояние игры из лога, советы локальной LLM и иконки существ.
Перетаскивается за заголовок, сворачивается в компактную полоску.
"""

import re

from PySide6.QtCore import Qt, QPoint, Signal, QObject
from PySide6.QtGui import QFont, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextBrowser, QSizePolicy, QSizeGrip,
)

import art_cache


class OverlaySignals(QObject):
    state_updated = Signal(dict)
    advice_updated = Signal(str)
    advice_started = Signal()
    advice_chunk = Signal(str)
    status_updated = Signal(str)
    art_ready = Signal(str)


STYLE = """
QWidget {
    background-color: rgba(18, 20, 26, 230);
    color: #e8e6e3;
    font-family: 'Segoe UI', sans-serif;
    font-size: 12px;
}
QLabel#title { color: #f0c96a; font-weight: bold; font-size: 13px; }
QLabel#state { color: #9fd3ff; }
QLabel#section { color: #f0c96a; font-weight: bold; margin-top: 4px; }
QLabel#minion { color: #e8e6e3; }
QTextBrowser {
    background-color: rgba(10, 11, 15, 200);
    border: 1px solid #3a3f4b;
    border-radius: 6px;
    padding: 6px;
}
QPushButton {
    background-color: #2d3340;
    border: 1px solid #4a5163;
    border-radius: 5px;
    padding: 4px 10px;
    color: #e8e6e3;
}
QPushButton:hover { background-color: #3a4253; }
QPushButton#advise { background-color: #6b4f1d; border-color: #f0c96a; color: #ffe9b0; font-weight: bold; }
QPushButton#advise:hover { background-color: #86632a; }
"""


def _fmt_minion(m: dict) -> str:
    name = m.get("name") or m.get("card_id") or "?"
    if m.get("kind") == "spell":
        t = f" т{m['tier']}" if m.get("tier") else ""
        return f"✨ {name}{t}"
    g = "★" if m.get("golden") else ""
    t = f" т{m['tier']}" if m.get("tier") else ""
    return f"{name}{g} {m.get('atk', 0)}/{m.get('hp', 0)}{t}"


class Overlay(QWidget):
    def __init__(self, config: dict):
        super().__init__()
        self.signals = OverlaySignals()
        self._drag_pos = QPoint()
        self._collapsed = False
        self.on_advise_click = None

        flags = Qt.FramelessWindowHint | Qt.Tool
        if config.get("overlay", {}).get("always_on_top", True):
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(config.get("overlay", {}).get("opacity", 0.93))

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(4)

        # Заголовок + кнопки
        top = QHBoxLayout()
        self.title = QLabel("🍺 BG Советник")
        self.title.setObjectName("title")
        top.addWidget(self.title)
        top.addStretch(1)
        self.btn_collapse = QPushButton("—")
        self.btn_collapse.setFixedWidth(30)
        self.btn_collapse.clicked.connect(self.toggle_collapse)
        self.btn_close = QPushButton("✕")
        self.btn_close.setFixedWidth(30)
        self.btn_close.clicked.connect(QApplication.quit)
        top.addWidget(self.btn_collapse)
        top.addWidget(self.btn_close)
        root.addLayout(top)

        self.body = QWidget()
        bl = QVBoxLayout(self.body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(3)

        self.lbl_state = QLabel("Ожидание Hearthstone…")
        self.lbl_state.setObjectName("state")
        bl.addWidget(self.lbl_state)

        self.shop_box = QVBoxLayout()
        self.shop_box.setSpacing(2)
        bl.addLayout(self.shop_box)

        self.board_box = QVBoxLayout()
        self.board_box.setSpacing(2)
        bl.addLayout(self.board_box)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #8a8f9c; font-style: italic;")
        bl.addWidget(self.lbl_status)

        self.advice = QTextBrowser()
        self.advice.setMinimumHeight(120)
        self.advice.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bl.addWidget(self.advice)

        self.btn_advise = QPushButton("💡 Совет")
        self.btn_advise.setObjectName("advise")
        self.btn_advise.clicked.connect(self._advise_clicked)
        bl.addWidget(self.btn_advise)

        root.addWidget(self.body)
        # уголок для растягивания окна
        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.addStretch(1)
        self.grip = QSizeGrip(self)
        grip_row.addWidget(self.grip, 0, Qt.AlignBottom | Qt.AlignRight)
        root.addLayout(grip_row)
        self.resize(config.get("overlay", {}).get("width", 400), 420)
        self.move(30, 60)
        self.setStyleSheet(STYLE)

        self._last_snap = None
        self._recommended = set()  # номера слотов магазина из совета
        self.signals.state_updated.connect(self._on_state)
        self.signals.advice_updated.connect(self._on_advice)
        self.signals.advice_started.connect(self._on_advice_started)
        self.signals.advice_chunk.connect(self._on_advice_chunk)
        self.signals.status_updated.connect(self.lbl_status.setText)
        self.signals.art_ready.connect(lambda _cid: self._render_lists())
        art_cache.add_listener(self.signals.art_ready.emit)

    # ---------- UI callbacks ----------

    def _advise_clicked(self):
        if self.on_advise_click:
            self.on_advise_click()

    def toggle_collapse(self):
        self._collapsed = not self._collapsed
        self.body.setVisible(not self._collapsed)
        self.btn_collapse.setText("+" if self._collapsed else "—")
        self.adjustSize()

    def _on_state(self, snap: dict):
        self._last_snap = snap
        phase = "Закупка" if snap.get("phase") == "recruit" else "Бой"
        self.lbl_state.setText(
            f"Ход {snap.get('turn', '?')} • {phase} • "
            f"Таверна {snap.get('tier', '?')} • Золото {snap.get('gold', '?')} • "
            f"HP {snap.get('health', '?')}+{snap.get('armor', 0)}"
        )
        self._render_lists()

    def _clear_box(self, box: QVBoxLayout):
        while box.count():
            item = box.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _on_advice_started(self):
        self.advice.clear()

    def _on_advice_chunk(self, tok: str):
        self.advice.moveCursor(QTextCursor.MoveOperation.End)
        self.advice.insertPlainText(tok)

    def _on_advice(self, text: str):
        self.advice.setPlainText(text)
        # вытаскиваем номера рекомендованных слотов: «№2», «слот 2», «#2»
        rec = set()
        for m in re.finditer(r"[№#]\s*(\d)\b|слот\D{0,5}(\d)\b|купить\D{0,10}(\d)\b", text, re.IGNORECASE):
            n = int(next(g for g in m.groups() if g))
            if 1 <= n <= 7:
                rec.add(n)
        self._recommended = rec
        self._render_lists()

    def _minion_row(self, m: dict, icon_w: int, icon_h: int,
                    prefix: str = "", highlight: bool = False) -> QWidget:
        row = QWidget()
        if highlight:
            row.setStyleSheet(
                "background-color: rgba(46, 140, 64, 130); border-radius: 6px;")
        lay = QHBoxLayout(row)
        lay.setContentsMargins(2, 1, 2, 1)
        lay.setSpacing(6)

        path = art_cache.ensure_art(m.get("card_id", ""))
        if path:
            pm = QPixmap(path)
            if not pm.isNull():
                icon = QLabel()
                icon.setFixedSize(icon_w, icon_h)
                icon.setPixmap(pm.scaled(
                    icon_w, icon_h,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
                lay.addWidget(icon)

        txt = QLabel(prefix + _fmt_minion(m))
        txt.setObjectName("minion")
        if highlight:
            txt.setStyleSheet("font-weight: bold; color: #d9ffd9;")
        lay.addWidget(txt, 1)
        return row

    def _render_lists(self):
        snap = self._last_snap
        if snap is None:
            return
        self._clear_box(self.shop_box)
        self._clear_box(self.board_box)

        shop = snap.get("shop", [])
        hdr = QLabel("🛒 Магазин Боба:" + (" ❄ заморожен" if snap.get("shop_frozen") else "")
                     if shop else "🛒 Магазин: —")
        hdr.setObjectName("section")
        self.shop_box.addWidget(hdr)
        for i, m in enumerate(shop, 1):
            self.shop_box.addWidget(
                self._minion_row(m, 40, 40, f"{i}. ", i in self._recommended))

        board = snap.get("board", [])
        hdr = QLabel("⚔ Доска:" if board else "⚔ Доска: —")
        hdr.setObjectName("section")
        self.board_box.addWidget(hdr)
        for m in board:
            self.board_box.addWidget(self._minion_row(m, 30, 30))

    # ---------- перетаскивание ----------

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton and not self._drag_pos.isNull():
            self.move(e.globalPosition().toPoint() - self._drag_pos)
