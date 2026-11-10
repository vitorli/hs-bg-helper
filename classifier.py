"""
classifier.py — быстрый энкодерный классификатор решений BG (наш аналог
System One): одна сводка состояния -> типизированные решения с уверенностью.

Энкодер — ModernBERT (тот, что внутри Laya multilingual, кэш HF),
поверх — 5 линейных голов: buy_slot (0..8, 0=ничего), upgrade, reroll,
freeze (да/нет), comp (племя или «держать текущую»).

Интеграция — каскад: уверенность выше порога -> мгновенный ответ
классификатора (~30-50 мс на CPU), иначе fallback на LLM.

Использование:
    from classifier import BGClassifier
    clf = BGClassifier("checkpoints/bg_clf")     # после train_classifier.py
    pred = clf.predict(state_text)               # {'buy_slot': (2, 0.81), ...}
"""

import glob
import os

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from features import COMP_TARGETS

BUY_CLASSES = 9          # 0 = ничего, 1..8 = слоты (маскируем лишние на инференсе)
COMP_CLASSES = ["держать текущую"] + COMP_TARGETS
MAX_LEN = 512

LAYA_REPO = "convaiinnovations/laya"
LAYA_SUBFOLDER = "multilingual"


def _snapshot_dir() -> str:
    """Локальный снапшот чекпоинта Laya multilingual (HF-кэш)."""
    pats = [
        os.path.expanduser(
            rf"~/.cache/huggingface/hub/models--convaiinnovations--laya"
            rf"/snapshots/*/{LAYA_SUBFOLDER}"),
        os.path.join(os.environ.get("HF_HOME", ""), "hub",
                     "models--convaiinnovations--laya", "snapshots",
                     "*", LAYA_SUBFOLDER),
    ]
    for pat in pats:
        hits = [d for d in glob.glob(pat)
                if os.path.exists(os.path.join(d, "model.safetensors"))]
        if hits:
            return max(hits, key=os.path.getmtime)
    raise FileNotFoundError(
        "Чекпоинт laya multilingual не найден в кэше — сначала "
        "прогоните predict через laya.Router (скачает в HF-кэш).")


def _load_encoder():
    """ModernBERT-энкодер из чекпоинта Laya (веса под префиксом encoder.)."""
    d = _snapshot_dir()
    cfg = AutoConfig.from_pretrained(os.path.join(d, "encoder"))
    base = AutoModel.from_config(cfg)
    sd = load_file(os.path.join(d, "model.safetensors"))
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items()
              if k.startswith("encoder.")}
    base.load_state_dict(enc_sd)
    return base, AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))


def build_model(encoder_name: str = LAYA_REPO):
    """Энкодер ModernBERT + 5 голов. Возвращает (model, tokenizer, device)."""
    if encoder_name == LAYA_REPO:
        base, tok = _load_encoder()
    else:  # произвольный HF-энкодер
        tok = AutoTokenizer.from_pretrained(encoder_name)
        base = AutoModel.from_pretrained(encoder_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class BGDecisionModel(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            d = base.config.hidden_size
            self.drop = torch.nn.Dropout(0.1)
            self.buy = torch.nn.Linear(d, BUY_CLASSES)
            self.upgrade = torch.nn.Linear(d, 2)
            self.reroll = torch.nn.Linear(d, 2)
            self.freeze = torch.nn.Linear(d, 2)
            self.comp = torch.nn.Linear(d, len(COMP_CLASSES))

        def forward(self, input_ids, attention_mask):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            # mean pooling по ненулевым токенам
            mask = attention_mask.unsqueeze(-1).float()
            h = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            h = self.drop(h)
            return {"buy": self.buy(h), "upgrade": self.upgrade(h),
                    "reroll": self.reroll(h), "freeze": self.freeze(h),
                    "comp": self.comp(h)}

    return BGDecisionModel(base).to(device), tok, device


class BGClassifier:
    def __init__(self, checkpoint_dir: str, encoder_name: str = LAYA_REPO):
        self.model, self.tok, self.device = build_model(encoder_name)
        ckpt = os.path.join(checkpoint_dir, "model.pt")
        state = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=False)   # в чекпоинте — только головы
        self.model.eval()

    @torch.no_grad()
    def predict(self, state_text: str, shop_size: int = 7,
                min_conf: float = 0.0) -> dict:
        """Возвращает {head: (label, confidence)}; слоты за пределами магазина
        маскируются, уверенность ниже min_conf помечается None."""
        enc = self.tok(state_text, truncation=True, max_length=MAX_LEN,
                       return_tensors="pt").to(self.device)
        logits = self.model(**enc)
        out = {}
        for head, logit in logits.items():
            probs = torch.softmax(logit, dim=-1)[0]
            if head == "buy":
                # слоты > размера магазина недоступны
                for s in range(shop_size + 1, BUY_CLASSES):
                    probs[s] = 0.0
                probs = probs / probs.sum().clamp(min=1e-9)
            conf, idx = probs.max(0)
            label = (int(idx), float(conf))
            if float(conf) < min_conf:
                label = (None, float(conf))
            out[head] = label
        return out

    def predict_text(self, state_text: str, shop_size: int = 7) -> str:
        """Человекочитаемый совет из предсказаний (для оверлея, быстрый путь)."""
        p = self.predict(state_text, shop_size=shop_size)
        lines = []
        buy, bc = p["buy"]
        if buy:
            lines.append(f"Купить: слот №{buy}")
        else:
            lines.append("Покупать нечего")
        if p["upgrade"][0] == 1:
            lines.append("Апнуть таверну")
        if p["reroll"][0] == 1:
            lines.append("Рероллить остаток")
        if p["freeze"][0] == 1:
            lines.append("Заморозить магазин")
        comp, cc = p["comp"]
        if comp:
            lines.append(f"Сборка: {COMP_CLASSES[comp]}")
        return "\n".join(lines)
