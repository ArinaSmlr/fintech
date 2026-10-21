"""Прогноз для новых данных.

Пример:
    from src.predict import predict
    prices = load_prices(end="2026-10-21")      # или свой DataFrame
    signals = predict(prices)                    # вероятность роста на 5 дней вперёд
    print(signals[signals["Date"] == signals["Date"].max()])
"""
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURES, build_features

ROOT = Path(__file__).resolve().parents[1]


def predict(df, model_path=ROOT / "models" / "model.pkl", threshold=None):
    """df — история цен: столбцы Date, Ticker, Close, Volume (минимум ~260 торговых дней
    на акцию, чтобы рассчитались годовые признаки).

    Возвращает Date, Ticker, p (вероятность роста цены через 5 торговых дней)
    и, если задан threshold, сигнал signal = 1 («покупать»).
    """
    model = joblib.load(model_path)
    data = build_features(df).dropna(subset=FEATURES)
    out = data[["Date", "Ticker"]].copy()
    out["p"] = model.predict_proba(data[FEATURES])[:, 1]
    if threshold is not None:
        out["signal"] = (out["p"] > threshold).astype(int)
    return out.reset_index(drop=True)
