"""
preprocessor.py — нормализация, технические индикаторы, IQR-фильтрация, формирование окон.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import ta
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет технические индикаторы через библиотеку ta."""
    df = df.copy()

    # Скользящие средние
    df["sma_10"]  = ta.trend.sma_indicator(df["close"], window=10)
    df["sma_20"]  = ta.trend.sma_indicator(df["close"], window=20)
    df["sma_60"]  = ta.trend.sma_indicator(df["close"], window=60)
    df["ema_14"]  = ta.trend.ema_indicator(df["close"], window=14)

    # Волатильность
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["atr_14"]  = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # Осцилляторы
    df["rsi_14"]  = ta.momentum.rsi(df["close"], window=14)
    macd = ta.trend.MACD(df["close"])
    df["macd"]      = macd.macd()
    df["macd_sig"]  = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Объём
    df["obv"]     = ta.volume.on_balance_volume(df["close"], df["volume"])
    df["adi"]     = ta.volume.acc_dist_index(df["high"], df["low"], df["close"], df["volume"])

    # Микроструктура
    df["hl_spread"]    = (df["high"] - df["low"]) / df["close"]
    df["vol_change"]   = df["volume"].pct_change()
    df["price_change"] = df["close"].pct_change()

    return df


def remove_anomalies(df: pd.DataFrame, iqr_coef: float = 3.0) -> pd.DataFrame:
    """IQR-фильтрация аномального объёма + флаг манипуляции."""
    df = df.copy()
    q1 = df["volume"].quantile(0.25)
    q3 = df["volume"].quantile(0.75)
    iqr = q3 - q1
    lo = q1 - iqr_coef * iqr
    hi = q3 + iqr_coef * iqr
    mask = (df["volume"] < lo) | (df["volume"] > hi)
    df["manipulation_flag"] = mask.astype(int)
    logger.info(f"Аномалий найдено: {mask.sum()} из {len(df)} ({mask.mean()*100:.1f}%)")
    return df


class DataPipeline:
    """Полный пайплайн предобработки: индикаторы → фильтрация → нормализация → окна."""

    def __init__(self, cfg_path: str = "config.yaml"):
        self.cfg = load_config(cfg_path)
        self.window     = self.cfg["model"]["window"]
        self.n_features = self.cfg["model"]["n_features"]
        self.iqr_coef   = self.cfg["anomaly"]["iqr_coef"]
        self.scaler     = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols: list = []

    # ---- список всех признаков -------------------------------------------------
    FEATURE_CANDIDATES = [
        "open", "high", "low", "close", "volume",
        "sma_10", "sma_20", "sma_60", "ema_14",
        "bb_upper", "bb_lower", "bb_width", "atr_14",
        "rsi_14", "macd", "macd_sig", "macd_diff",
        "obv", "adi",
        "hl_spread", "vol_change", "price_change",
        "manipulation_flag",
    ]

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Обучает скейлер и возвращает (X, y) для обучения модели.
        X shape: (n_samples, window, n_features)
        y shape: (n_samples,)
        """
        df = add_indicators(df)
        df = remove_anomalies(df, self.iqr_coef)
        df = df.dropna().reset_index(drop=True)

        # Используем все кандидаты (SHAP-ранжирование оставляем как TODO для
        # расширения — в МВП берём топ-20 по дисперсии)
        available = [c for c in self.FEATURE_CANDIDATES if c in df.columns]
        # Топ-N по дисперсии как простая замена SHAP для МВП
        variances = df[available].var().sort_values(ascending=False)
        self.feature_cols = variances.index[:self.n_features].tolist()

        feat = df[self.feature_cols].values
        target = df["close"].values

        feat_scaled = self.scaler.fit_transform(feat)

        X, y = self._build_windows(feat_scaled, target)
        logger.info(f"Dataset: X={X.shape}, y={y.shape}, features={self.feature_cols}")
        return X, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Применяет обученный скейлер к новым данным (инференс)."""
        df = add_indicators(df)
        df = remove_anomalies(df, self.iqr_coef)
        df = df.dropna().reset_index(drop=True)

        feat = df[self.feature_cols].values
        feat_scaled = self.scaler.transform(feat)

        # Берём последнее окно для прогноза
        if len(feat_scaled) < self.window:
            raise ValueError(f"Недостаточно данных: нужно {self.window}, есть {len(feat_scaled)}")
        return feat_scaled[-self.window:][np.newaxis, :, :]  # (1, window, n_features)

    def _build_windows(self, feat: np.ndarray, target: np.ndarray):
        X, y = [], []
        for i in range(len(feat) - self.window):
            X.append(feat[i: i + self.window])
            y.append(target[i + self.window])
        return np.array(X), np.array(y)

    def save(self, symbol: str, cfg_path: str = "config.yaml"):
        cfg = load_config(cfg_path)
        os.makedirs(cfg["paths"]["scalers"], exist_ok=True)
        path = os.path.join(cfg["paths"]["scalers"], f"{symbol}_pipeline.pkl")
        joblib.dump({"scaler": self.scaler, "features": self.feature_cols,
                     "window": self.window}, path)
        logger.info(f"Pipeline сохранён: {path}")

    @classmethod
    def load(cls, symbol: str, cfg_path: str = "config.yaml") -> "DataPipeline":
        cfg = load_config(cfg_path)
        path = os.path.join(cfg["paths"]["scalers"], f"{symbol}_pipeline.pkl")
        data = joblib.load(path)
        pipe = cls(cfg_path)
        pipe.scaler       = data["scaler"]
        pipe.feature_cols = data["features"]
        pipe.window       = data["window"]
        logger.info(f"Pipeline загружен: {path} | features={pipe.feature_cols[:5]}...")
        return pipe
