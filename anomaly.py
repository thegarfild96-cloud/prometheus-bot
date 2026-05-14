"""
anomaly.py — детекция манипулятивной активности через Z-Score объёма.
"""

import logging
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class AnomalyDetector:
    """Определяет манипулятивный фон по Z-Score и IQR текущего объёма."""

    def __init__(self, cfg_path: str = "config.yaml"):
        self.cfg = load_config(cfg_path)
        self.z_threshold = self.cfg["anomaly"]["zscore_threshold"]
        self.iqr_coef    = self.cfg["anomaly"]["iqr_coef"]

    def check(self, df: pd.DataFrame) -> dict:
        """
        Принимает DataFrame с колонкой 'volume'.
        Возвращает словарь:
          - flag: bool — есть ли аномалия
          - z_score: float — Z-Score последней свечи
          - level: str — 'normal' / 'warning' / 'critical'
          - message: str — текст для пользователя
        """
        volumes = df["volume"].values
        if len(volumes) < 24:
            return {"flag": False, "z_score": 0.0,
                    "level": "normal", "message": "Недостаточно данных"}

        # Z-Score последнего значения относительно последних 168 свечей (7 дней)
        window = volumes[-168:]
        mu  = window.mean()
        std = window.std() + 1e-8
        z   = float((volumes[-1] - mu) / std)

        # IQR-флаг
        q1, q3 = np.percentile(window, [25, 75])
        iqr = q3 - q1
        iqr_flag = (volumes[-1] < q1 - self.iqr_coef * iqr or
                    volumes[-1] > q3 + self.iqr_coef * iqr)

        flag = abs(z) > self.z_threshold or iqr_flag

        if abs(z) > self.z_threshold * 1.5 or (iqr_flag and abs(z) > self.z_threshold):
            level = "critical"
            msg = (f"🔴 Критический уровень аномалии!\n"
                   f"Z-Score объёма: {z:+.2f}σ\n"
                   f"Риск манипуляции высок. Рекомендуется снизить позицию.")
        elif flag:
            level = "warning"
            msg = (f"🟡 Повышенная активность!\n"
                   f"Z-Score объёма: {z:+.2f}σ\n"
                   f"Возможен манипулятивный паттерн. Будьте осторожны.")
        else:
            level = "normal"
            msg = f"🟢 Рыночный фон в норме.\nZ-Score объёма: {z:+.2f}σ"

        logger.info(f"Anomaly check: z={z:.2f}, flag={flag}, level={level}")
        return {"flag": flag, "z_score": z, "level": level, "message": msg}

    def get_manipulation_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает список временных меток с аномальным объёмом
        для нанесения на график.
        """
        volumes = df["volume"].values
        mu  = np.mean(volumes)
        std = np.std(volumes) + 1e-8
        z_scores = (volumes - mu) / std

        df = df.copy()
        df["z_score"] = z_scores
        df["is_anomaly"] = np.abs(z_scores) > self.z_threshold
        return df[df["is_anomaly"]][["datetime", "close", "volume", "z_score"]]
