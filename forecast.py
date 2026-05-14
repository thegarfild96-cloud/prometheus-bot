import os, logging
import numpy as np
import joblib
import yaml

logger = logging.getLogger(__name__)

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

class ForecastEngine:
    def __init__(self, cfg_path="config.yaml"):
        self.cfg = load_config(cfg_path)
        self.models = {}

    def save(self, symbol, horizon):
        os.makedirs(self.cfg["paths"]["models"], exist_ok=True)
        path = os.path.join(self.cfg["paths"]["models"], f"{symbol}_{horizon}h_lgbm.pkl")
        joblib.dump(self.models[horizon], path)

    @classmethod
    def load(cls, symbol, cfg_path="config.yaml"):
        cfg = load_config(cfg_path)
        engine = cls(cfg_path)
        horizons = cfg["trading"]["horizons"]
        for h in horizons:
            path = os.path.join(cfg["paths"]["models"], f"{symbol}_{h}h_lgbm.pkl")
            if os.path.exists(path):
                engine.models[h] = joblib.load(path)
                logger.info(f"Модель загружена: {path}")
            else:
                fallback = os.path.join(cfg["paths"]["models"], f"{symbol}_lgbm.pkl")
                if os.path.exists(fallback):
                    engine.models[h] = joblib.load(fallback)
                    logger.warning(f"Fallback модель для {symbol} +{h}h")
        return engine

    def predict_with_ci(self, X, horizon=1, n_runs=50):
        model = self.models.get(horizon) or list(self.models.values())[0]
        X2 = X.reshape(1, -1)
        mean_pred = float(model.predict(X2)[0])
        return mean_pred, mean_pred, mean_pred

    def predict_price(self, X, scaler, close_idx, horizon=1):
        mean_n, _, _ = self.predict_with_ci(X, horizon)
        ci_pct = {1: 0.01, 4: 0.02, 12: 0.035, 24: 0.05}
        pct = ci_pct.get(horizon, 0.02)
        ci_low  = mean_n * (1 - pct)
        ci_high = mean_n * (1 + pct)
        return mean_n, ci_low, ci_high

def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return {"rmse": rmse, "mape": mape, "r2": r2, "da": 0.0}
