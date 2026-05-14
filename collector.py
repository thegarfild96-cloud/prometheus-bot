"""
collector.py — модуль сбора исторических и текущих данных с Binance API.
Сохраняет свечи OHLCV в SQLite.
"""

import sqlite3
import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_db(cfg: dict) -> sqlite3.Connection:
    conn = sqlite3.connect(cfg["paths"]["db"])
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_candles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            open_time   INTEGER NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      REAL,
            close_time  INTEGER,
            UNIQUE(symbol, open_time)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT,
            horizon_h     INTEGER,
            forecast_time INTEGER,
            target_time   INTEGER,
            predicted     REAL,
            actual        REAL,
            ci_low        REAL,
            ci_high       REAL,
            rmse          REAL,
            mape          REAL,
            anomaly_flag  INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Загружает свечи с Binance REST API порциями по 1000."""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    current = start_ms

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            break

        if not data:
            break

        all_klines.extend(data)
        current = data[-1][6] + 1  # close_time последней свечи + 1 мс
        time.sleep(0.2)            # уважаем rate limit

    return all_klines


def klines_to_df(klines: list) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(klines, columns=cols)
    numeric = ["open", "high", "low", "close", "volume"]
    df[numeric] = df[numeric].astype(float)
    df["open_time"] = df["open_time"].astype(int)
    df["close_time"] = df["close_time"].astype(int)
    return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]


def save_candles(conn: sqlite3.Connection, symbol: str, df: pd.DataFrame) -> int:
    saved = 0
    for _, row in df.iterrows():
        try:
            conn.execute(
                """INSERT OR IGNORE INTO raw_candles
                   (symbol, open_time, open, high, low, close, volume, close_time)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, int(row.open_time), row.open, row.high,
                 row.low, row.close, row.volume, int(row.close_time))
            )
            saved += 1
        except Exception as e:
            logger.warning(f"Insert error: {e}")
    conn.commit()
    return saved


def load_history(symbol: str, days: int = 730, cfg_path: str = "config.yaml") -> int:
    """Загружает историю за последние N дней и сохраняет в БД."""
    cfg = load_config(cfg_path)
    conn = get_db(cfg)

    end_ms = int(datetime.utcnow().timestamp() * 1000)
    start_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    interval = cfg["binance"]["interval"]

    logger.info(f"Загружаю {days} дней для {symbol} [{interval}]...")
    klines = fetch_klines(symbol, interval, start_ms, end_ms)

    if not klines:
        logger.error("Нет данных от Binance")
        conn.close()
        return 0

    df = klines_to_df(klines)
    saved = save_candles(conn, symbol, df)
    logger.info(f"Сохранено {saved} новых свечей для {symbol}")
    conn.close()
    return saved


def load_candles_from_db(symbol: str, limit: int = 5000,
                         cfg_path: str = "config.yaml") -> pd.DataFrame:
    """Читает последние N свечей из БД."""
    cfg = load_config(cfg_path)
    conn = get_db(cfg)
    df = pd.read_sql_query(
        "SELECT * FROM raw_candles WHERE symbol=? ORDER BY open_time DESC LIMIT ?",
        conn, params=(symbol, limit)
    )
    conn.close()
    df = df.sort_values("open_time").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def update_latest(symbol: str, cfg_path: str = "config.yaml") -> int:
    """Подгружает только недостающие свечи (дельта от последней в БД)."""
    cfg = load_config(cfg_path)
    conn = get_db(cfg)

    row = conn.execute(
        "SELECT MAX(open_time) FROM raw_candles WHERE symbol=?", (symbol,)
    ).fetchone()
    last_ts = row[0] if row[0] else 0

    # Если данных нет вообще — грузим полную историю
    if last_ts == 0:
        conn.close()
        return load_history(symbol, cfg["binance"]["history_days"], cfg_path)

    start_ms = last_ts + 1
    end_ms = int(datetime.utcnow().timestamp() * 1000)
    interval = cfg["binance"]["interval"]

    klines = fetch_klines(symbol, interval, start_ms, end_ms)
    if not klines:
        conn.close()
        return 0

    df = klines_to_df(klines)
    saved = save_candles(conn, symbol, df)
    logger.info(f"Обновлено {saved} свечей для {symbol}")
    conn.close()
    return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for pair in ["BTCUSDT", "ETHUSDT"]:
        load_history(pair, days=730)
