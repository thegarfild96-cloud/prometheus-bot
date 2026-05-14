"""
live.py — получение live-данных с Binance WebSocket.
Цена обновляется каждую секунду.
"""
import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime
import websockets
import yaml

logger = logging.getLogger(__name__)

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

# Кэш последних цен в памяти
LIVE_PRICES = {}

async def stream_price(symbol: str):
    """Подключается к Binance WebSocket и обновляет LIVE_PRICES."""
    url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info(f"WebSocket подключён: {symbol}")
                async for msg in ws:
                    data = json.loads(msg)
                    LIVE_PRICES[symbol] = {
                        "price":  float(data["c"]),
                        "change": float(data["P"]),
                        "high":   float(data["h"]),
                        "low":    float(data["l"]),
                        "volume": float(data["v"]),
                        "time":   datetime.utcnow().strftime("%H:%M:%S"),
                    }
        except Exception as e:
            logger.warning(f"WebSocket {symbol} ошибка: {e} — переподключение через 3с")
            await asyncio.sleep(3)

async def start_live_streams(pairs: list):
    """Запускает стримы для всех пар параллельно."""
    tasks = [asyncio.create_task(stream_price(p)) for p in pairs]
    await asyncio.gather(*tasks)

def get_live_price(symbol: str) -> dict | None:
    """Возвращает последнюю известную цену из кэша."""
    return LIVE_PRICES.get(symbol)
