"""
bot.py — телеграм-бот системы Prometheus.
Запуск: python bot.py

Команды:
  /start              — приветствие
  /help               — справка
  /pairs              — доступные пары
  /forecast BTC 1h    — прогноз цены
  /anomaly BTC        — проверка манипулятивного фона
  /chart BTC          — исторический график 30 дней
  /history BTC        — точность последних 10 прогнозов
"""

import asyncio
import logging
import os
import sqlite3
from datetime import datetime

import yaml
from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message, FSInputFile,
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
)
from aiogram.utils.markdown import bold, code

from live import start_live_streams, get_live_price
import asyncio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bot")

# ---- Загрузка конфига ----------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

CFG = load_config()
TOKEN = CFG["telegram"]["token"]
PAIRS = CFG["trading"]["pairs"]
HORIZONS = CFG["trading"]["horizons"]

bot = Bot(token=TOKEN)
dp  = Dispatcher()
router = Router()
dp.include_router(router)

# ---- Хелперы -------------------------------------------------------------------

def pair_from_arg(arg: str) -> str | None:
    """BTC → BTCUSDT, BTCUSDT → BTCUSDT"""
    arg = arg.upper().strip()
    if not arg.endswith("USDT"):
        arg += "USDT"
    return arg if arg in PAIRS else None


def load_engines() -> dict:
    """Ленивая загрузка моделей при первом обращении."""
    from forecast import ForecastEngine
    from preprocessor import DataPipeline
    engines, pipelines = {}, {}
    for pair in PAIRS:
        # Проверяем любой из форматов файла модели
        model_path   = os.path.join(CFG["paths"]["models"], f"{pair}_lgbm.pkl")
        model_path_h = os.path.join(CFG["paths"]["models"], f"{pair}_1h_lgbm.pkl")
        scaler_path  = os.path.join(CFG["paths"]["scalers"], f"{pair}_pipeline.pkl")
        has_model = os.path.exists(model_path) or os.path.exists(model_path_h)
        if has_model and os.path.exists(scaler_path):
            try:
                engines[pair]  = ForecastEngine.load(pair)
                pipelines[pair] = DataPipeline.load(pair)
                logger.info(f"Модель загружена: {pair}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить {pair}: {e}")
    return engines, pipelines


# Кэш моделей (загружаем один раз при старте)
ENGINES: dict  = {}
PIPELINES: dict = {}

# ---- Клавиатуры ----------------------------------------------------------------

def main_keyboard() -> InlineKeyboardMarkup:
    buttons = []
    for pair in PAIRS:
        symbol = pair.replace("USDT", "")
        buttons.append([
            InlineKeyboardButton(text=f"📈 {symbol} 1ч",  callback_data=f"fc_{pair}_1"),
            InlineKeyboardButton(text=f"📈 {symbol} 4ч",  callback_data=f"fc_{pair}_4"),
            InlineKeyboardButton(text=f"📈 {symbol} 24ч", callback_data=f"fc_{pair}_24"),
        ])
    buttons.append([
        InlineKeyboardButton(text="⚠️ Аномалии BTC", callback_data="an_BTCUSDT"),
        InlineKeyboardButton(text="⚠️ Аномалии ETH", callback_data="an_ETHUSDT"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# ---- Хэндлеры команд -----------------------------------------------------------

@router.message(CommandStart())
async def cmd_start(message: Message):
    text = (
        "🔮 <b>Prometheus</b> — система прогнозирования криптовалют\n\n"
        "Используется гибридная нейросетевая модель sLSTM с фильтрацией "
        "манипулятивного фона.\n\n"
        "<b>Доступные команды:</b>\n"
        "/forecast BTC 1h — прогноз на 1/4/12/24 часа\n"
        "/anomaly BTC — проверка рыночного фона\n"
        "/chart BTC — график за 30 дней\n"
        "/history BTC — точность прошлых прогнозов\n"
        "/pairs — список активов\n/live — живые цены всех пар\n/fg — Fear & Greed индекс\n/stats — статистика точности\n/subscribe — подписаться на уведомления\n"
        "/help — справка\n\n"
        "Или нажмите кнопку ниже 👇"
    )
    await message.answer(text, parse_mode="HTML", reply_markup=main_keyboard())


@router.message(Command("help"))
async def cmd_help(message: Message):
    text = (
        "📖 <b>Справка Prometheus</b>\n\n"
        "<b>/forecast</b> [ПАРА] [ГОРИЗОНТ]\n"
        "  Пример: <code>/forecast BTC 4h</code>\n"
        "  Горизонты: 1h, 4h, 12h, 24h\n\n"
        "<b>/anomaly</b> [ПАРА]\n"
        "  Пример: <code>/anomaly ETH</code>\n"
        "  Показывает Z-Score объёма и уровень риска\n\n"
        "<b>/chart</b> [ПАРА]\n"
        "  Пример: <code>/chart BTC</code>\n"
        "  График цены за последние 30 дней\n\n"
        "<b>/history</b> [ПАРА]\n"
        "  Статистика точности последних 10 прогнозов\n\n"
        "<b>/pairs</b> — список доступных активов\n\n"
        "⚠️ Прогнозы носят информационный характер и не являются "
        "инвестиционными рекомендациями."
    )
    await message.answer(text, parse_mode="HTML")


@router.message(Command("pairs"))
async def cmd_pairs(message: Message):
    lines = ["📋 <b>Доступные торговые пары:</b>\n"]
    for p in PAIRS:
        symbol = p.replace("USDT", "")
        has_model = p in ENGINES
        status = "✅ модель готова" if has_model else "⏳ требуется обучение"
        lines.append(f"  • <b>{symbol}/USDT</b> — {status}")
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("forecast"))
async def cmd_forecast(message: Message):
    parts = message.text.split()
    # /forecast BTC 4h  → parts = ['/forecast', 'BTC', '4h']
    if len(parts) < 3:
        await message.answer(
            "Использование: <code>/forecast BTC 4h</code>\n"
            "Горизонты: 1h, 4h, 12h, 24h", parse_mode="HTML"
        )
        return

    symbol  = pair_from_arg(parts[1])
    horizon_raw = parts[2].lower().replace("h", "").replace("ч", "")
    try:
        horizon = int(horizon_raw)
    except ValueError:
        horizon = 1

    if symbol not in ENGINES:
        await message.answer(
            f"⚠️ Модель для <b>{parts[1].upper()}</b> не найдена.\n"
            f"Запустите <code>python train.py --symbol {parts[1].upper()}USDT</code>",
            parse_mode="HTML"
        )
        return

    await _send_forecast(message, symbol, horizon)



@router.message(Command("live"))
async def cmd_live(message: Message):
    """Показывает живые цены всех пар."""
    lines = ["📡 <b>Live prices — Prometheus</b>\n"]
    for pair in PAIRS:
        live = get_live_price(pair)
        if live:
            symbol = pair.replace("USDT", "")
            arrow = "📈" if live["change"] > 0 else "📉"
            lines.append(
                f"{arrow} <b>{symbol}/USDT</b>\n"
                f"   Цена: <b>${live['price']:,.2f}</b>\n"
                f"   24ч: <b>{live['change']:+.2f}%</b>\n"
                f"   High: ${live['high']:,.2f} | Low: ${live['low']:,.2f}\n"
                f"   Объём: {live['volume']:,.0f}\n"
                f"   Обновлено: {live['time']} UTC\n"
            )
        else:
            lines.append(f"⏳ {pair} — ожидание данных...")
    await message.answer("\n".join(lines), parse_mode="HTML")

@router.message(Command("anomaly"))
async def cmd_anomaly(message: Message):
    parts = message.text.split()
    arg   = parts[1] if len(parts) > 1 else "BTC"
    symbol = pair_from_arg(arg)
    if not symbol:
        await message.answer(f"Пара не найдена: {arg}")
        return
    await _send_anomaly(message, symbol)


@router.message(Command("chart"))
async def cmd_chart(message: Message):
    parts = message.text.split()
    arg   = parts[1] if len(parts) > 1 else "BTC"
    symbol = pair_from_arg(arg)
    if not symbol:
        await message.answer(f"Пара не найдена: {arg}")
        return
    await _send_chart(message, symbol)


@router.message(Command("history"))
async def cmd_history(message: Message):
    parts  = message.text.split()
    arg    = parts[1] if len(parts) > 1 else "BTC"
    symbol = pair_from_arg(arg)
    if not symbol:
        await message.answer(f"Пара не найдена: {arg}")
        return
    await _send_history(message, symbol)


# ---- Callback-обработчики (кнопки) ---------------------------------------------

@router.callback_query(F.data.startswith("fc_"))
async def cb_forecast(call: CallbackQuery):
    # fc_BTCUSDT_4
    _, symbol, h = call.data.split("_")
    await call.answer()
    if symbol not in ENGINES:
        await call.message.answer("⚠️ Модель не найдена. Нужно запустить train.py")
        return
    await _send_forecast(call.message, symbol, int(h))


@router.callback_query(F.data.startswith("an_"))
async def cb_anomaly(call: CallbackQuery):
    _, symbol = call.data.split("_")
    await call.answer()
    await _send_anomaly(call.message, symbol)


# ---- Внутренние функции --------------------------------------------------------

async def _send_forecast(message: Message, symbol: str, horizon: int):
    """Генерирует прогноз и отправляет график."""
    from collector import load_candles_from_db, update_latest
    from anomaly import AnomalyDetector
    from visualizer import build_forecast_chart

    await message.answer("⏳ Генерирую прогноз...")

    try:
        # Обновляем данные
        update_latest(symbol)
        df = load_candles_from_db(symbol, limit=5000)

        # Предобработка → инференс
        pipe   = PIPELINES[symbol]
        engine = ENGINES[symbol]

        X = pipe.transform(df)
        close_idx = pipe.feature_cols.index("close") if "close" in pipe.feature_cols else 3
        predicted, ci_low, ci_high = engine.predict_price(X, pipe.scaler, close_idx, horizon=horizon)

        # Детекция аномалий
        detector  = AnomalyDetector()
        anomaly   = detector.check(df)
        anomaly_periods = detector.get_manipulation_periods(df.tail(168))

        # График
        chart_path = build_forecast_chart(
            symbol, df, predicted, ci_low, ci_high,
            horizon, anomaly_periods
        )

        # Текст ответа
        direction = "📈" if predicted > df["close"].iloc[-1] else "📉"
        warn = f"\n\n{anomaly['message']}" if anomaly["flag"] else ""
        # Берём live цену если доступна
        live = get_live_price(symbol)
        current = live["price"] if live else df["close"].iloc[-1]
        delta_pct = (predicted - current) / current * 100

        live = get_live_price(symbol)
        current = live["price"] if live else df["close"].iloc[-1]
        delta_pct = (predicted - current) / current * 100

        # Ширина доверительного интервала
        ci_width = ci_high - ci_low
        ci_half = ci_width / 2 if ci_width > 0 else 1

        # Вероятность прогноза: насколько изменение превышает неопределённость
        raw_prob = min(100, abs(delta_pct) / (ci_half / current * 100) * 50)
        probability = round(raw_prob)

        if probability >= 70:
            prob_label = "🟢 Высокая"
        elif probability >= 45:
            prob_label = "🟡 Средняя"
        else:
            prob_label = "🔴 Низкая"

        # Надёжность: CI шире изменения → слабый сигнал
        reliable = abs(delta_pct) > (ci_half / current * 100)

        # Порог сигнала — только если вероятность достаточная
        min_threshold = 0.5 if probability >= 45 else 1.0

        if delta_pct > min_threshold and reliable:
            strength = min(abs(delta_pct) * 10, 100)
            signal_block = f"\n━━━━━━━━━━━━━━━━\nСигнал: <b>🟢 LONG</b> ({strength:.0f}%)\nВероятность: <b>{probability}%</b> {prob_label}\nВход: <b>${current:,.2f}</b> | Стоп: <b>${current*0.995:,.2f}</b>\nЦель: <b>${predicted:,.2f}</b> | Потенциал: <b>{abs(delta_pct):.2f}%</b>\n━━━━━━━━━━━━━━━━"
        elif delta_pct < -min_threshold and reliable:
            strength = min(abs(delta_pct) * 10, 100)
            signal_block = f"\n━━━━━━━━━━━━━━━━\nСигнал: <b>🔴 SHORT</b> ({strength:.0f}%)\nВероятность: <b>{probability}%</b> {prob_label}\nВход: <b>${current:,.2f}</b> | Стоп: <b>${current*1.005:,.2f}</b>\nЦель: <b>${predicted:,.2f}</b> | Потенциал: <b>{abs(delta_pct):.2f}%</b>\n━━━━━━━━━━━━━━━━"
        else:
            signal_block = f"\n━━━━━━━━━━━━━━━━\nСигнал: <b>⚪ НЕЙТРАЛЬНО</b>\nВероятность: <b>{probability}%</b> {prob_label}\nCI шире изменения — ждём чёткого движения\n━━━━━━━━━━━━━━━━"

        warn = f"\n\n{anomaly['message']}" if anomaly["flag"] else ""

        caption = (
            f"🔮 <b>Prometheus — {symbol}</b>\n\n"
            f"Текущая цена: <b>${current:,.2f}</b>\n"
            f"Прогноз +{horizon}ч: <b>${predicted:,.2f}</b>\n"
            f"Изменение: <b>{delta_pct:+.2f}%</b>"
            f"{signal_block}\n\n"
            f"Доверительный интервал (95%):\n"
            f"  ${ci_low:,.2f} — ${ci_high:,.2f}"
            f"{warn}\n\n"
            f"<i>⚠️ Не является инвестиционной рекомендацией</i>"
        )

        photo = FSInputFile(chart_path)
        await message.answer_photo(photo=photo, caption=caption, parse_mode="HTML")

        # Сохраняем прогноз в БД
        _log_forecast(symbol, horizon, predicted, ci_low, ci_high, anomaly["flag"])

    except Exception as e:
        logger.exception(f"Forecast error for {symbol}: {e}")
        await message.answer(
            f"❌ Ошибка при генерации прогноза: {e}\n"
            f"Убедитесь, что модель обучена: <code>python train.py</code>",
            parse_mode="HTML"
        )


async def _send_anomaly(message: Message, symbol: str):
    from collector import load_candles_from_db, update_latest
    from anomaly import AnomalyDetector

    try:
        update_latest(symbol)
        df = load_candles_from_db(symbol, limit=500)
        detector = AnomalyDetector()
        result   = detector.check(df)

        text = (
            f"🔍 <b>Анализ аномалий — {symbol}</b>\n\n"
            f"{result['message']}\n\n"
            f"Последний объём: {df['volume'].iloc[-1]:,.0f}\n"
            f"Средний объём (7д): {df['volume'].tail(168).mean():,.0f}\n\n"
            f"<i>Обновлено: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC</i>"
        )
        await message.answer(text, parse_mode="HTML")

    except Exception as e:
        logger.exception(e)
        await message.answer(f"❌ Ошибка: {e}")


async def _send_chart(message: Message, symbol: str):
    from collector import load_candles_from_db, update_latest
    from visualizer import build_history_chart

    try:
        update_latest(symbol)
        df  = load_candles_from_db(symbol, limit=720)
        path = build_history_chart(symbol, df)
        photo = FSInputFile(path)
        caption = (
            f"📊 <b>{symbol}</b> — последние 30 дней\n"
            f"Текущая цена: <b>${df['close'].iloc[-1]:,.2f}</b>\n"
            f"Мин: ${df['close'].tail(720).min():,.2f} | "
            f"Макс: ${df['close'].tail(720).max():,.2f}"
        )
        await message.answer_photo(photo=photo, caption=caption, parse_mode="HTML")

    except Exception as e:
        logger.exception(e)
        await message.answer(f"❌ Ошибка: {e}")


async def _send_history(message: Message, symbol: str):
    conn = sqlite3.connect(CFG["paths"]["db"])
    rows = conn.execute(
        """SELECT horizon_h, predicted, actual, rmse, mape, anomaly_flag,
                  datetime(forecast_time/1000, 'unixepoch') as dt
           FROM forecast_log
           WHERE symbol=? AND actual IS NOT NULL
           ORDER BY forecast_time DESC LIMIT 10""",
        (symbol,)
    ).fetchall()
    conn.close()

    if not rows:
        await message.answer(
            f"По <b>{symbol}</b> история прогнозов пока пуста.\n"
            f"Используйте /forecast чтобы сгенерировать первый прогноз.",
            parse_mode="HTML"
        )
        return

    lines = [f"📋 <b>История прогнозов — {symbol}</b>\n"]
    for r in rows:
        h, pred, actual, rmse, mape, flag, dt = r
        flag_str = " ⚠️" if flag else ""
        if actual:
            acc = abs(pred - actual) / actual * 100
            lines.append(
                f"{dt} | +{h}ч{flag_str}\n"
                f"  Прогноз: ${pred:,.0f} | Факт: ${actual:,.0f} | Δ {acc:.2f}%"
            )
        else:
            lines.append(f"{dt} | +{h}ч{flag_str} | Прогноз: ${pred:,.0f} | ожидаем...")

    avg_mape = sum(r[4] for r in rows if r[4]) / max(len(rows), 1)
    lines.append(f"\nСредний MAPE: <b>{avg_mape:.2f}%</b>")

    await message.answer("\n".join(lines), parse_mode="HTML")


def _log_forecast(symbol: str, horizon: int, predicted: float,
                  ci_low: float, ci_high: float, anomaly: bool):
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    conn = sqlite3.connect(CFG["paths"]["db"])
    conn.execute(
        """INSERT INTO forecast_log
           (symbol, horizon_h, forecast_time, target_time, predicted,
            ci_low, ci_high, anomaly_flag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, horizon, now_ms,
         now_ms + horizon * 3600 * 1000,
         predicted, ci_low, ci_high, int(anomaly))
    )
    conn.commit()
    conn.close()


# ---- Запуск --------------------------------------------------------------------


# ---- Автоуведомления --------------------------------------------------------

ALERT_CHAT_IDS = set()  # список chat_id которые подписались на уведомления

@router.message(Command("subscribe"))
async def cmd_subscribe(message: Message):
    ALERT_CHAT_IDS.add(message.chat.id)
    await message.answer(
        "✅ <b>Подписка активирована!</b>\n\n"
        "Буду присылать уведомления когда:\n"
        "• Сигнал LONG/SHORT сильнее 5%\n"
        "• Обнаружена аномалия на рынке\n"
        "• Резкое движение цены > 2%\n\n"
        "Отписаться: /unsubscribe",
        parse_mode="HTML"
    )

@router.message(Command("unsubscribe"))
async def cmd_unsubscribe(message: Message):
    ALERT_CHAT_IDS.discard(message.chat.id)
    await message.answer("❌ Уведомления отключены.")

async def check_and_alert():
    """Проверяет рынок каждые 15 минут и шлёт уведомления."""
    if not ALERT_CHAT_IDS:
        return
    if not ENGINES:
        return

    from collector import load_candles_from_db, update_latest
    from anomaly import AnomalyDetector

    for symbol in PAIRS:
        if symbol not in ENGINES:
            continue
        try:
            update_latest(symbol)
            df = load_candles_from_db(symbol, limit=5000)
            pipe = PIPELINES[symbol]
            engine = ENGINES[symbol]

            X = pipe.transform(df)
            close_idx = pipe.feature_cols.index("close") if "close" in pipe.feature_cols else 3
            predicted, ci_low, ci_high = engine.predict_price(X, pipe.scaler, close_idx)

            live = get_live_price(symbol)
            current = live["price"] if live else df["close"].iloc[-1]
            delta_pct = (predicted - current) / current * 100

            detector = AnomalyDetector()
            anomaly = detector.check(df)

            # Формируем алерт
            alerts = []

            if delta_pct > 5:
                alerts.append(
                    f"🚀 <b>СИЛЬНЫЙ СИГНАЛ LONG — {symbol}</b>\n"
                    f"Текущая цена: ${current:,.2f}\n"
                    f"Прогноз: ${predicted:,.2f} (+{delta_pct:.2f}%)\n"
                    f"Стоп: ${current*0.995:,.2f} | Цель: ${predicted:,.2f}"
                )
            elif delta_pct < -5:
                alerts.append(
                    f"🔻 <b>СИЛЬНЫЙ СИГНАЛ SHORT — {symbol}</b>\n"
                    f"Текущая цена: ${current:,.2f}\n"
                    f"Прогноз: ${predicted:,.2f} ({delta_pct:.2f}%)\n"
                    f"Стоп: ${current*1.005:,.2f} | Цель: ${predicted:,.2f}"
                )

            if anomaly["flag"] and anomaly["level"] == "critical":
                alerts.append(
                    f"⚠️ <b>АНОМАЛИЯ — {symbol}</b>\n"
                    f"{anomaly['message']}"
                )

            # Проверка резкого движения через WebSocket
            if live and abs(live["change"]) > 2:
                direction = "📈" if live["change"] > 0 else "📉"
                alerts.append(
                    f"{direction} <b>РЕЗКОЕ ДВИЖЕНИЕ — {symbol}</b>\n"
                    f"Изменение за 24ч: {live['change']:+.2f}%\n"
                    f"Текущая цена: ${live['price']:,.2f}\n"
                    f"High: ${live['high']:,.2f} | Low: ${live['low']:,.2f}"
                )

            # Отправляем всем подписчикам
            for chat_id in ALERT_CHAT_IDS.copy():
                for alert in alerts:
                    try:
                        await bot.send_message(chat_id, alert, parse_mode="HTML")
                    except Exception as e:
                        logger.warning(f"Не удалось отправить алерт {chat_id}: {e}")
                        ALERT_CHAT_IDS.discard(chat_id)

        except Exception as e:
            logger.warning(f"Alert error {symbol}: {e}")


async def validate_forecasts():
    """Заполняет фактические цены для истёкших прогнозов."""
    from collector import load_candles_from_db
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    conn = sqlite3.connect(CFG["paths"]["db"])

    # Берём прогнозы у которых истёк target_time но нет actual
    rows = conn.execute("""
        SELECT id, symbol, target_time FROM forecast_log
        WHERE actual IS NULL AND target_time <= ?
        LIMIT 50
    """, (now_ms,)).fetchall()

    for row_id, symbol, target_time in rows:
        try:
            df = load_candles_from_db(symbol, limit=500)
            if df.empty:
                continue
            # Ищем свечу ближайшую к target_time
            df["diff"] = (df["open_time"] - target_time).abs()
            closest = df.loc[df["diff"].idxmin()]
            actual_price = float(closest["close"])

            # Получаем прогноз для расчёта метрик
            pred_row = conn.execute(
                "SELECT predicted FROM forecast_log WHERE id=?", (row_id,)
            ).fetchone()
            if pred_row:
                predicted = pred_row[0]
                rmse = float((predicted - actual_price) ** 2) ** 0.5
                mape = abs(predicted - actual_price) / actual_price * 100

                conn.execute("""
                    UPDATE forecast_log
                    SET actual=?, rmse=?, mape=?
                    WHERE id=?
                """, (actual_price, rmse, mape, row_id))
                logger.info(f"Validated {symbol}: pred={predicted:.2f} actual={actual_price:.2f} mape={mape:.2f}%")
        except Exception as e:
            logger.warning(f"Validation error {symbol}: {e}")

    conn.commit()
    conn.close()

async def validation_scheduler():
    """Проверяет истёкшие прогнозы каждые 30 минут."""
    await asyncio.sleep(30)
    while True:
        await validate_forecasts()
        await asyncio.sleep(1800)  # 30 минут

async def alert_scheduler():
    """Запускает проверку каждые 15 минут."""
    await asyncio.sleep(60)  # ждём 1 минуту после старта
    while True:
        await check_and_alert()
        await asyncio.sleep(900)  # 15 минут


# ---- Fear & Greed + Stats ---------------------------------------------------

async def get_fear_greed() -> dict:
    """Загружает Fear & Greed индекс с alternative.me."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                data = await resp.json()
                val = int(data["data"][0]["value"])
                classification = data["data"][0]["value_classification"]
                if val <= 25:
                    emoji = "🔴"
                elif val <= 45:
                    emoji = "🟠"
                elif val <= 55:
                    emoji = "🟡"
                elif val <= 75:
                    emoji = "🟢"
                else:
                    emoji = "🚀"
                return {"value": val, "label": classification, "emoji": emoji}
    except Exception:
        return {"value": 50, "label": "Neutral", "emoji": "🟡"}


@router.message(Command("fg"))
async def cmd_fear_greed(message: Message):
    """Fear & Greed индекс."""
    fg = await get_fear_greed()
    bar_filled = int(fg["value"] / 5)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)

    if fg["value"] <= 25:
        advice = "Рынок в панике — исторически хорошее время для покупки"
    elif fg["value"] <= 45:
        advice = "Страх на рынке — осторожный вход в лонг"
    elif fg["value"] <= 55:
        advice = "Нейтральный рынок — ждём сигнала"
    elif fg["value"] <= 75:
        advice = "Жадность на рынке — осторожно с лонгами"
    else:
        advice = "Экстремальная жадность — высокий риск коррекции"

    text = (
        f"😱🤑 <b>Fear & Greed Index</b>\n\n"
        f"{fg['emoji']} <b>{fg['value']}/100 — {fg['label']}</b>\n\n"
        f"[{bar}]\n\n"
        f"💡 {advice}\n\n"
        f"<i>Обновляется каждый час · alternative.me</i>"
    )
    await message.answer(text, parse_mode="HTML")


@router.message(Command("stats"))
async def cmd_stats(message: Message):
    """Статистика точности прогнозов."""
    conn = sqlite3.connect(CFG["paths"]["db"])

    # Общая статистика
    rows = conn.execute("""
        SELECT symbol, horizon_h,
               COUNT(*) as total,
               AVG(CASE WHEN actual > 0
                   THEN ABS(predicted - actual) / actual * 100
                   ELSE NULL END) as avg_mape,
               MIN(CASE WHEN actual > 0
                   THEN ABS(predicted - actual) / actual * 100
                   ELSE NULL END) as min_mape,
               MAX(CASE WHEN actual > 0
                   THEN ABS(predicted - actual) / actual * 100
                   ELSE NULL END) as max_mape,
               SUM(CASE WHEN actual > 0 AND
                   ABS(predicted - actual) / actual * 100 < 1
                   THEN 1 ELSE 0 END) as accurate
        FROM forecast_log
        WHERE actual IS NOT NULL AND actual > 0
        GROUP BY symbol, horizon_h
        ORDER BY symbol, horizon_h
    """).fetchall()
    conn.close()

    if not rows:
        await message.answer(
            "📊 <b>Статистика пока пуста</b>\n\n"
            "Прогнозы сохраняются автоматически.\n"
            "Используйте /forecast чтобы накопить статистику.",
            parse_mode="HTML"
        )
        return

    lines = ["📊 <b>Статистика точности Prometheus</b>\n"]
    current_symbol = None

    for row in rows:
        symbol, horizon, total, avg_mape, min_mape, max_mape, accurate = row
        if symbol != current_symbol:
            lines.append(f"\n<b>{symbol}</b>")
            current_symbol = symbol

        accuracy_pct = (accurate / total * 100) if total > 0 else 0
        avg_mape = avg_mape or 0

        if avg_mape < 1:
            quality = "🟢 Отлично"
        elif avg_mape < 2:
            quality = "🟡 Хорошо"
        else:
            quality = "🔴 Слабо"

        lines.append(
            f"  +{horizon}ч: {quality}\n"
            f"  MAPE: {avg_mape:.2f}% | Точных: {accuracy_pct:.0f}%\n"
            f"  Прогнозов: {total}"
        )

    await message.answer("\n".join(lines), parse_mode="HTML")



@router.message(Command("backtest"))
async def cmd_backtest(message: Message):
    """Backtesting стратегии на исторических данных."""
    parts  = message.text.split()
    arg    = parts[1] if len(parts) > 1 else "BTC"
    days   = int(parts[2]) if len(parts) > 2 else 7
    symbol = pair_from_arg(arg)
    if not symbol:
        await message.answer(f"Пара не найдена: {arg}")
        return
    if symbol not in ENGINES:
        await message.answer("Модель не найдена. Запустите train.py")
        return

    await message.answer(f"⏳ Запускаю backtest {symbol} за {days} дней...")

    try:
        import numpy as np
        from collector import load_candles_from_db

        # Загружаем с запасом для окна модели (168 свечей)
        df = load_candles_from_db(symbol, limit=days * 24 + 300)
        if len(df) < 200:
            await message.answer("Недостаточно данных для backtesting")
            return

        pipe   = PIPELINES[symbol]
        engine = ENGINES[symbol]
        close_idx = pipe.feature_cols.index("close") if "close" in pipe.feature_cols else 3
        window = pipe.window  # 168

        # Тестируем на последних days*24 свечах
        # Для каждой свечи берём предыдущие window+1 для трансформации
        test_start = len(df) - days * 24
        if test_start < window:
            test_start = window

        trades   = []
        balance  = 10000.0
        position = None

        for i in range(test_start, len(df) - 1):
            # Берём окно данных до текущей свечи
            window_df = df.iloc[i - window : i + 1].copy()
            if len(window_df) < window:
                continue

            try:
                X = pipe.transform(window_df)
            except Exception:
                continue

            predicted, ci_low, ci_high = engine.predict_price(
                X, pipe.scaler, close_idx, horizon=1
            )
            current   = float(df["close"].iloc[i])
            next_price = float(df["close"].iloc[i + 1])
            delta_pct = (predicted - current) / current * 100
            ci_width  = abs(ci_high - ci_low)
            prob      = min(100, abs(delta_pct) / (ci_width / current * 100 / 2 + 1e-8) * 50)

            # Закрываем позицию
            if position:
                pnl_pct = ((next_price - position["entry"]) / position["entry"] * 100
                           if position["side"] == "long"
                           else (position["entry"] - next_price) / position["entry"] * 100)
                pnl_usd = balance * 0.02 * pnl_pct / 100
                balance += pnl_usd
                trades.append({
                    "side":  position["side"],
                    "entry": position["entry"],
                    "exit":  next_price,
                    "pnl":   pnl_usd,
                    "win":   pnl_usd > 0,
                })
                position = None

            # Открываем позицию при достаточно сильном сигнале
            if not position and prob >= 40:
                if delta_pct > 0.3:
                    position = {"side": "long",  "entry": current}
                elif delta_pct < -0.3:
                    position = {"side": "short", "entry": current}

        if not trades:
            await message.answer(
                f"За {days} дней не было ни одного сигнала.\n"
                f"Попробуйте увеличить период: /backtest BTC 14"
            )
            return

        total    = len(trades)
        wins     = sum(1 for t in trades if t["win"])
        win_rate = wins / total * 100
        total_pnl = sum(t["pnl"] for t in trades)
        total_ret = total_pnl / 10000 * 100
        avg_win  = np.mean([t["pnl"] for t in trades if t["win"]]) if wins > 0 else 0
        avg_loss = np.mean([t["pnl"] for t in trades if not t["win"]]) if wins < total else 0

        # Просадка
        peak, max_dd, bal = 10000.0, 0.0, 10000.0
        for t in trades:
            bal  += t["pnl"]
            peak  = max(peak, bal)
            max_dd = max(max_dd, (peak - bal) / peak * 100)

        # Шарп
        returns = [t["pnl"] / 10000 for t in trades]
        sharpe  = (np.mean(returns) / (np.std(returns) + 1e-8)) * (252 ** 0.5) if len(returns) > 1 else 0

        emoji = "🟢" if total_ret > 0 else "🔴"
        text = (
            f"📊 <b>Backtest — {symbol} ({days} дней)</b>\n\n"
            f"{emoji} <b>Итог: {total_ret:+.2f}%</b> "
            f"(${total_pnl:+.2f} с $10,000)\n\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Сделок: <b>{total}</b>\n"
            f"Прибыльных: <b>{wins}</b> ({win_rate:.0f}%)\n"
            f"Убыточных: <b>{total - wins}</b> ({100-win_rate:.0f}%)\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Ср. прибыль: <b>${avg_win:+.2f}</b>\n"
            f"Ср. убыток: <b>${avg_loss:+.2f}</b>\n"
            f"Макс. просадка: <b>{max_dd:.2f}%</b>\n"
            f"Коэф. Шарпа: <b>{sharpe:.2f}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Размер позиции: 2% депозита\n"
            f"<i>Backtesting на исторических данных. "
            f"Не гарантирует будущих результатов.</i>"
        )
        await message.answer(text, parse_mode="HTML")

    except Exception as e:
        logger.exception(e)
        await message.answer(f"Ошибка: {e}")


async def on_startup():
    global ENGINES, PIPELINES
    logger.info("Prometheus bot starting...")
    ENGINES, PIPELINES = load_engines()
    if not ENGINES:
        logger.warning(
            "Модели не найдены. Запустите: python train.py\n"
            "Бот запущен, но /forecast будет недоступен до обучения."
        )
    else:
        logger.info(f"Загружены модели для: {list(ENGINES.keys())}")
    # Запускаем WebSocket стримы в фоне
    asyncio.create_task(start_live_streams(PAIRS))
    logger.info(f"Live WebSocket запущен для: {PAIRS}")
    asyncio.create_task(alert_scheduler())
    asyncio.create_task(validation_scheduler())
    logger.info("Планировщик валидации прогнозов запущен (каждые 30 мин)")
    logger.info("Планировщик уведомлений запущен (каждые 15 мин)")


async def main():
    await on_startup()
    logger.info("Бот запущен. Ожидание сообщений...")
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
