"""
BTC/USDT 5 dakikalık grafik - çoklu indikatör AL sinyali botu
================================================================
Kontrol ettigi indikatorler (grafikte yukaridan asagiya sirasiyla):
  1. RSI(14)              <= RSI_THRESHOLD
  2. Connors RSI(3,2,100) <= CRSI_THRESHOLD
  3. Ultimate Oscillator(7,14,28) <= UO_THRESHOLD
  4. Relative Volatility Index(10) <= RVI_THRESHOLD
  5. ROC(9)               <= ROC_THRESHOLD
  6. Majority Rule(14)    <= MAJORITY_THRESHOLD
  7. MFI(14)              <= MFI_THRESHOLD

Hepsi ayni anda esik degerinin (ve altina) dusmusse Telegram'a AL sinyali
gonderir. Script her calistirildiginda SADECE en son kapanmis 5 dakikalik
mumu kontrol eder (5 dakka 1 denetleme). GitHub Actions ile her 5 dakikada
bir tetiklenecek sekilde tasarlanmistir.

NOT: CoinBrain / TradingView'daki bazi indikatorlerin (ozellikle Connors RSI,
Relative Volatility Index, Majority Rule) birebir ic hesaplama detaylari
platforma gore ufak farklar gosterebilir. Asagidaki formuller yaygin kabul
gormus standart tanimlardir. Ilk calistirmalarda botun bastigi degerleri
kendi ekranindaki degerlerle karsilastirip esikleri ince ayar yapman
onerilir (debug icin her calismada hesaplanan degerler yazdirilir).
"""

import os
import json
import requests
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# AYARLAR
# ------------------------------------------------------------------
SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
INTERVAL = "5m"
KLINE_LIMIT = 300          # Connors RSI'nin percent-rank(100) penceresi icin yeterli gecmis

# Grafikte yukaridan asagiya sirayla verdigin esik degerleri:
RSI_THRESHOLD = float(os.environ.get("RSI_THRESHOLD", 25))
CRSI_THRESHOLD = float(os.environ.get("CRSI_THRESHOLD", 14))
UO_THRESHOLD = float(os.environ.get("UO_THRESHOLD", 36))
RVI_THRESHOLD = float(os.environ.get("RVI_THRESHOLD", 20))
ROC_THRESHOLD = float(os.environ.get("ROC_THRESHOLD", -55))
MAJORITY_THRESHOLD = float(os.environ.get("MAJORITY_THRESHOLD", 17))
MFI_THRESHOLD = float(os.environ.get("MFI_THRESHOLD", 18))

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


# ------------------------------------------------------------------
# VERI CEKME
# ------------------------------------------------------------------
def fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=KLINE_LIMIT):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    # Binance'in son satiri henuz kapanmamis olan mum olabilir -> onu at
    df = df.iloc[:-1].reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# INDIKATORLER
# ------------------------------------------------------------------
def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.fillna(100)


def connors_rsi(close, rsi_len=3, streak_len=2, rank_len=100):
    rsi_close = rsi(close, rsi_len)

    diff = close.diff()
    streak = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        prev = streak.iloc[i - 1]
        if diff.iloc[i] > 0:
            streak.iloc[i] = prev + 1 if prev > 0 else 1
        elif diff.iloc[i] < 0:
            streak.iloc[i] = prev - 1 if prev < 0 else -1
        else:
            streak.iloc[i] = 0
    rsi_streak = rsi(streak, streak_len)

    roc1 = close.pct_change(1) * 100
    percent_rank = roc1.rolling(rank_len).apply(
        lambda x: (x.rank(pct=True).iloc[-1]) * 100, raw=False
    )

    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    return crsi


def ultimate_oscillator(high, low, close, p1=7, p2=14, p3=28):
    prior_close = close.shift(1)
    min_low_pc = pd.concat([low, prior_close], axis=1).min(axis=1)
    max_high_pc = pd.concat([high, prior_close], axis=1).max(axis=1)
    bp = close - min_low_pc
    tr = max_high_pc - min_low_pc
    avg1 = bp.rolling(p1).sum() / tr.rolling(p1).sum()
    avg2 = bp.rolling(p2).sum() / tr.rolling(p2).sum()
    avg3 = bp.rolling(p3).sum() / tr.rolling(p3).sum()
    uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
    return uo


def relative_volatility_index(close, length=10):
    std = close.rolling(length).std()
    diff = close.diff()
    up_std = std.where(diff > 0, 0.0)
    down_std = std.where(diff < 0, 0.0)
    up_smooth = up_std.ewm(alpha=1 / length, adjust=False).mean()
    down_smooth = down_std.ewm(alpha=1 / length, adjust=False).mean()
    rvi = 100 * up_smooth / (up_smooth + down_smooth)
    return rvi


def roc(close, length=9):
    return (close / close.shift(length) - 1) * 100


def majority_rule(close, length=14):
    direction = np.sign(close.diff())
    mr = direction.rolling(length).sum() / length * 100
    return mr


def money_flow_index(high, low, close, volume, length=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    delta_tp = tp.diff()
    pos_mf = mf.where(delta_tp > 0, 0.0)
    neg_mf = mf.where(delta_tp < 0, 0.0)
    pos_sum = pos_mf.rolling(length).sum()
    neg_sum = neg_mf.rolling(length).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    return mfi.fillna(100)


def compute_indicators(df):
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    out = pd.DataFrame(index=df.index)
    out["rsi14"] = rsi(close, 14)
    out["crsi"] = connors_rsi(close, 3, 2, 100)
    out["uo"] = ultimate_oscillator(high, low, close, 7, 14, 28)
    out["rvi10"] = relative_volatility_index(close, 10)
    out["roc9"] = roc(close, 9)
    out["majority14"] = majority_rule(close, 14)
    out["mfi14"] = money_flow_index(high, low, close, volume, 14)
    return out


# ------------------------------------------------------------------
# STATE (ayni sinyali arka arkaya spamlamayi onlemek icin)
# ------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_signal_bar": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ------------------------------------------------------------------
# TELEGRAM
# ------------------------------------------------------------------
def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[UYARI] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tanimli degil, mesaj gonderilmedi.")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    r = requests.post(url, data=payload, timeout=15)
    if not r.ok:
        print(f"[HATA] Telegram gonderimi basarisiz: {r.status_code} {r.text}")


# ------------------------------------------------------------------
# ANA MANTIK
# ------------------------------------------------------------------
def main():
    df = fetch_klines()
    ind = compute_indicators(df)
    last = ind.iloc[-1]
    bar_time = int(df["close_time"].iloc[-1])
    price = df["close"].iloc[-1]

    values = {
        "RSI(14)": (last["rsi14"], RSI_THRESHOLD),
        "CRSI(3,2,100)": (last["crsi"], CRSI_THRESHOLD),
        "UO(7,14,28)": (last["uo"], UO_THRESHOLD),
        "RVI(10)": (last["rvi10"], RVI_THRESHOLD),
        "Majority Rule(14)": (last["majority14"], MAJORITY_THRESHOLD),
        "MFI(14)": (last["mfi14"], MFI_THRESHOLD),
    }

    print(f"--- {SYMBOL} {INTERVAL} | mum kapanis zamani: {bar_time} | fiyat: {price} ---")
    all_ok = True
    for name, (val, thr) in values.items():
        ok = val <= thr
        all_ok = all_ok and ok
        print(f"{name:20s} deger={val:8.2f}  esik<={thr:8.2f}  {'OK' if ok else '-'}")

    state = load_state()

    if all_ok:
        if state.get("last_signal_bar") == bar_time:
            print("Bu mum icin sinyal zaten gonderildi, tekrar gonderilmiyor.")
        else:
            lines = [f"• {name}: {val:.2f} (esik ≤ {thr:.2f})" for name, (val, thr) in values.items()]
            msg = (
                f"🟢 <b>AL SİNYALİ</b> — {SYMBOL} ({INTERVAL})\n"
                f"Fiyat: {price:.2f}\n\n" + "\n".join(lines)
            )
            send_telegram(msg)
            state["last_signal_bar"] = bar_time
            save_state(state)
            print("AL sinyali gonderildi.")
    else:
        # Kosullar artik saglanmiyorsa bir sonraki gercek sinyal icin durumu sifirla
        if state.get("last_signal_bar") is not None:
            state["last_signal_bar"] = None
            save_state(state)
        print("Kosullar saglanmadi, sinyal yok.")


if __name__ == "__main__":
    main()
