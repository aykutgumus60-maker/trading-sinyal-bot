"""
180 GUNLUK GERIYE DONUK TEST (BACKTEST)
========================================
bot.py'deki AYNI indikator formullerini ve AYNI esik degerlerini kullanarak
gecmis 5 dakikalik mumlar uzerinde "hepsi esik altina dustu" kosulunun
kac kere gerceklestigini sayar (bot.py'deki mantikla ayni: kosul once
FALSE iken TRUE'ya donerse 1 sinyal sayilir, TRUE kaldigi surece tekrar
sayilmaz - gercek botun spam yapmama davranisiyla birebir ayni).

Cikti:
  - Konsola ay ay sinyal sayisi ve aylik ortalama
  - signals_180gun.csv dosyasina her sinyalin tarihi/saati ve fiyati

Calistirmak icin (GitHub Actions ile):
  Actions -> "BTC Backtest (180 gun)" -> Run workflow
  Calisma bitince sayfanin altindaki "Artifacts" bolumunden
  signals_180gun.csv dosyasini indirebilirsin.

Yerelde (Termux/bilgisayar) calistirmak icin:
  pip install -r requirements.txt
  python backtest.py            (varsayilan 180 gun)
  python backtest.py 90         (istersen farkli gun sayisi)
"""

import sys
import time
import requests
import pandas as pd

from bot import (
    SYMBOL, BINANCE_KLINES_URL,
    rsi, connors_rsi, ultimate_oscillator, relative_volatility_index,
    roc, majority_rule, money_flow_index,
    RSI_THRESHOLD, CRSI_THRESHOLD, UO_THRESHOLD, RVI_THRESHOLD,
    ROC_THRESHOLD, MAJORITY_THRESHOLD, MFI_THRESHOLD,
)

INTERVAL = "5m"
DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 180
BUFFER_DAYS = 3          # CRSI(100) gibi indikatorlerin isinmasi icin ekstra gecmis
LIMIT = 1000             # Binance'in tek istekteki mum limiti


def fetch_range(symbol, interval, start_ms, end_ms):
    rows = []
    cur = start_ms
    interval_ms = 5 * 60 * 1000
    while cur < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cur, "endTime": end_ms, "limit": LIMIT,
        }
        r = requests.get(BINANCE_KLINES_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows.extend(data)
        cur = data[-1][0] + interval_ms
        print(f"  ... {len(rows)} mum cekildi (son: {pd.to_datetime(data[-1][0], unit='ms')})")
        time.sleep(0.25)
        if len(data) < LIMIT:
            break
    return rows


def build_df(rows):
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["dt"] = pd.to_datetime(df["close_time"], unit="ms")
    return df


def compute_all(df):
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


def main():
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (DAYS + BUFFER_DAYS) * 24 * 60 * 60 * 1000

    print(f"{SYMBOL} {INTERVAL} icin son {DAYS} gun (+{BUFFER_DAYS} gun isinma) cekiliyor...")
    rows = fetch_range(SYMBOL, INTERVAL, start_ms, now_ms)
    df = build_df(rows)
    df = df.iloc[:-1].reset_index(drop=True)  # son mum kapanmamis olabilir
    print(f"Toplam {len(df)} mum indirildi.")

    ind = compute_all(df)

    cond = (
        (ind["rsi14"] <= RSI_THRESHOLD) &
        (ind["crsi"] <= CRSI_THRESHOLD) &
        (ind["uo"] <= UO_THRESHOLD) &
        (ind["rvi10"] <= RVI_THRESHOLD) &
        (ind["roc9"] <= ROC_THRESHOLD) &
        (ind["majority14"] <= MAJORITY_THRESHOLD) &
        (ind["mfi14"] <= MFI_THRESHOLD)
    )

    # Sadece asil istenen DAYS penceresini degerlendir (isinma donemini at)
    cutoff = pd.Timestamp(now_ms - DAYS * 24 * 60 * 60 * 1000, unit="ms")
    mask_window = df["dt"] >= cutoff

    # bot.py'deki gibi: sinyal sadece kosul FALSE -> TRUE'ya donunce sayilir
    rising_edge = cond & (~cond.shift(1).fillna(False))
    signal_mask = rising_edge & mask_window

    signals = df.loc[signal_mask, ["dt", "close"]].copy()
    signals.columns = ["tarih_saat", "fiyat"]
    signals.to_csv("signals_180gun.csv", index=False)

    total = len(signals)
    months = max(DAYS / 30.44, 1e-9)
    monthly_avg = total / months

    print("\n=== AYLIK DAGILIM ===")
    if total > 0:
        by_month = signals["tarih_saat"].dt.to_period("M").value_counts().sort_index()
        for period, count in by_month.items():
            print(f"  {period}: {count} sinyal")
    else:
        print("  Bu esiklerle son donemde hic sinyal olusmadi.")

    print(f"\nToplam sinyal (son {DAYS} gun): {total}")
    print(f"Aylik ortalama: {monthly_avg:.2f} sinyal/ay")
    print("\nDetayli liste 'signals_180gun.csv' dosyasina yazildi.")


if __name__ == "__main__":
    main()
