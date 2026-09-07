"""
SAT SINYALI - 180 GUNLUK GERIYE DONUK TEST
=============================================
AL sinyalinin ayna (mirror) versiyonu: RSI, CRSI, UO, RVI, Majority Rule,
MFI'nin hepsi AYNI ANDA asiri alinmis (overbought) esiklerinin UZERINE
ciktiginda "SAT" sinyali sayar.

bot.py icindeki *_SAT_THRESHOLD degerlerini kullanir (henuz botun kendisine
bagli degil, sadece bu script ile kalibrasyon icin).

Cikti AL backtest'i ile ayni format: aylik dagilim, toplam sinyal, aylik
ortalama, ve her indikatorun tek basina esik ustune cikma sikligi.

Calistirmak icin (GitHub Actions ile):
  Actions -> "BTC SAT Backtest (180 gun)" -> Run workflow
"""

import sys
import time
import pandas as pd

from bot import (
    SYMBOL,
    rsi, connors_rsi, ultimate_oscillator, relative_volatility_index,
    roc, majority_rule, money_flow_index,
    RSI_SAT_THRESHOLD, CRSI_SAT_THRESHOLD, UO_SAT_THRESHOLD,
    RVI_SAT_THRESHOLD, MAJORITY_SAT_THRESHOLD, MFI_SAT_THRESHOLD,
)
from backtest import fetch_range, build_df, BUFFER_DAYS

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 180
INTERVAL = "5m"


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
    df = df.iloc[:-1].reset_index(drop=True)
    print(f"Toplam {len(df)} mum indirildi.")

    ind = compute_all(df)

    indiv = {
        "RSI(14)": ind["rsi14"] >= RSI_SAT_THRESHOLD,
        "CRSI(3,2,100)": ind["crsi"] >= CRSI_SAT_THRESHOLD,
        "UO(7,14,28)": ind["uo"] >= UO_SAT_THRESHOLD,
        "RVI(10)": ind["rvi10"] >= RVI_SAT_THRESHOLD,
        "Majority Rule(14)": ind["majority14"] >= MAJORITY_SAT_THRESHOLD,
        "MFI(14)": ind["mfi14"] >= MFI_SAT_THRESHOLD,
    }

    cond = (
        indiv["RSI(14)"] &
        indiv["CRSI(3,2,100)"] &
        indiv["UO(7,14,28)"] &
        indiv["RVI(10)"] &
        indiv["Majority Rule(14)"] &
        indiv["MFI(14)"]
    )

    cutoff = pd.Timestamp(now_ms - DAYS * 24 * 60 * 60 * 1000, unit="ms")
    mask_window = df["dt"] >= cutoff

    print("\n=== HER INDIKATORUN TEK BASINA ESIK USTUNE CIKMA SIKLIGI (SAT) ===")
    print("(yuksek yuzde = cok 'gevsek'/dar esik, sik tetikliyor - darbogaz DEGIL)")
    print("(dusuk yuzde = cok 'siki'/genis esik, nadir tetikliyor - asil darbogaz BU)")
    for name, series in indiv.items():
        pct = series[mask_window].mean() * 100
        print(f"  {name:20s} %{pct:5.1f} mumda esik ustunde")

    avg_count = sum(s[mask_window].astype(int) for s in indiv.values()).mean()
    print(f"\n  Ortalama ayni anda esik ustune cikan indikator sayisi: {avg_count:.2f} / 6")

    rising_edge = cond & (~cond.shift(1).fillna(False))
    signal_mask = rising_edge & mask_window

    signals = df.loc[signal_mask, ["dt", "close"]].copy()
    signals.columns = ["tarih_saat", "fiyat"]
    signals.to_csv("sat_signals_180gun.csv", index=False)

    total = len(signals)
    months = max(DAYS / 30.44, 1e-9)
    monthly_avg = total / months

    print("\n=== AYLIK DAGILIM (SAT) ===")
    if total > 0:
        by_month = signals["tarih_saat"].dt.to_period("M").value_counts().sort_index()
        for period, count in by_month.items():
            print(f"  {period}: {count} sinyal")
    else:
        print("  Bu esiklerle son donemde hic SAT sinyali olusmadi.")

    print(f"\nToplam SAT sinyali (son {DAYS} gun): {total}")
    print(f"Aylik ortalama: {monthly_avg:.2f} sinyal/ay")
    print("\nDetayli liste 'sat_signals_180gun.csv' dosyasina yazildi.")


if __name__ == "__main__":
    main()
