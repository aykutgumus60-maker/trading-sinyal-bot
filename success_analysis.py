"""
SINYAL BASARI ANALIZI
======================
backtest.py'nin bulduğu her AL sinyalinden sonra:
  - 1 gun icinde RSI(14) hic 70 ve uzerine cikmis mi?
  - 3 gun icinde RSI(14) hic 70 ve uzerine cikmis mi?
kontrol eder. Ayrica her sinyalin o anki tum indikator degerlerini
(RSI, CRSI, UO, RVI, Majority Rule, MFI) ve fiyatini tabloya yazar.

Not: Son 3 gun icindeki sinyaller icin henuz 3 gunluk gelecek verisi
tam olmayabilir - bunlar "henuz degerlendirilemiyor" olarak isaretlenir,
basari oranina dahil edilmez.

Calistirmak icin (GitHub Actions ile):
  Actions -> "BTC Sinyal Basari Analizi" -> Run workflow
  Bitince Artifacts'tan CSV dosyasini indir.

Yerelde calistirmak icin:
  python success_analysis.py            (varsayilan 180 gun)
  python success_analysis.py 90
"""

import sys
import pandas as pd

from bot import (
    rsi, connors_rsi, ultimate_oscillator, relative_volatility_index,
    roc, majority_rule, money_flow_index,
    RSI_THRESHOLD, CRSI_THRESHOLD, UO_THRESHOLD, RVI_THRESHOLD,
    MAJORITY_THRESHOLD, MFI_THRESHOLD, SYMBOL,
)
from backtest import fetch_range, build_df, DAYS as DEFAULT_DAYS, BUFFER_DAYS

import time

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DAYS
RSI_TARGET = 70.0
CANDLES_PER_DAY = 24 * 60 // 5  # 5 dakikalik mum -> gunde 288 mum
WINDOW_1D = 1 * CANDLES_PER_DAY
WINDOW_3D = 3 * CANDLES_PER_DAY


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

    print(f"{SYMBOL} 5m icin son {DAYS} gun (+{BUFFER_DAYS} gun isinma) cekiliyor...")
    rows = fetch_range(SYMBOL, "5m", start_ms, now_ms)
    df = build_df(rows)
    df = df.iloc[:-1].reset_index(drop=True)
    print(f"Toplam {len(df)} mum indirildi.")

    ind = compute_all(df)

    cond = (
        (ind["rsi14"] <= RSI_THRESHOLD) &
        (ind["crsi"] <= CRSI_THRESHOLD) &
        (ind["uo"] <= UO_THRESHOLD) &
        (ind["rvi10"] <= RVI_THRESHOLD) &
        (ind["majority14"] <= MAJORITY_THRESHOLD) &
        (ind["mfi14"] <= MFI_THRESHOLD)
    )

    cutoff = pd.Timestamp(now_ms - DAYS * 24 * 60 * 60 * 1000, unit="ms")
    mask_window = df["dt"] >= cutoff
    rising_edge = cond & (~cond.shift(1).fillna(False))
    signal_idx = df.index[rising_edge & mask_window].tolist()

    print(f"\nToplam {len(signal_idx)} sinyal bulundu. Her biri icin ileri yon kontrolu yapiliyor...")

    rows_out = []
    n_total_bilgi_yeterli_1g = 0
    n_basarili_1g = 0
    n_total_bilgi_yeterli_3g = 0
    n_basarili_3g = 0

    max_pct_1g_list = []
    max_pct_3g_list = []
    min_pct_3g_list = []

    for i in signal_idx:
        sig_time = df["dt"].iloc[i]
        sig_price = df["close"].iloc[i]

        # 1 GUN kontrolu
        end_1d = i + WINDOW_1D
        has_full_1d = end_1d < len(df)
        basari_1g = None
        max_pct_1g = None
        if has_full_1d:
            window = ind["rsi14"].iloc[i + 1: end_1d + 1]
            basari_1g = bool((window >= RSI_TARGET).any())
            n_total_bilgi_yeterli_1g += 1
            if basari_1g:
                n_basarili_1g += 1
            price_window_1d = df["close"].iloc[i + 1: end_1d + 1]
            max_pct_1g = (price_window_1d.max() / sig_price - 1) * 100
            max_pct_1g_list.append(max_pct_1g)

        # 3 GUN kontrolu
        end_3d = i + WINDOW_3D
        has_full_3d = end_3d < len(df)
        basari_3g = None
        ilk_70_zamani = None
        ilk_70_fiyat = None
        yuzde_degisim = None
        max_pct_3g = None
        min_pct_3g = None
        if has_full_3d:
            window3 = ind["rsi14"].iloc[i + 1: end_3d + 1]
            hit = window3[window3 >= RSI_TARGET]
            basari_3g = bool(len(hit) > 0)
            n_total_bilgi_yeterli_3g += 1
            if basari_3g:
                n_basarili_3g += 1
                first_hit_idx = hit.index[0]
                ilk_70_zamani = df["dt"].iloc[first_hit_idx]
                ilk_70_fiyat = df["close"].iloc[first_hit_idx]
                yuzde_degisim = (ilk_70_fiyat / sig_price - 1) * 100

            price_window_3d = df["close"].iloc[i + 1: end_3d + 1]
            max_pct_3g = (price_window_3d.max() / sig_price - 1) * 100
            min_pct_3g = (price_window_3d.min() / sig_price - 1) * 100
            max_pct_3g_list.append(max_pct_3g)
            min_pct_3g_list.append(min_pct_3g)

        rows_out.append({
            "tarih_saat": sig_time,
            "fiyat": round(sig_price, 2),
            "RSI14": round(ind["rsi14"].iloc[i], 2),
            "CRSI": round(ind["crsi"].iloc[i], 2),
            "UO": round(ind["uo"].iloc[i], 2),
            "RVI": round(ind["rvi10"].iloc[i], 2),
            "MajorityRule": round(ind["majority14"].iloc[i], 2),
            "MFI": round(ind["mfi14"].iloc[i], 2),
            "1gun_icinde_RSI70": basari_1g if has_full_1d else "veri_yetersiz",
            "1gun_max_yuzde_kazanc": round(max_pct_1g, 2) if max_pct_1g is not None else None,
            "3gun_icinde_RSI70": basari_3g if has_full_3d else "veri_yetersiz",
            "70e_ulasma_zamani": ilk_70_zamani,
            "70deki_fiyat": round(ilk_70_fiyat, 2) if ilk_70_fiyat else None,
            "yuzde_degisim_70e_kadar": round(yuzde_degisim, 2) if yuzde_degisim is not None else None,
            "3gun_max_yuzde_kazanc": round(max_pct_3g, 2) if max_pct_3g is not None else None,
            "3gun_max_yuzde_kayip": round(min_pct_3g, 2) if min_pct_3g is not None else None,
        })

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv("signal_basari_analizi.csv", index=False)

    print("\n=== SONUC ===")
    if n_total_bilgi_yeterli_1g > 0:
        pct1 = n_basarili_1g / n_total_bilgi_yeterli_1g * 100
        print(f"1 GUN icinde RSI70+: {n_basarili_1g} / {n_total_bilgi_yeterli_1g} sinyal  (%{pct1:.1f})")
    else:
        print("1 gunluk analiz icin yeterli veri yok.")

    if n_total_bilgi_yeterli_3g > 0:
        pct3 = n_basarili_3g / n_total_bilgi_yeterli_3g * 100
        print(f"3 GUN icinde RSI70+: {n_basarili_3g} / {n_total_bilgi_yeterli_3g} sinyal  (%{pct3:.1f})")
    else:
        print("3 gunluk analiz icin yeterli veri yok.")

    degerlendirilemeyen = len(signal_idx) - n_total_bilgi_yeterli_3g
    if degerlendirilemeyen > 0:
        print(f"(Son {degerlendirilemeyen} sinyal icin henuz 3 gunluk gelecek verisi olusmadi, sayima dahil edilmedi.)")

    if n_basarili_3g > 0:
        basarili_yuzdeler = out_df.loc[out_df["yuzde_degisim_70e_kadar"].notna(), "yuzde_degisim_70e_kadar"]
        print(f"\nBasarili sinyallerde ortalama fiyat artisi (sinyalden 70'e kadar): %{basarili_yuzdeler.mean():.2f}")

    print("\n=== FIYAT DEGISIM DAGILIMI (TUM SINYALLER) ===")
    if max_pct_1g_list:
        s = pd.Series(max_pct_1g_list)
        print(f"1 GUN icindeki EN YUKSEK kazanc %  -> min: {s.min():.2f}  medyan: {s.median():.2f}  ortalama: {s.mean():.2f}  maks: {s.max():.2f}")
    if max_pct_3g_list:
        s = pd.Series(max_pct_3g_list)
        print(f"3 GUN icindeki EN YUKSEK kazanc %  -> min: {s.min():.2f}  medyan: {s.median():.2f}  ortalama: {s.mean():.2f}  maks: {s.max():.2f}")
    if min_pct_3g_list:
        s = pd.Series(min_pct_3g_list)
        print(f"3 GUN icindeki EN BUYUK kayip   %  -> min: {s.min():.2f}  medyan: {s.median():.2f}  ortalama: {s.mean():.2f}  maks: {s.max():.2f}")

    print("\nDetayli tablo 'signal_basari_analizi.csv' dosyasina yazildi.")


if __name__ == "__main__":
    main()
