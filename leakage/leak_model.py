# -*- coding: utf-8 -*-
"""Контролируемый эксперимент: утечка будущего при шумоподавлении (таблица 1, рисунки 1-2).

Идея: латентная «эффективная» цена = случайное блуждание x_t (нет реальной
предсказуемости знака приращения). Наблюдаемая цена y_t = x_t + eps_t, где
eps_t — микроструктурный шум. Сравниваем предобработку:
  - dwt_full     : вейвлет db4, универсальный мягкий порог, по ВСЕМУ ряду (некаузальный);
  - dwt_causal   : тот же вейвлет, но в скользящем окне, оставляем только последнюю точку;
  - kf_filter    : фильтр Калмана (локальный уровень), онлайн/каузальный;
  - kf_smoother  : сглаживатель Калмана–RTS, использует будущее (некаузальный).

Метрики:
  1) качество очистки: RMSE к истинному x_t (и снижение против наблюдения);
  2) утечка будущего: наивная стратегия sign(Δ denoised_t) на приращении y_{t+1};
     на случайном блуждании честная точность = 50%, Sharpe = 0; любой избыток —
     это «протечка» будущего в признак;
  3) запаздывание каузального фильтра (сдвиг кросс-корреляции).
"""
import json
import os
import numpy as np
import pandas as pd
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
BARS_PER_YEAR = 365 * 24 * 4          # 15-минутные бары, рынок 24/7
SIG = 0.0010                          # ст. отклонение приращения латентной цены
NOISE = 0.0022                        # ст. отклонение микроструктурного шума
N = 2000
WAVELET = "db4"
LEVELS = 5


# ----------------------------- алгоритмы ------------------------------------
def dwt_full(y, wavelet=WAVELET, levels=LEVELS, mode="constant"):
    """Неканоничное шумоподавление по всему ряду (как dwt_denoise в ноутбуке)."""
    coeffs = pywt.wavedec(y, wavelet, level=levels, mode=mode)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(y)))
    c = [coeffs[0]] + [pywt.threshold(ci, thr, mode="soft") for ci in coeffs[1:]]
    rec = pywt.waverec(c, wavelet, mode=mode)
    return np.asarray(rec[:len(y)])


def dwt_causal(y, wavelet=WAVELET, levels=LEVELS, window=None, mode="constant"):
    """Каузальный вариант: DWT в скользящем окне, берём только последнюю точку."""
    dec_len = pywt.Wavelet(wavelet).dec_len
    if window is None:
        window = dec_len * (2 ** levels)
    out = np.array(y, dtype=float)
    for end in range(1, len(y) + 1):
        start = max(0, end - window)
        w = y[start:end]
        if len(w) < dec_len * 2:
            out[end - 1] = w[-1]
            continue
        lv = min(levels, pywt.dwt_max_level(len(w), dec_len))
        coeffs = pywt.wavedec(w, wavelet, level=lv, mode=mode)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(len(w))) if sigma > 0 else 0.0
        c = [coeffs[0]] + [pywt.threshold(ci, thr, mode="soft") for ci in coeffs[1:]]
        rec = pywt.waverec(c, wavelet, mode=mode)
        out[end - 1] = rec[:len(w)][-1]
    return out


def kalman_local_level(y, q, r):
    """Каузальный фильтр Калмана для модели локального уровня + сглаживатель RTS."""
    n = len(y)
    xf = np.zeros(n)      # апостериорная оценка (фильтр)
    Pf = np.zeros(n)
    xp = np.zeros(n)      # априорная оценка
    Pp = np.zeros(n)
    x, P = y[0], 1.0
    for t in range(n):
        # прогноз
        xpr, Ppr = x, P + q
        xp[t], Pp[t] = xpr, Ppr
        # коррекция
        K = Ppr / (Ppr + r)
        x = xpr + K * (y[t] - xpr)
        P = (1 - K) * Ppr
        xf[t], Pf[t] = x, P
    # сглаживатель Рауха–Тунга–Штрибеля (использует будущее)
    xs = xf.copy()
    Ps = Pf.copy()
    for t in range(n - 2, -1, -1):
        C = Pf[t] / Pp[t + 1]
        xs[t] = xf[t] + C * (xs[t + 1] - xp[t + 1])
        Ps[t] = Pf[t] + C * C * (Ps[t + 1] - Pp[t + 1])
    return xf, xs


# ----------------------------- симуляция ------------------------------------
def simulate(seed):
    rng = np.random.default_rng(seed)
    incr = rng.normal(0.0, SIG, N)
    x = np.cumsum(incr)                    # латентная лог-цена (случайное блуждание)
    eps = rng.normal(0.0, NOISE, N)
    y = x + eps                            # наблюдаемая лог-цена
    return x, y


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def strat_metrics(denoised, y, x=None):
    """Наивная стратегия по знаку приращения очищенного ряда.
    Сделка в t по знаку (denoised_t - denoised_{t-1}), доход = y_{t+1} - y_t.
    Дополнительно (если задан x) — точность предсказания знака ИСТИННОГО
    будущего приращения x_{t+1}-x_t: прямой тест на утечку будущего."""
    d = np.diff(denoised)                  # длина N-1, индекс t=1..N-1 приращение
    sig = np.sign(d[:-1])                  # сигнал в t (t=1..N-2)
    fut = np.diff(y)[1:]                   # доход t->t+1 (y_{t+1}-y_t), выравнен с sig
    pnl = sig * fut
    mask = sig != 0
    acc = float(np.mean((sig[mask] > 0) == (fut[mask] > 0))) if mask.sum() else 0.5
    mu, sd = pnl.mean(), pnl.std(ddof=1)
    sharpe = float(mu / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    acc_true = None
    if x is not None:
        futx = np.diff(x)[1:]
        acc_true = float(np.mean((sig[mask] > 0) == (futx[mask] > 0))) if mask.sum() else 0.5
    return acc, sharpe, np.cumsum(pnl), acc_true


def lag_of(denoised, x):
    """Сдвиг максимума кросс-корреляции приращений (запаздывание фильтра)."""
    a = np.diff(denoised); b = np.diff(x)
    a = (a - a.mean()); b = (b - b.mean())
    lags = range(0, 15)
    cc = [np.corrcoef(a[k:], b[:len(b) - k])[0, 1] if len(b) - k > 5 else 0 for k in lags]
    return int(np.argmax(cc))


# ----------------------------- Монте-Карло ----------------------------------
def monte_carlo(M=50):
    q = SIG ** 2
    r = NOISE ** 2
    keys = ["obs", "dwt_full", "dwt_causal", "kf_filter", "kf_smoother"]
    agg = {k: {"rmse": [], "acc": [], "acc_true": [], "sharpe": [], "lag": []} for k in keys}
    for s in range(M):
        x, y = simulate(1000 + s)
        series = {
            "obs": y,
            "dwt_full": dwt_full(y),
            "dwt_causal": dwt_causal(y),
        }
        kf, ks = kalman_local_level(y, q, r)
        series["kf_filter"] = kf
        series["kf_smoother"] = ks
        for k, d in series.items():
            agg[k]["rmse"].append(rmse(d, x))
            acc, sh, _, acc_true = strat_metrics(d, y, x)
            agg[k]["acc"].append(acc)
            agg[k]["acc_true"].append(acc_true)
            agg[k]["sharpe"].append(sh)
            agg[k]["lag"].append(lag_of(d, x))
    res = {}
    base = np.mean(agg["obs"]["rmse"])
    for k in keys:
        rm = np.array(agg[k]["rmse"])
        res[k] = {
            "rmse_mean": float(rm.mean()),
            "rmse_std": float(rm.std()),
            "rmse_reduction_pct": float((1 - rm.mean() / base) * 100),
            "acc_mean": float(np.mean(agg[k]["acc"])),
            "acc_std": float(np.std(agg[k]["acc"])),
            "acc_true_mean": float(np.mean(agg[k]["acc_true"])),
            "acc_true_std": float(np.std(agg[k]["acc_true"])),
            "sharpe_mean": float(np.mean(agg[k]["sharpe"])),
            "sharpe_std": float(np.std(agg[k]["sharpe"])),
            "lag_median": float(np.median(agg[k]["lag"])),
        }
    return res


# ----------------------------- графики --------------------------------------
def make_figures():
    x, y = simulate(1000)
    df = dwt_full(y)
    dc = dwt_causal(y)
    kf, ks = kalman_local_level(y, SIG ** 2, NOISE ** 2)

    a, b = 900, 1050
    plt.figure(figsize=(7.2, 3.7))
    idx = np.arange(a, b)
    plt.plot(idx, y[a:b], color="0.72", lw=0.9, label="Наблюдаемая цена (с шумом)")
    plt.plot(idx, x[a:b], color="black", lw=1.6, label="Истинная (латентная) цена")
    plt.plot(idx, df[a:b], color="tab:red", lw=1.3, ls="--",
             label="Вейвлет по всему ряду (некаузальный)")
    plt.plot(idx, dc[a:b], color="tab:blue", lw=1.3,
             label="Вейвлет каузальный (оконный)")
    plt.xlabel("Отсчёт времени t"); plt.ylabel("Лог-цена")
    plt.legend(fontsize=7, loc="best"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/bi_fig1_denoise.png", dpi=300)
    plt.close()

    # накопленный PnL наивной стратегии
    _, _, pnl_full, _ = strat_metrics(df, y)
    _, _, pnl_caus, _ = strat_metrics(dc, y)
    _, _, pnl_sm, _ = strat_metrics(ks, y)
    _, _, pnl_kf, _ = strat_metrics(kf, y)
    plt.figure(figsize=(7.2, 3.7))
    plt.plot(pnl_full, color="tab:red", lw=1.5, ls="--",
             label="Вейвлет по всему ряду (некаузальный)")
    plt.plot(pnl_sm, color="tab:orange", lw=1.2, ls=":",
             label="Сглаживатель Калмана–RTS (некаузальный)")
    plt.plot(pnl_caus, color="tab:blue", lw=1.5, label="Вейвлет каузальный")
    plt.plot(pnl_kf, color="tab:green", lw=1.2, label="Фильтр Калмана (каузальный)")
    plt.axhline(0, color="gray", lw=0.6)
    plt.xlabel("Отсчёт времени t"); plt.ylabel("Накопленный PnL (лог-доход)")
    plt.legend(fontsize=7, loc="best"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/bi_fig2_leakage_pnl.png", dpi=300)
    plt.close()
    print("figures saved")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    make_figures()
    res = monte_carlo(M=50)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "table1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
