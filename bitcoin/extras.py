"""Numbers and figures for the bitcoin experiments."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.argv = ["corrected.py"]
src = open("corrected.py", encoding="utf-8").read()
exec(src.split("# ---------------- primary model")[0])
fit_part = df.loc[:first_eval - 1]
c = pickle.load(open("ablation_main.pkl", "rb"))
probs, folds, R = c["probs"], c["folds"], c["R"]
pp_reg, pp_noreg = pickle.load(open("primary_main.pkl", "rb"))
df = df.assign(pp_reg=pp_reg, pp_noreg=pp_noreg).dropna(subset=["pp_reg"]).reset_index(drop=True)
WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(WORK, exist_ok=True)

print("rows used", len(df), df.open_time.iloc[0], df.open_time.iloc[-1])
for f, (tr, va, te) in enumerate(folds):
    print("fold", f, "train", df.open_time.iloc[0], "..", df.open_time.iloc[tr.stop - 1], "val", df.open_time.iloc[va.start],
          "test", df.open_time.iloc[te.start], "..", df.open_time.iloc[te.stop - 1])
y = df["target"].values
mask = ~np.isnan(pp_noreg[-len(df):])
from sklearn.metrics import roc_auc_score
print("primary noreg AUC on rows used:", round(roc_auc_score(df["target"], df["pp_noreg"]), 4),
      "reg:", round(roc_auc_score(df["target"], df["pp_reg"]), 4))

idx = np.concatenate([np.arange(f[2].start, f[2].stop) for f in folds])
tst = df.iloc[idx]
print("UP RATE test by regime:", tst.groupby("market_mode")["target"].mean().round(4).to_dict(),
      "n:", tst["market_mode"].value_counts().to_dict())
print("UP RATE fit by regime:", fit_part.groupby("market_mode")["target"].mean().round(4).to_dict())
print("UP RATE overall test", round(tst["target"].mean(), 4), "fit", round(fit_part["target"].mean(), 4))

x_all = 2 * np.concatenate(probs["regime+stack"]["test"]) - 1
print("x quantiles test:", np.quantile(x_all, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(3))

exec("HP = " + src.split("HP = ")[1].split("ARMS = ")[0])
feats = BASE + ["market_mode", "pp_reg"]
tr, va, te = folds[-1]
m = CatBoostClassifier(**HP, random_seed=0, cat_features=["market_mode"])
m.fit(df[feats].iloc[tr], df["target"].iloc[tr], eval_set=(df[feats].iloc[va], df["target"].iloc[va]))
imp = pd.Series(m.get_feature_importance(), index=feats).sort_values(ascending=False)
print("IMPORTANCE (regime+stack, last fold):\n", imp.round(2).to_string())

# ---- regime figures
sse = [KMeans(n_clusters=k, random_state=42, n_init=3).fit(sc.transform(fit_part[CLUS])).inertia_ for k in range(1, 11)]
fig, ax = plt.subplots(figsize=(721 / 100, 393 / 100), dpi=100)
ax.plot(range(1, 11), sse, marker="o", color="#555555")
ax.set_title("Метод локтя для определения числа кластеров")
ax.set_xlabel("Количество кластеров k")
ax.set_ylabel("SSE")
ax.grid(True, alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(WORK, "a2_fig1_new.png"), dpi=100)
plt.close(fig)

NAMES = {"0": "Бычий тренд", "1": "Медвежий тренд", "2": "Боковой с накоплением", "3": "Волатильность", "4": "Боковой с распределением"}
SHADE = {"0": "#4d4d4d", "1": "#7a7a7a", "2": "#cfcfcf", "3": "#202020", "4": "#a6a6a6"}
seg = df.iloc[folds[-1][2].start + 600: folds[-1][2].start + 800].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(1589 / 100, 590 / 100), dpi=100)
for i, md in enumerate(seg["market_mode"]):
    ax.axvspan(i - 0.5, i + 0.5, color=SHADE[md], alpha=0.35, lw=0)
ax.plot(seg["close"].values, color="black", lw=1.1)
handles = [plt.Line2D([0], [0], color=SHADE[k], lw=8, alpha=0.6) for k in NAMES]
ax.legend(handles, NAMES.values(), loc="upper left", fontsize=9, framealpha=0.9)
ax.set_title(f"Рыночные режимы: {seg.open_time.iloc[0]:%d.%m.%Y %H:%M} - {seg.open_time.iloc[-1]:%d.%m.%Y %H:%M} (тестовый период)")
ax.set_xlabel("Номер 15-минутной свечи")
ax.set_ylabel("Цена, USD")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK, "a2_fig2_new.png"), dpi=100)
plt.close(fig)
print("fig2 segment", seg.open_time.iloc[0], seg.open_time.iloc[-1], seg["market_mode"].value_counts().to_dict())

# ---- position-rule figures
exec(src.split("# ---------------- decision rule ----------------")[1].split("close = df[")[0])
xs = np.linspace(-0.3, 0.3, 3001)
fig, ax = plt.subplots(figsize=(1050 / 128, 540 / 128), dpi=128)
ax.plot(xs, rule(xs, -0.08, 0.06, 60.0, 0.5, True), color="black", lw=1.8, label="полное правило (d = -0,08; u = 0,06; β = 60; ε = 0,5)")
ax.plot(xs, np.tanh(12 * xs), color="#555555", lw=1.6, ls="--", label="частный случай d = u = 0, ε = 0: tanh(βx), β = 12")
ax.plot(xs, np.where(xs >= 0.06, 1.0, np.where(xs <= -0.08, -1.0, 0.0)), color="#999999", lw=1.4, ls=":", label="предел β → ∞: пороговое правило")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Центрированная уверенность модели x")
ax.set_ylabel("Позиция P")
ax.set_ylim(-1.1, 1.1)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Семейство правил отображения уверенности в позицию")
fig.tight_layout()
fig.savefig(os.path.join(WORK, "a3_fig1_v2.png"), dpi=128)
plt.close(fig)

F = pd.read_pickle("fee_sweep_all.pkl")
S = F[F.crit.isin(["sr", "-"])].pivot_table(index="fee", columns="kind", values="sr", aggfunc="mean")
fig, ax = plt.subplots(figsize=(1050 / 128, 560 / 128), dpi=128)
style = {"tanh_beta": ("tanh(βx), 1 параметр", "black", "-", "o"),
         "threshold": ("пороговое, 2 параметра", "#666666", "--", "s"),
         "tanh_cont": ("полное правило, 4 параметра", "#888888", "-.", "^"),
         "sign": ("по знаку прогноза", "#aaaaaa", ":", "x")}
for k, (lab, col, ls, mk) in style.items():
    ax.plot(S.index, S[k], color=col, ls=ls, marker=mk, lw=1.6, label=lab)
ax.axhline(0.38, color="black", lw=0.9, ls=(0, (1, 1)), label="купи и держи (0,38)")
ax.axhline(0, color="gray", lw=0.6)
ax.set_xticks(list(S.index))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−")))
ax.set_xlabel("Ставка издержек k, % от оборота")
ax.set_ylabel("Средний SR на тесте (6 мес.)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
ax.set_title("Коэффициент Шарпа на тестовых месяцах в зависимости от издержек")
fig.tight_layout()
fig.savefig(os.path.join(WORK, "a3_fig2.png"), dpi=128)
plt.close(fig)
print("figs saved")
