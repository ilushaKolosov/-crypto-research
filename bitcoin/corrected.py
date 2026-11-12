"""Bitcoin 15-minute experiments.

Fixes relative to the original notebook:
  * single log transform (close_return = diff log close); GK, Amihud, VWAP on raw prices;
  * scaler + k-means fit only on data preceding the first evaluation block (causal regimes);
  * rows without an out-of-sample primary probability are dropped;
  * 2x2 ablation (regime label x stacked primary probability) on identical base features and
    identical fixed hyperparameters, 3 CatBoost seeds, walk-forward over the last 6 months;
  * Google Trends column excluded (full-period normalisation, interpolation uses future points);
  * DeLong test for paired AUC on the pooled test months.
Decision rule tuned on the month preceding each test month (seeded TPE),
cost on position change k*S*|dP| and original holding cost, continuous short branch, baselines.
"""
import os
import pickle
import sys
import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
WITH_TRENDS = "--trends" in sys.argv
TAG = "trends" if WITH_TRENDS else "main"
BLOCK = 2880
N_FOLDS = 6
SEEDS = (0, 1, 2)
K = 0.02 / 100
NY = 24 * 4 * 365


def P(*a):
    print(*a, flush=True)


# ---------------- features ----------------
def realized_vol(lp1, N):
    return np.sqrt((lp1.diff() ** 2).rolling(N).mean()).resample("15min").last()


def build_features(d15, d1, N1=3, N2=14):
    d = d15.set_index("open_time").copy()
    h = d1.set_index("open_time")
    lc = np.log(d["close"])
    lp1 = np.log(h["close"])
    d["target"] = (d["close"].shift(-1) > d["close"]).astype(int)
    d["close_return"] = lc.diff()
    for N in (N1, N2):
        s = realized_vol(lp1, N * 15).reindex(d.index)
        mu = s ** 2 / 2 + (lc - lc.shift(N)) / N
        d[f"realized_vol_{N}"] = s
        d[f"trend_strength_{N}"] = (mu - s ** 2 / 2) * (np.sqrt(N) / s)
    d["market_pressure"] = 2 * d["taker_buy_base_volume"] / d["volume"] - 1
    hl, co = np.log(d["high"] / d["low"]), np.log(d["close"] / d["open"])
    d["gk_vol"] = np.sqrt(np.maximum(0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2, 0))
    d["amihud"] = d["close"].pct_change().abs() / d["quote_volume"]
    d["vwap"] = (d["close"] * d["volume"]).cumsum() / d["volume"].cumsum()
    return d.reset_index()


CLUS = ["trend_strength_3", "trend_strength_14", "realized_vol_3", "realized_vol_14", "market_pressure"]
BASE = ["volume", "close_return", "realized_vol_3", "realized_vol_14", "trend_strength_3",
        "trend_strength_14", "market_pressure", "gk_vol", "amihud", "vwap"]
if WITH_TRENDS:
    BASE = ["trends"] + BASE

d15 = pd.read_csv(os.path.join("data", "df_15min.csv.gz"))
d15["open_time"] = pd.to_datetime(d15["open_time"])
d1 = pd.read_csv(os.path.join("data", "df_1min.csv.gz"))
d1["open_time"] = pd.to_datetime(d1["open_time"])
df = build_features(d15, d1)
df = df.iloc[:-1]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=BASE + ["target"]).reset_index(drop=True)
n = len(df)
first_eval = n - (N_FOLDS + 1) * BLOCK
P(f"[{TAG}] rows {n}, {df.open_time.iloc[0]} .. {df.open_time.iloc[-1]}; first eval row {first_eval} ({df.open_time.iloc[first_eval]})")

sc = StandardScaler().fit(df.loc[:first_eval - 1, CLUS])
km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(sc.transform(df.loc[:first_eval - 1, CLUS]))
df["market_mode"] = km.predict(sc.transform(df[CLUS])).astype(str)
P("regime shares (fit part):", df.loc[:first_eval - 1, "market_mode"].value_counts(normalize=True).round(3).to_dict())


# ---------------- primary model, rolling out-of-sample ----------------
def rolling_oos(feats, cat, window=30000, shift=3000):
    X, Y = df[feats], df["target"]
    pr = np.full(n, np.nan)
    mk = lambda: CatBoostClassifier(iterations=300, learning_rate=0.1, depth=5, l2_leaf_reg=3,
                                    verbose=0, random_seed=0, cat_features=cat, thread_count=-1)
    for tr, va in TimeSeriesSplit(5).split(X.iloc[:window]):
        m = mk().fit(X.iloc[tr], Y.iloc[tr])
        pr[va] = m.predict_proba(X.iloc[va])[:, 1]
    aucs = []
    for s in range(0, n - window, shift):
        a, b = s + window, min(s + window + shift, n)
        m = mk().fit(X.iloc[s:a], Y.iloc[s:a])
        pr[a:b] = m.predict_proba(X.iloc[a:b])[:, 1]
        aucs.append(roc_auc_score(Y.iloc[a:b], pr[a:b]))
    P(f"  primary {'with' if cat else 'without'} regime: {len(aucs)} windows, mean AUC {np.mean(aucs):.4f}")
    return pr


cache = f"primary_{TAG}.pkl"
if os.path.exists(cache):
    df["pp_reg"], df["pp_noreg"] = pickle.load(open(cache, "rb"))
else:
    df["pp_reg"] = rolling_oos(BASE + ["market_mode"], ["market_mode"])
    df["pp_noreg"] = rolling_oos(BASE, [])
    pickle.dump((df["pp_reg"].values, df["pp_noreg"].values), open(cache, "wb"))
df = df.dropna(subset=["pp_reg"]).reset_index(drop=True)
n = len(df)

# ---------------- 2x2 ablation, walk-forward ----------------
HP = dict(iterations=1000, learning_rate=0.05655, depth=6, l2_leaf_reg=10.7376, bagging_temperature=0.6119,
          random_strength=0.9061, loss_function="Logloss", eval_metric="AUC", use_best_model=True,
          early_stopping_rounds=50, verbose=0, thread_count=-1)
ARMS = {
    "base": (BASE, False),
    "regime": (BASE + ["market_mode"], False),
    "stack": (BASE + ["pp_noreg"], False),
    "regime+stack": (BASE + ["market_mode", "pp_reg"], True),
}
folds = []
for f in range(N_FOLDS):
    te0 = n - (N_FOLDS - f) * BLOCK
    folds.append((slice(0, te0 - BLOCK), slice(te0 - BLOCK, te0), slice(te0, te0 + BLOCK)))

res, probs = [], {a: {"val": [], "test": []} for a in ARMS}
for a, (feats, _) in (ARMS.items() if not os.path.exists(f"ablation_{TAG}.pkl") else []):
    cat = ["market_mode"] if "market_mode" in feats else []
    for f, (tr, va, te) in enumerate(folds):
        pv, pt = [], []
        for s in SEEDS:
            m = CatBoostClassifier(**HP, random_seed=s, cat_features=cat)
            m.fit(df.loc[df.index[tr], feats], df["target"].iloc[tr], eval_set=(df[feats].iloc[va], df["target"].iloc[va]))
            pv.append(m.predict_proba(df[feats].iloc[va])[:, 1])
            pt.append(m.predict_proba(df[feats].iloc[te])[:, 1])
        pv, pt = np.mean(pv, 0), np.mean(pt, 0)
        y = df["target"].iloc[te].values
        probs[a]["val"].append(pv)
        probs[a]["test"].append(pt)
        res.append(dict(arm=a, fold=f, month=str(df.open_time.iloc[te.start])[:10],
                        auc=roc_auc_score(y, pt), acc=accuracy_score(y, pt > 0.5)))
        P(f"  {a:13s} fold {f} {res[-1]['month']}: AUC {res[-1]['auc']:.4f} acc {res[-1]['acc']:.4f}")
if res:
    R = pd.DataFrame(res)
    pickle.dump(dict(R=R, probs=probs, folds=folds), open(f"ablation_{TAG}.pkl", "wb"))
else:
    _c = pickle.load(open(f"ablation_{TAG}.pkl", "rb"))
    R, probs = _c["R"], _c["probs"]
P("\nABLATION per month (AUC):\n", R.pivot(index="month", columns="arm", values="auc").round(4).to_string())
P("\nABLATION mean over months:\n", R.groupby("arm")[["auc", "acc"]].agg(["mean", "std"]).round(4).to_string())


def delong(y, p1, p2):
    pos, neg = y == 1, y == 0
    m, k = pos.sum(), neg.sum()
    out = []
    for p in (p1, p2):
        X, Yn = p[pos], p[neg]
        r_all = stats.rankdata(np.r_[X, Yn])
        rx, ry = stats.rankdata(X), stats.rankdata(Yn)
        v10 = (r_all[:m] - rx) / k
        v01 = 1 - (r_all[m:] - ry) / m
        out.append((v10.mean(), v10, v01))
    s10 = np.cov(np.vstack([out[0][1], out[1][1]]))
    s01 = np.cov(np.vstack([out[0][2], out[1][2]]))
    S = s10 / m + s01 / k
    d = out[0][0] - out[1][0]
    z = d / np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    return out[0][0], out[1][0], z, 2 * stats.norm.sf(abs(z))


yall = np.concatenate([df["target"].iloc[te].values for _, _, te in folds])
pall = {a: np.concatenate(probs[a]["test"]) for a in ARMS}
for a, b in (("stack", "base"), ("regime", "base"), ("regime+stack", "regime"), ("regime+stack", "base")):
    A, B, z, p = delong(yall, pall[a], pall[b])
    dm = R[R.arm == a].set_index("fold").auc - R[R.arm == b].set_index("fold").auc
    P(f"DELONG pooled {a} vs {b}: {A:.4f} vs {B:.4f}, z={z:.2f}, p={p:.3f}; "
      f"monthly diff {dm.mean():+.4f} ± {dm.std(ddof=1) / np.sqrt(len(dm)):.4f}, wins {int((dm > 0).sum())}/{len(dm)}")
for a in ARMS:
    P(f"POOLED {a}: AUC {roc_auc_score(yall, pall[a]):.4f} acc {accuracy_score(yall, pall[a] > 0.5):.4f}")


# ---------------- decision rule ----------------
def rule(x, d, u, beta, eps, cont):
    pos = np.zeros_like(x)
    L, S = x >= u, x <= d
    pos[L] = np.tanh(beta * (x[L] - u) ** (1 + eps))
    pos[S] = np.tanh(-beta * np.abs(x[S] - d if cont else x[S] + d) ** (1 + eps))
    return pos


def pnl(pos, S, cost):
    p = pos[:-1]
    gross = p * (S[1:] - S[:-1])
    if cost == "hold":
        c = S[1:] * np.abs(p) * K
    else:
        c = S[:-1] * np.abs(np.diff(np.r_[0.0, p])) * K
    return gross - c


def stats_(r):
    vol = r.std() * np.sqrt(NY)
    return r.mean() * NY, vol, (r.mean() * NY / vol if vol > 0 else np.nan)


def tune(x, S, crit, cost, kind, seed=0, n_trials=600):
    def obj(t):
        if kind == "linear":
            pos = np.clip(t.suggest_float("c", 0.1, 100, log=True) * x, -1, 1)
        else:
            d = t.suggest_float("d", -1.0, 1.0)
            u = t.suggest_float("u", d, 1.0)
            if kind == "threshold":
                pos = np.where(x >= u, 1.0, np.where(x <= d, -1.0, 0.0))
            else:
                pos = rule(x, d, u, t.suggest_float("beta", 1.0, 100.0), t.suggest_float("epsilon", 0.01, 10.0), kind == "tanh_cont")
        r = pnl(pos, S, cost)
        if crit == "pnl":
            return r.sum()
        sr = stats_(r)[2]
        return -1e9 if not np.isfinite(sr) else sr

    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    return st.best_params


def apply(x, prm, kind):
    if kind == "linear":
        return np.clip(prm["c"] * x, -1, 1)
    if kind == "threshold":
        return np.where(x >= prm["u"], 1.0, np.where(x <= prm["d"], -1.0, 0.0))
    return rule(x, prm["d"], prm["u"], prm["beta"], prm["epsilon"], kind == "tanh_cont")


close = df["close"].values
rows = []
MAIN = "regime+stack"
for f, (tr, va, te) in enumerate(folds):
    xv, xt = 2 * probs[MAIN]["val"][f] - 1, 2 * probs[MAIN]["test"][f] - 1
    Sv, St = close[va], close[te]
    month = str(df.open_time.iloc[te.start])[:10]
    bh = St[1:] - St[:-1]
    rows.append(dict(month=month, strategy="buy&hold", cost="-", crit="-", **dict(zip(("ret", "vol", "sr"), stats_(bh)))))
    for cost in ("change", "hold"):
        rows.append(dict(month=month, strategy="sign", cost=cost, crit="-", **dict(zip(("ret", "vol", "sr"), stats_(pnl(np.sign(xt), St, cost))))))
        for kind in ("tanh_orig", "tanh_cont", "threshold", "linear"):
            if cost == "hold" and kind != "tanh_orig":
                continue
            for crit in ("pnl", "sr"):
                prm = tune(xv, Sv, crit, cost, kind, seed=f)
                r = pnl(apply(xt, prm, kind), St, cost)
                rows.append(dict(month=month, strategy=kind, cost=cost, crit=crit, **dict(zip(("ret", "vol", "sr"), stats_(r)))))
    P(f"  rule fold {f} {month} done")
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
P("\nRULE per month:\n", T.pivot_table(index=["strategy", "cost", "crit"], columns="month", values="sr").round(2).to_string())
agg = T.groupby(["strategy", "cost", "crit"]).agg(ret=("ret", "mean"), vol=("vol", "mean"), sr_mean=("sr", "mean"),
                                                   sr_med=("sr", "median"), sr_pos=("sr", lambda s: int((s > 0).sum())))
P("\nRULE mean over months:\n", agg.round(2).to_string())
pickle.dump(dict(R=R, T=T, probs=probs, folds=folds, months=R.month.unique().tolist()), open(f"corrected_{TAG}.pkl", "wb"))
P("DONE")
