import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ("epicf", "iraq", "pm25", "cancer")


def setup(dataset, origin=56):
    z = np.load(os.path.join(HERE, "datasets", dataset, "temporal.npz"))
    d = dict(y=z["outcome_factual"][..., 0].astype(float), A=z["treatment_factual"][..., 0].astype(float), S=z["x_factual"][:, 0].astype(float))
    d["T"] = d["y"].shape[1]
    d["o"] = max(3, round(origin / 168 * d["T"]))
    d["H"] = d["T"] - d["o"]
    o = d["o"]
    d.update(Lw=max(2, round(2 * o / 3)), k=max(1, round(o / 6)), step=max(1, o // 28), Ly=np.log1p(np.maximum(d["y"], 0)))
    return d


def split(dataset):
    sp = pd.read_csv(os.path.join(HERE, "datasets", dataset, "split_manifest.csv")).split.to_numpy()
    return np.where(sp == "train")[0], np.where(sp == "test")[0]


def metric(p, truth):
    sc = np.maximum(truth.max(1) - truth.min(1), 1.0)
    return float(np.mean(np.sqrt(((p - truth) ** 2).mean(1)) / sc))


def all_metrics(p, truth, S=None):
    sc = np.maximum(truth.max(1) - truth.min(1), 1.0)
    out = {"R-MSE": round(metric(p, truth), 4), "R-MAE": round(float(np.mean(np.abs(p - truth).mean(1) / sc)), 4)}
    if S is None:
        out["PI90"] = None; out["R-Wass"] = out["R-MAE"]
    else:
        lo, hi = np.quantile(S, .05, 0), np.quantile(S, .95, 0)
        out["PI90"] = round(float(((truth >= lo) & (truth <= hi)).mean()), 3)
        out["R-Wass"] = round(float((np.abs(S - truth[None]).mean(0).mean(1) / sc).mean()), 4)
    return out
