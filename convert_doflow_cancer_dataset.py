"""Builds datasets/cancer from the DoFlow cancer file.

python convert_doflow_cancer_dataset.py path/to/training_data.csv
"""
import os, sys, numpy as np, pandas as pd

df = pd.read_csv(sys.argv[1]).sort_values(["patient_id", "time"])
L = df.groupby("patient_id").size()
T = int(L.mode().iloc[0])
keep = L[L >= T].index
keep = np.sort(np.random.default_rng(0).choice(keep, 10000, replace=False))
df = df[df.patient_id.isin(keep)].groupby("patient_id").head(T)
n = len(keep)

y = df.cancer_volume.to_numpy().reshape(n, T, 1).astype("float32")
a = ((df.chemo_application.to_numpy() + df.radio_application.to_numpy()) > 0).reshape(n, T, 1).astype("float32")
x = np.zeros((n, T, 1), "float32")

os.makedirs("datasets/cancer", exist_ok=True)
np.savez_compressed("datasets/cancer/temporal.npz", outcome_factual=y, treatment_factual=a, x_factual=x)
split = np.where(np.random.default_rng(0).permutation(n) < round(0.8 * n), "train", "test")
pd.DataFrame({"unit": np.arange(n), "split": split}).to_csv("datasets/cancer/split_manifest.csv", index=False)
print("patients", n, "train", (split == "train").sum(), "test", (split == "test").sum(), "days", T)
