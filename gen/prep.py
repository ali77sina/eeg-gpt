"""Prepare continuous 100 Hz F3-F4 signals for generative modelling.
0.5-40 Hz bandpass @250 Hz -> resample 100 Hz -> robust per-night scaling (median/IQR) -> clip +-10.
Train = first 20 nights (sorted id), held-out = last 5."""
import os, glob, json, numpy as np
from scipy.signal import butter, lfilter, resample
ROOT = os.path.expanduser("~/morph-net"); SRC = f"{ROOT}/data/dodh_npz"; OUT = f"{ROOT}/data/gen"; os.makedirs(OUT, exist_ok=True)
CHAN = "F3_F4"; FS = 100
b, a = butter(5, [0.5 / 125, 40 / 125], btype="band")
rids = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{SRC}/*.npz"))
split = {"train": rids[:20], "heldout": rids[20:]}
json.dump(split, open(f"{OUT}/split.json", "w"), indent=1)
for r in rids:
    d = np.load(f"{SRC}/{r}.npz"); x = lfilter(b, a, d[CHAN].astype(np.float64)); hyp = d["hypnogram"].astype(np.int8)
    x = resample(x, int(len(x) * FS / 250)); n = min(len(x) // (30 * FS), len(hyp)); x = x[: n * 30 * FS]; hyp = hyp[:n]
    med = np.median(x); iqr = np.subtract(*np.percentile(x, [75, 25])); x = np.clip((x - med) / (iqr / 1.349), -10, 10).astype(np.float32)
    np.savez(f"{OUT}/{r}.npz", x=x, hyp=hyp)
    print(r, "train" if r in split["train"] else "heldout", x.shape, f"std={x.std():.2f} clipped={(np.abs(x) >= 10).mean()*100:.3f}%")
