"""Supplementary probes from cached features: DOD-O (apnea patients, never trained on) as a zero-shot transfer target,
plus a pooled 'all frontal-mastoid' and 'all frontal-bipolar' probe across datasets."""
import os, sys, json, csv, glob, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
OUT = os.environ.get("GEN_OUT", "/workspace/gen")
def canon(ch):
    c = ch.upper().replace("A1", "M1").replace("A2", "M2")
    if c.startswith("HB"): return "HB-bipolar" if "-" in c else "HB-raw"
    site = re.match(r"^(FPZ|FP|F|CZ|C|PZ|P|O|T)", c); site = site.group(1) if site else "X"; site = {"FPZ": "Fp", "FP": "Fp", "CZ": "C", "PZ": "P"}.get(site, site)
    if "-" not in c: return f"{site}-raw"
    ref = c.split("-")[1]; m = re.match(r"^(FP|F|C|P|O|T)", ref); refc = "M" if ref.startswith("M") else (m.group(1) if m else "X")
    return f"{site}-{refc}"
feat = {}
for p in glob.glob(f"{OUT}/feat/*.npz"):
    ds, rec, ch = os.path.basename(p)[:-4].split("__", 2); d = np.load(p, allow_pickle=True)
    feat[(ds, rec, ch)] = dict(X=d["feats"].item()["30s"], y=d["y"], nll=float(d["nll_tok"].mean()), group=canon(ch))
split = {(s["dataset"], s["rec_id"], s["chan"]): s["split"] for s in csv.DictReader(open(f"{OUT}/streams.csv"))}
def sel(ds=None, group=None, sp=None):
    return [k for k, v in feat.items() if (ds is None or k[0] in ds) and (group is None or v["group"] in group) and (sp is None or split.get(k) == sp)]
def probe(tr, te):
    if not tr or not te: return None
    Xtr = np.concatenate([feat[k]["X"] for k in tr]); ytr = np.concatenate([feat[k]["y"] for k in tr]); Xte = np.concatenate([feat[k]["X"] for k in te]); yte = np.concatenate([feat[k]["y"] for k in te])
    m, mt = ytr >= 0, yte >= 0
    if m.sum() < 500 or mt.sum() < 100: return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.5)).fit(Xtr[m], ytr[m]); p = clf.predict(Xte[mt])
    return dict(acc=float(accuracy_score(yte[mt], p)), mf1=float(f1_score(yte[mt], p, average="macro")), n_test=int(mt.sum()), n_train=int(m.sum()), cm=confusion_matrix(yte[mt], p, labels=range(5)).tolist())
res = {}
res["dodo_bits_per_token"] = float(np.mean([feat[k]["nll"] for k in sel(ds=["dodo"])])); res["dodh_bits_per_token"] = float(np.mean([feat[k]["nll"] for k in sel(ds=["dodh"])]))
res["dodo_F-M_from_dodh+pn2018_F-M"] = probe(sel(ds=["dodh", "pn2018"], group=["F-M"], sp="train"), sel(ds=["dodo"], group=["F-M"]))
res["dodo_F-M_from_all_F-M"] = probe(sel(group=["F-M"], sp="train"), sel(ds=["dodo"], group=["F-M"]))
res["dodo_F3-F4_from_dodh_F-F"] = probe(sel(ds=["dodh"], group=["F-F"], sp="train"), sel(ds=["dodo"], group=["F-F"]))
res["dodo_F3-F4_from_all_frontal_bipolar"] = probe(sel(group=["F-F", "HB-bipolar", "Fp-F"], sp="train"), sel(ds=["dodo"], group=["F-F"]))
res["dodo_C-M_from_all_C-M"] = probe(sel(group=["C-M"], sp="train"), sel(ds=["dodo"], group=["C-M"]))
res["heldout_F-M_from_all_F-M_by_dataset"] = {d: probe(sel(group=["F-M"], sp="train"), sel(ds=[d], group=["F-M"], sp="heldout")) for d in ["pn2018", "hmc", "dodh"]}
json.dump(res, open(f"{OUT}/analysis_dodo.json", "w"), indent=1)
for k, v in res.items():
    if isinstance(v, dict) and "acc" in v: print(f"{k:45s} acc {v['acc']*100:5.1f}  mf1 {v['mf1']*100:5.1f}  n_test {v['n_test']}")
    elif isinstance(v, dict): print(k, {kk: (round(vv['acc']*100,1), round(vv['mf1']*100,1)) if vv else None for kk, vv in v.items()})
    else: print(k, round(v, 3))
