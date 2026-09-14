"""Whole-night evaluation of the corpus-scale generative model on held-out subjects of every dataset.
1. bits/token per dataset and per sleep stage; surprise at stage transitions
2. linear probes (train-split subjects -> held-out subjects): per dataset, per canonical channel group, and cross-dataset transfer
   (probe trained on PhysioNet frontal-mastoid, tested on DOD-H / HMC / Bitbrain), plus headband-specific probes
3. context ablation: pooled hidden state over last 1 / 5 / 10 / 30 s of each epoch
Outputs GEN_OUT/analysis.json
"""
import os, sys, json, csv, glob, time, math, re, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GPT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
torch.manual_seed(0); np.random.seed(0); random.seed(0)
W = os.environ.get("CORPUS", "/workspace/corpus"); OUT = os.environ.get("GEN_OUT", "/workspace/gen")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = json.load(open(f"{OUT}/tags.json")); K, CTX, tags = cfg["K"], cfg["CTX"], cfg["tags"]; tag_id = {t: K + i for i, t in enumerate(tags)}
EPT, HALF = 300, CTX // 2; POOL = {"1s": 10, "5s": 50, "10s": 100, "30s": 300}
MAX_TRAIN_RECS = int(os.environ.get("MAX_TRAIN_RECS", 60))       # probe-training recordings per dataset
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def canon(ch):
    """canonical channel group: <site>-<reference class>"""
    c = ch.upper().replace("A1", "M1").replace("A2", "M2")
    if c.startswith("HB"): return "HB-bipolar" if "-" in c else "HB-raw"
    site = re.match(r"^(FPZ|FP|F|CZ|C|PZ|P|O|T)", c); site = site.group(1) if site else "X"; site = {"FPZ": "Fp", "FP": "Fp", "CZ": "C", "PZ": "P"}.get(site, site)
    if "-" not in c: return f"{site}-raw"
    ref = c.split("-")[1]; refc = "M" if ref.startswith("M") else re.match(r"^(FP|F|C|P|O|T)", ref).group(1) if re.match(r"^(FP|F|C|P|O|T)", ref) else "X"
    return f"{site}-{refc}"

streams = list(csv.DictReader(open(f"{OUT}/streams.csv")))
for s in streams: s["group"] = canon(s["chan"])
rows = {}
for m in glob.glob(f"{W}/manifest_*.csv"):
    for r in csv.DictReader(open(m)): rows[(r["dataset"], r["rec_id"])] = r
def hyp_of(s): return np.load(f"{W}/{s['dataset']}/{s['rec_id']}.npz")["hyp"].astype(int)
def tok_path(s): return f"{OUT}/tokens/{s['dataset']}/{s['rec_id']}__{s['chan']}.npy"

gpt = GPT(K, len(tags), CTX, cfg["L"], cfg["DM"], cfg["H"]).to(dev); gpt.load_state_dict(torch.load(f"{OUT}/gpt.pt", map_location=dev)); gpt.eval()
DM = cfg["DM"]

@torch.no_grad()
def stream_features(s):
    t = np.load(tok_path(s)).astype(np.int64); hyp = hyp_of(s); n_ep = min(len(t) // EPT, len(hyp)); t = t[: n_ep * EPT]
    if n_ep < 10 or len(t) < CTX + 2: return None
    tag = next((tag_id[k] for k in (f"{s['dataset']}__{s['chan']}", f"dodh__{s['chan']}", f"pn2018__{s['chan']}") if k in tag_id), None)
    if tag is None: return None                       # derivation unknown to the model (e.g. DOD-O FP1-F3): skip
    cache = f"{OUT}/feat/{s['dataset']}__{s['rec_id']}__{s['chan']}.npz"
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True); return dict(nll_tok=d["nll_tok"], nll_ep=d["nll_ep"], feats=d["feats"].item(), y=d["y"])
    nll = np.zeros(len(t), np.float32); hid = np.zeros((len(t), DM), np.float16)
    starts = list(range(0, len(t) - CTX, HALF)); first = True
    for b in range(0, len(starts), 48):
        ss = starts[b:b + 48]; x = torch.tensor(np.stack([np.concatenate([[tag], t[s0:s0 + CTX]]) for s0 in ss]), device=dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits, h = gpt(x[:, :-1], return_hidden=True)
        l = F.cross_entropy(logits.float().reshape(-1, K), x[:, 1:].reshape(-1), reduction="none").reshape(len(ss), CTX) / math.log(2)
        for j, s0 in enumerate(ss):
            lo = 0 if (first and j == 0) else HALF
            nll[s0 + lo: s0 + CTX] = l[j, lo:].cpu().numpy(); hid[s0 + lo: s0 + CTX] = h[j, lo:].float().cpu().numpy().astype(np.float16)
        first = False
    H3 = hid.reshape(n_ep, EPT, DM)
    feats = {k: H3[:, EPT - v:, :].astype(np.float32).mean(1) for k, v in POOL.items()}
    out = dict(nll_tok=nll[HALF:], nll_ep=nll.reshape(n_ep, EPT).mean(1), feats=feats, y=hyp[:n_ep])
    os.makedirs(f"{OUT}/feat", exist_ok=True); np.savez(cache, nll_tok=out["nll_tok"], nll_ep=out["nll_ep"], feats=np.array(feats, dtype=object), y=out["y"])
    return out

# ---------------- choose streams: all held-out; probe-train = up to MAX_TRAIN_RECS recordings per dataset (all their channels)
ho = [s for s in streams if s["split"] == "heldout"]
tr_recs = {}
for s in streams:
    if s["split"] == "train": tr_recs.setdefault(s["dataset"], {}).setdefault(s["rec_id"], []).append(s)
tr = []
for d, recs in tr_recs.items():
    keys = sorted(recs); random.Random(0).shuffle(keys)
    for k in keys[:MAX_TRAIN_RECS]: tr += recs[k]
log(f"{len(ho)} held-out streams, {len(tr)} probe-train streams; groups: {sorted({s['group'] for s in streams})}")
feat = {}; t0 = time.time()
for i, s in enumerate(tr + ho):
    f = stream_features(s)
    if f is not None: feat[(s["dataset"], s["rec_id"], s["chan"])] = f
    if (i + 1) % 200 == 0: log(f"features {i+1}/{len(tr)+len(ho)} ({time.time()-t0:.0f}s)")
def sel(pool, **kw): return [s for s in pool if all(s[k] in v if isinstance(v, (list, set, tuple)) else s[k] == v for k, v in kw.items()) and (s["dataset"], s["rec_id"], s["chan"]) in feat]
def F_(ss, key): return np.concatenate([feat[(s["dataset"], s["rec_id"], s["chan"])]["feats"][key] for s in ss]), np.concatenate([feat[(s["dataset"], s["rec_id"], s["chan"])]["y"] for s in ss])
res = {"n_heldout_streams": len(ho), "n_probe_train_streams": len(tr), "groups": sorted({s["group"] for s in streams})}

# ---------------- 1. bits/token per dataset and per stage (held-out)
res["bits_per_dataset"] = {}; res["nll_by_stage"] = {}
for d in sorted({s["dataset"] for s in ho}):
    ss = sel(ho, dataset=d); nl = np.concatenate([feat[(s["dataset"], s["rec_id"], s["chan"])]["nll_tok"] for s in ss])
    res["bits_per_dataset"][d] = float(nl.mean())
    ne = np.concatenate([feat[(s["dataset"], s["rec_id"], s["chan"])]["nll_ep"] for s in ss]); y = np.concatenate([feat[(s["dataset"], s["rec_id"], s["chan"])]["y"] for s in ss])
    res["nll_by_stage"][d] = {st: float(ne[y == i].mean()) for i, st in enumerate(["W", "N1", "N2", "N3", "R"]) if (y == i).sum() > 50}
trans, stable = [], []
for s in ho:
    f = feat.get((s["dataset"], s["rec_id"], s["chan"]))
    if f is None: continue
    y, e = f["y"], f["nll_ep"]
    for i in range(1, len(y) - 1):
        if y[i] < 0: continue
        (trans if (y[i] != y[i - 1] or y[i] != y[i + 1]) else stable).append(e[i])
res["nll_transition_vs_stable"] = [float(np.mean(trans)), float(np.mean(stable))]

# ---------------- 2. probes
def probe(train_ss, test_ss, key="30s", C=0.5):
    if not train_ss or not test_ss: return None
    Xtr, ytr = F_(train_ss, key); Xte, yte = F_(test_ss, key); m = ytr >= 0; mt = yte >= 0
    if m.sum() < 500 or mt.sum() < 100 or len(np.unique(ytr[m])) < 5: return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=C)); clf.fit(Xtr[m], ytr[m]); p = clf.predict(Xte[mt])
    return dict(acc=float(accuracy_score(yte[mt], p)), mf1=float(f1_score(yte[mt], p, average="macro")), n_test_epochs=int(mt.sum()), cm=confusion_matrix(yte[mt], p, labels=range(5)).tolist())
res["probe_per_dataset_30s"] = {}
for d in sorted({s["dataset"] for s in ho}):
    r = probe(sel(tr, dataset=d), sel(ho, dataset=d)); res["probe_per_dataset_30s"][d] = r; log(f"probe {d}: {r and (round(r['acc'],3), round(r['mf1'],3))}")
res["probe_context_ablation_all"] = {k: (lambda r: r and dict(acc=r["acc"], mf1=r["mf1"]))(probe(sel(tr), sel(ho), key=k)) for k in POOL}
log(f"context ablation (all datasets pooled): {res['probe_context_ablation_all']}")
res["probe_per_group_30s"] = {}
for g in sorted({s["group"] for s in ho}):
    r = probe(sel(tr, group=g), sel(ho, group=g)); res["probe_per_group_30s"][g] = r and dict(acc=r["acc"], mf1=r["mf1"], n=r["n_test_epochs"])
# cross-dataset transfer: frontal-mastoid probe trained on PhysioNet only
fm_tr = sel(tr, dataset="pn2018", group="F-M"); res["transfer_from_pn2018_F-M"] = {}
for d in ["dodh", "hmc", "bitbrain", "cap", "sleepedf"]:
    for g in ["F-M", "F-F", "Fp-C", "HB-bipolar", "C-M"]:
        te = sel(ho, dataset=d, group=g)
        if te: r = probe(fm_tr, te); res["transfer_from_pn2018_F-M"][f"{d}/{g}"] = r and dict(acc=r["acc"], mf1=r["mf1"], n=r["n_test_epochs"])
# headband-specific: Bitbrain headband bipolar and DOD-H F3-F4 (the MorpheusNet derivation)
res["headband"] = {}
for name, d, g in [("bitbrain_HB_bipolar", "bitbrain", "HB-bipolar"), ("dodh_F3-F4", "dodh", "F-F"), ("bitbrain_F3-F4", "bitbrain", "F-F")]:
    r_in = probe(sel(tr, dataset=d, group=g), sel(ho, dataset=d, group=g))          # same-derivation probe
    r_all = probe(sel(tr, group=["F-F", "HB-bipolar", "Fp-F"]), sel(ho, dataset=d, group=g))   # probe trained on all frontal-bipolar streams across datasets
    res["headband"][name] = dict(same_dataset=r_in and dict(acc=r_in["acc"], mf1=r_in["mf1"]), all_frontal_bipolar=r_all and dict(acc=r_all["acc"], mf1=r_all["mf1"]))
    log(f"headband {name}: {res['headband'][name]}")
try:
    mn = json.load(open(os.path.expanduser("~/morph-net/results/F3_F4/results.json")))
    res["morpheusnet_dodh_F3F4_25fold_mean"] = float(np.mean([v["m_seq"]["acc"] for v in mn.values()]))
except Exception: pass
json.dump(res, open(f"{OUT}/analysis.json", "w"), indent=1); log(json.dumps({k: v for k, v in res.items() if k not in ("probe_per_dataset_30s",)}, indent=1)[:4000]); log("done")
