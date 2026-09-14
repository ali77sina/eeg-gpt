"""Whole-night evaluation of the second-scale generative model on the 5 held-out nights."""
import os, sys, json, time, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from gpt import GPT, K, CTX, DM, dev, split, D, OUT, log
from tokenizer import VQVAE, sig, hyp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
torch.manual_seed(0); np.random.seed(0)
EPT = 300                                             # tokens per 30 s epoch
POOL = {"1s": 10, "5s": 50, "10s": 100, "30s": 300}    # pool hidden states over the last k tokens of each epoch
gpt = GPT().to(dev); gpt.load_state_dict(torch.load(f"{OUT}/gpt.pt", map_location=dev)); gpt.eval()
vq = VQVAE().to(dev); vq.load_state_dict(torch.load(f"{OUT}/tokenizer.pt", map_location=dev)); vq.eval()

@torch.no_grad()
def night_features(r):
    """teacher-forced pass, stride CTX/2, keep the second half of each window (>=256 tokens of context)"""
    t = np.load(f"{D}/tokens/{r}.npy").astype(np.int64); n_ep = min(len(t) // EPT, len(hyp[r])); t = t[: n_ep * EPT]
    nll = np.zeros(len(t), np.float32); hid = np.zeros((len(t), DM), np.float16); half = CTX // 2
    starts = list(range(0, len(t) - CTX, half)); first = True
    for b in range(0, len(starts), 32):
        ss = starts[b:b + 32]; x = torch.tensor(np.stack([t[s:s + CTX + 1] for s in ss]), device=dev)
        logits, h = gpt(x[:, :-1], return_hidden=True); l = F.cross_entropy(logits.reshape(-1, K), x[:, 1:].reshape(-1), reduction="none").reshape(len(ss), CTX) / math.log(2)
        for j, s in enumerate(ss):
            lo = 0 if (first and j == 0) else half
            nll[s + 1 + lo: s + 1 + CTX] = l[j, lo:].cpu().numpy(); hid[s + 1 + lo: s + 1 + CTX] = h[j, lo:].cpu().numpy().astype(np.float16)
        first = False
    feats = {k: hid.reshape(n_ep, EPT, DM)[:, EPT - v:, :].astype(np.float32).mean(1) for k, v in POOL.items()}
    bow = np.stack([np.bincount(t[e * EPT:(e + 1) * EPT], minlength=K) for e in range(n_ep)]).astype(np.float32) / EPT
    return dict(nll_tok=nll, nll_ep=nll.reshape(n_ep, EPT).mean(1), feats=feats, bow=bow, y=hyp[r][:n_ep].astype(int))

t0 = time.time(); N = {}
for r in split["train"] + split["heldout"]:
    N[r] = night_features(r); log(f"{r} {'train' if r in split['train'] else 'heldout'} epochs {len(N[r]['y'])} mean NLL {N[r]['nll_tok'][256:].mean():.3f} bits  ({time.time()-t0:.0f}s)")
res = {}
# ---------- 1. surprise (NLL) vs sleep structure on held-out nights
ho = split["heldout"]; y_all = np.concatenate([N[r]["y"] for r in ho]); nll_all = np.concatenate([N[r]["nll_ep"] for r in ho])
res["heldout_bits_per_token"] = float(np.concatenate([N[r]["nll_tok"][256:] for r in ho]).mean())
res["nll_by_stage_bits"] = {s: float(nll_all[y_all == i].mean()) for i, s in enumerate(["W", "N1", "N2", "N3", "R"]) if (y_all == i).any()}
trans, stable = [], []
for r in ho:
    y, e = N[r]["y"], N[r]["nll_ep"]
    for i in range(1, len(y) - 1):
        if y[i] < 0: continue
        (trans if (y[i] != y[i - 1] or y[i] != y[i + 1]) else stable).append(e[i])
res["nll_transition_epochs"] = float(np.mean(trans)); res["nll_stable_epochs"] = float(np.mean(stable)); res["n_transition_epochs"] = len(trans)
# within-night: correlation of epoch NLL with time of night
res["corr_nll_vs_time"] = float(np.mean([np.corrcoef(np.arange(len(N[r]["nll_ep"])), N[r]["nll_ep"])[0, 1] for r in ho]))
# ---------- 2. linear probes: train nights -> held-out nights (subject-independent)
def probe(key, getter):
    Xtr = np.concatenate([getter(N[r]) for r in split["train"]]); ytr = np.concatenate([N[r]["y"] for r in split["train"]])
    Xho = np.concatenate([getter(N[r]) for r in ho]); yho = y_all
    m = ytr >= 0; clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)); clf.fit(Xtr[m], ytr[m])
    p = clf.predict(Xho); per = {r: float(accuracy_score(N[r]["y"], clf.predict(getter(N[r])))) for r in ho}
    return dict(acc=float(accuracy_score(yho, p)), mf1=float(f1_score(yho, p, average="macro")), per_night_acc=per, cm=confusion_matrix(yho, p, labels=range(5)).tolist())
res["probe_bag_of_tokens_30s"] = probe("bow", lambda n: n["bow"])
for k in POOL: res[f"probe_hidden_{k}"] = probe(k, lambda n, k=k: n["feats"][k]); log(f"probe hidden {k}: acc {res[f'probe_hidden_{k}']['acc']:.3f} mf1 {res[f'probe_hidden_{k}']['mf1']:.3f}")
log(f"probe bag-of-tokens: acc {res['probe_bag_of_tokens_30s']['acc']:.3f}")
# MorpheusNet reference on the same 5 subjects (folds 20-24 of the F3-F4 run)
try:
    mn = json.load(open(f"{OUT}/../F3_F4/results.json")); res["morpheusnet_same_subjects"] = {r: next(v["m_seq"]["acc"] for v in mn.values() if v["test_record"] == r) for r in ho}
    res["morpheusnet_same_subjects_mean"] = float(np.mean(list(res["morpheusnet_same_subjects"].values())))
except Exception as e: res["morpheusnet_same_subjects"] = repr(e)
# ---------- 3. generation: seed 25.6 s of real N2 from a held-out night, sample 60 s, compare band power to the real continuation
@torch.no_grad()
def sample(seed, n, temp=1.0):
    x = torch.tensor(seed, device=dev)[None]
    for _ in range(n):
        logits = gpt(x[:, -CTX:])[:, -1] / temp; x = torch.cat([x, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return x[0, len(seed):].cpu().numpy()
r = ho[0]; t = np.load(f"{D}/tokens/{r}.npy").astype(np.int64); y = N[r]["y"]
e0 = next(i for i in range(20, len(y) - 4) if all(y[i:i + 4] == 2)); s0 = e0 * EPT
seed = t[s0:s0 + 256]; real = t[s0 + 256:s0 + 256 + 600]; gen = sample(seed, 600)
def decode(tok): return vq.dec(vq.vq.emb[torch.tensor(tok, device=dev)].t()[None])[0, 0].cpu().numpy()
with torch.no_grad(): wr, wg = decode(real), decode(gen)
def bands(w):
    P = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2; f = np.fft.rfftfreq(len(w), 0.01); tot = P[(f >= 0.5) & (f < 30)].sum()
    return {b: float(P[(f >= lo) & (f < hi)].sum() / tot) for b, (lo, hi) in dict(delta=(0.5, 4), theta=(4, 8), alpha=(8, 12), sigma=(12, 16), beta=(16, 30)).items()}
res["gen_relative_band_power"] = dict(real_continuation=bands(wr), generated=bands(wg))
res["gen_token_repeat_rate"] = float((gen[1:] == gen[:-1]).mean()); res["real_token_repeat_rate"] = float((real[1:] == real[:-1]).mean())
res["gen_unique_tokens"] = int(len(np.unique(gen))); res["real_unique_tokens"] = int(len(np.unique(real)))
json.dump(res, open(f"{OUT}/analysis.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(12, 7)); tt = np.arange(1000) / 100
ax[0].plot(tt, wr[:1000], lw=0.7); ax[0].set_title("real continuation (N2, held-out night), 10 s"); ax[1].plot(tt, wg[:1000], lw=0.7, color="C1"); ax[1].set_title("generated continuation from the same 25.6 s seed, 10 s")
r0 = ho[0]; ax[2].plot(N[r0]["nll_ep"], lw=0.6, label="epoch NLL (bits/token)"); ax2 = ax[2].twinx(); ax2.plot(N[r0]["y"], color="k", lw=0.8, alpha=0.5, label="hypnogram"); ax2.set_yticks(range(5)); ax2.set_yticklabels(["W", "N1", "N2", "N3", "R"]); ax[2].set_title(f"whole night {r0}: model surprise vs hypnogram"); ax[2].set_xlabel("30 s epoch")
fig.tight_layout(); fig.savefig(f"{OUT}/analysis.png", dpi=150)
log(json.dumps(res, indent=1)); log("done")
