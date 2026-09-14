"""Generate EEG continuations from the corpus GPT (run 1) seeded with real held-out DOD-H epochs of each stage."""
import os, sys, json, glob, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from models import GPT, VQVAE
torch.manual_seed(0); np.random.seed(0)
# CKPT: folder with gpt.pt, tokenizer.pt, tags.json, split.json   DODH: folder of converted DOD-H npz (see corpus/convert.py)   OUTDIR: where figures go
R = os.environ.get("CKPT", "checkpoints"); DODH = os.environ.get("DODH", "corpus_out/dodh"); OUTDIR = os.environ.get("OUTDIR", "figures"); os.makedirs(OUTDIR, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
cfg = json.load(open(f"{R}/tags.json")); K, CTX, tags = cfg["K"], cfg["CTX"], cfg["tags"]; tag_id = {t: K + i for i, t in enumerate(tags)}
gpt = GPT(K, len(tags), CTX, cfg["L"], cfg["DM"], cfg["H"]).to(dev); gpt.load_state_dict(torch.load(f"{R}/gpt_15M_corpus.pt", map_location=dev)); gpt.eval()
vq = VQVAE().to(dev); vq.load_state_dict(torch.load(f"{R}/tokenizer_corpus.pt", map_location=dev)); vq.eval()
hold = [s for d, s in json.load(open(f"{R}/split.json"))["hold"] if d == "dodh"]
CHAN = "F3-F4"; tag = tag_id[f"dodh__{CHAN}"]
@torch.no_grad()
def encode(x):
    out = []
    for s0 in range(0, len(x), 30000): out.append(vq.encode(torch.tensor(x[s0:s0 + 30000], device=dev)[None, None])[0].cpu().numpy())
    return np.concatenate(out)
@torch.no_grad()
def decode(tok): return vq.decode_tokens(torch.tensor(tok, device=dev)[None])[0, 0].cpu().numpy()
@torch.no_grad()
def sample(seed, n, temp=1.0):
    x = torch.tensor(np.concatenate([[tag], seed]), device=dev)[None]
    for _ in range(n):
        ctx = x[:, -CTX:] if x.shape[1] <= CTX else torch.cat([x[:, :1], x[:, -(CTX - 1):]], 1)   # keep the tag token in front
        logits = gpt(ctx)[:, -1] / temp; x = torch.cat([x, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return x[0, len(seed) + 1:].cpu().numpy()
d = np.load(f"{DODH}/{hold[0]}.npz"); chans = [str(c) for c in d["chans"]]; x = d["x"][chans.index(CHAN)].astype(np.float32); hyp = d["hyp"]
toks = encode(x); EPT = 300; SEED, GEN = 256, 200      # 25.6 s seed, 20 s generation
out = {}
for st, name in enumerate(["W", "N1", "N2", "N3", "R"]):
    idx = [e for e in range(3, len(hyp) - 2) if all(hyp[e - 1:e + 2] == st)]        # stable stage, 3 epochs
    if not idx: continue
    e = idx[len(idx) // 2]; s0 = e * EPT
    seed = toks[s0 - SEED + EPT:s0 + EPT]                                            # seed ends at end of epoch e
    real = toks[s0 + EPT:s0 + EPT + GEN]; gen = sample(seed, GEN)
    out[name] = dict(seed=decode(seed[-100:]), real=decode(real), gen=decode(gen), raw_real=x[(s0 + EPT) * 10:(s0 + EPT + GEN) * 10])
    print(name, "epoch", e, "unique tokens real/gen", len(np.unique(real)), len(np.unique(gen)), flush=True)
# long free-running sample (60 s) from an N2 seed
e = [e for e in range(3, len(hyp) - 2) if all(hyp[e - 1:e + 2] == 2)][10]; s0 = e * EPT
long_gen = decode(sample(toks[s0 - SEED + EPT:s0 + EPT], 600)); long_real = x[(s0 + EPT) * 10:(s0 + EPT + 600) * 10]
np.savez(f"{OUTDIR}/samples.npz", **{f"{k}_{kk}": v for k, d_ in out.items() for kk, v in d_.items()}, long_gen=long_gen, long_real=long_real)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#52514e"
fig, axes = plt.subplots(len(out), 2, figsize=(14, 2.1 * len(out)), sharex=True, facecolor="#fcfcfb")
t = np.arange(1000) / 100
for i, (name, d_) in enumerate(out.items()):
    a = axes[i, 0]; a.plot(t, d_["raw_real"][:1000], lw=0.6, color=BLUE); a.set_ylabel(name, rotation=0, labelpad=14, fontsize=11, color=MUTED)
    b = axes[i, 1]; b.plot(t, d_["gen"][:1000], lw=0.6, color=ORANGE)
    for ax in (a, b): ax.set_ylim(-6, 6); ax.set_facecolor("#fcfcfb"); [ax.spines[k].set_visible(False) for k in ("top", "right")]; ax.tick_params(colors=MUTED, labelsize=8)
axes[0, 0].set_title("real EEG, the 10 s that actually followed the seed (DOD-H held-out subject, F3-F4)", fontsize=9, loc="left", color=MUTED)
axes[0, 1].set_title("model continuation from the same 25 s seed, first 10 s (15M GPT, temperature 1)", fontsize=9, loc="left", color=MUTED)
axes[-1, 0].set_xlabel("s", color=MUTED); axes[-1, 1].set_xlabel("s", color=MUTED); fig.tight_layout(); fig.savefig(f"{OUTDIR}/samples_by_stage.png", dpi=150)
# spectra + long sample
def psd(w): P = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2; f = np.fft.rfftfreq(len(w), 0.01); return f, P / P[(f >= 0.5) & (f < 30)].sum()
fig, axes = plt.subplots(2, 1, figsize=(14, 6), facecolor="#fcfcfb")
tl = np.arange(len(long_gen)) / 100; axes[0].plot(tl, long_real, lw=0.5, color=BLUE, label="real, 60 s"); axes[0].plot(tl, long_gen - 9, lw=0.5, color=ORANGE, label="generated, 60 s (offset)"); axes[0].legend(frameon=False, loc="upper right", fontsize=8); axes[0].set_yticks([]); axes[0].set_title("N2 seed: real vs free-running generation, 60 s", fontsize=9, loc="left", color=MUTED)
for name, d_ in out.items():
    f, Pr = psd(d_["raw_real"]); f, Pg = psd(d_["gen"]); axes[1].semilogy(f, Pr, lw=1.2, label=f"{name} real"); axes[1].semilogy(f, Pg, lw=1.2, ls="--", color=axes[1].lines[-1].get_color(), label=f"{name} generated")
axes[1].set_xlim(0.5, 30); axes[1].set_xlabel("Hz", color=MUTED); axes[1].set_ylabel("relative power", color=MUTED); axes[1].legend(frameon=False, ncol=5, fontsize=7); axes[1].set_title("power spectra of the 20 s continuations, solid = real, dashed = generated", fontsize=9, loc="left", color=MUTED)
for ax in axes: ax.set_facecolor("#fcfcfb"); [ax.spines[k].set_visible(False) for k in ("top", "right")]; ax.tick_params(colors=MUTED, labelsize=8)
fig.tight_layout(); fig.savefig(f"{OUTDIR}/samples_long_and_spectra.png", dpi=150); print("saved")
