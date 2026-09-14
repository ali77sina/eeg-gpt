"""Corpus-scale VQ-VAE tokenizer: train on random 10 s crops drawn from a rotating RAM cache of recordings
(all channels, all datasets), then encode every (recording, channel) stream to int16 tokens.
Outputs: /workspace/gen/tokenizer.pt, /workspace/gen/tokens/<dataset>/<rec>__<chan>.npy, /workspace/gen/streams.csv
"""
import os, sys, json, csv, glob, time, math, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import VQVAE, stft_loss
torch.manual_seed(0); np.random.seed(0); random.seed(0)
W = os.environ.get("CORPUS", "/workspace/corpus"); OUT = os.environ.get("GEN_OUT", "/workspace/gen"); os.makedirs(f"{OUT}/tokens", exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
K, DIM, CROP, BS, STEPS, LR = 512, 64, 1000, 128, int(os.environ.get("TOK_STEPS", 20000)), 3e-4
CACHE_N, CACHE_REFRESH = 96, 400            # recordings held in RAM, steps between swapping a quarter of them
HOLD_FRAC = 0.1                             # held-out subjects per dataset
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

# ---------------- manifest + subject-level split
rows = []
for m in glob.glob(f"{W}/manifest_*.csv"): rows += list(csv.DictReader(open(m)))
EXCL = set(filter(None, os.environ.get("EXCLUDE_DS", "").split(",")))   # datasets kept entirely out of tokenizer/GPT training (e.g. "dodo" or "dodh,bitbrain")
rows = [r for r in rows if int(r["n_epochs"]) >= 60 and r["dataset"] not in EXCL]
subs = sorted({(r["dataset"], r["subject"]) for r in rows}); random.Random(0).shuffle(subs)
hold = set(); by_ds = {}
for d, s in subs: by_ds.setdefault(d, []).append((d, s))
for d, lst in by_ds.items(): hold |= set(lst[: max(1, int(len(lst) * HOLD_FRAC))])
for r in rows: r["split"] = "heldout" if (r["dataset"], r["subject"]) in hold else "train"
train_rows = [r for r in rows if r["split"] == "train"]; ho_rows = [r for r in rows if r["split"] == "heldout"]
log(f"{len(rows)} recordings, {len(train_rows)} train / {len(ho_rows)} held-out; per dataset: " + ", ".join(f"{d}={sum(1 for r in rows if r['dataset']==d)}" for d in sorted(by_ds)))
json.dump(dict(hold=sorted(list(hold))), open(f"{OUT}/split.json", "w"))

def rpath(r):                                         # manifest paths were written on the CPU pod; remap to this machine's corpus root
    return f"{W}/{r['dataset']}/{os.path.basename(r['path'])}"
def load(r):
    d = np.load(rpath(r)); return d["x"]            # (n_chan, T) float16
class Cache:
    def __init__(s, rows, n): s.rows, s.n = rows, n; s.items = [load(random.choice(rows)) for _ in range(n)]
    def refresh(s, k):
        for i in random.sample(range(s.n), k): s.items[i] = load(random.choice(s.rows))
    def batch(s, bs):
        xs = []
        for _ in range(bs):
            x = random.choice(s.items); c = np.random.randint(x.shape[0]); s0 = np.random.randint(0, x.shape[1] - CROP); xs.append(x[c, s0:s0 + CROP])
        return torch.tensor(np.stack(xs).astype(np.float32), device=dev)[:, None]

if __name__ == "__main__":
    ENCODE_ONLY = "--encode-only" in sys.argv and os.path.exists(f"{OUT}/tokenizer.pt")
    m = VQVAE(K, DIM).to(dev); log(f"tokenizer params {sum(p.numel() for p in m.parameters())/1e3:.0f}k on {dev}")
    if not ENCODE_ONLY:
        tr, ho = Cache(train_rows, CACHE_N), Cache(ho_rows, 24); log("caches loaded")
        opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        t0 = time.time(); hist = []
        for step in range(1, STEPS + 1):
            m.train(); x = tr.batch(BS); y, idx, commit = m(x)
            rec = F.mse_loss(y, x); loss = rec + 0.25 * commit + 0.5 * stft_loss(x, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if step % CACHE_REFRESH == 0: tr.refresh(CACHE_N // 4)
            if step % 500 == 0 or step == 1:
                m.eval()
                with torch.no_grad():
                    xv = ho.batch(256); yv, iv, _ = m(xv); rv = F.mse_loss(yv, xv).item(); snr = 10 * math.log10(xv.var().item() / rv)
                    p = torch.bincount(iv.flatten(), minlength=K).float(); p = p / p.sum(); perp = math.exp(-(p[p > 0] * p[p > 0].log()).sum().item())
                hist.append(dict(step=step, train_rec=rec.item(), heldout_rec=rv, snr_db=snr, code_perplexity=perp, codes_used=int((p > 0).sum())))
                log(f"step {step:6d} rec {rec.item():.4f} | heldout rec {rv:.4f} SNR {snr:5.2f} dB | perplexity {perp:6.1f} used {int((p>0).sum())}/{K} | {(time.time()-t0)/step:.3f}s/step")
                torch.save(m.state_dict(), f"{OUT}/tokenizer.pt"); json.dump(hist, open(f"{OUT}/tokenizer_hist.json", "w"), indent=1)
    else:
        m.load_state_dict(torch.load(f"{OUT}/tokenizer.pt", map_location=dev)); log("loaded tokenizer")
    # ---------------- encode every stream
    m.eval(); streams = []; t0 = time.time(); SEG = 30000
    for i, r in enumerate(rows):
        x = load(r); chans = [str(c) for c in np.load(rpath(r))["chans"]]; od = f"{OUT}/tokens/{r['dataset']}"; os.makedirs(od, exist_ok=True)
        for c, name in enumerate(chans):
            p = f"{od}/{r['rec_id']}__{name}.npy"
            if not os.path.exists(p):
                toks = []
                for s0 in range(0, x.shape[1], SEG):
                    seg = torch.tensor(x[c, s0:s0 + SEG].astype(np.float32), device=dev)[None, None]; toks.append(m.encode(seg)[0].cpu().numpy())
                np.save(p, np.concatenate(toks).astype(np.int16))
            streams.append(dict(dataset=r["dataset"], rec_id=r["rec_id"], subject=r["subject"], chan=name, split=r["split"], n_tokens=int(x.shape[1] // 10), path=p))
        if (i + 1) % 100 == 0: log(f"encoded {i+1}/{len(rows)} recordings ({time.time()-t0:.0f}s)")
    with open(f"{OUT}/streams.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(streams[0])); w.writeheader(); w.writerows(streams)
    log(f"done: {len(streams)} streams, {sum(s['n_tokens'] for s in streams)/1e9:.2f}B tokens")
