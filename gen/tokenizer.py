"""VQ-VAE tokenizer for single-channel EEG @100 Hz: one code per 100 ms (stride 10), codebook 512 x 64, EMA updates."""
import os, json, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
ROOT = os.path.expanduser("~/morph-net"); D = f"{ROOT}/data/gen"; OUT = f"{ROOT}/results/gen"; os.makedirs(OUT, exist_ok=True); os.makedirs(f"{D}/tokens", exist_ok=True)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
K, DIM, STRIDE, CROP, BS, STEPS, LR = 512, 64, 10, 1000, 64, 8000, 3e-4
split = json.load(open(f"{D}/split.json"))
sig = {r: np.load(f"{D}/{r}.npz")["x"] for r in split["train"] + split["heldout"]}
hyp = {r: np.load(f"{D}/{r}.npz")["hyp"] for r in sig}
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class Enc(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(1, 32, 7, padding=3), nn.GELU(), nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
                              nn.Conv1d(64, 128, 5, stride=5, padding=0), nn.GELU(), nn.Conv1d(128, 128, 3, padding=1), nn.GELU(), nn.Conv1d(128, DIM, 1))
    def forward(s, x): return s.net(x)                       # (B,1,T) -> (B,DIM,T/10)
class Dec(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(DIM, 128, 3, padding=1), nn.GELU(), nn.ConvTranspose1d(128, 64, 5, stride=5), nn.GELU(),
                              nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1), nn.GELU(), nn.Conv1d(32, 32, 7, padding=3), nn.GELU(), nn.Conv1d(32, 1, 1))
    def forward(s, z): return s.net(z)
class VQ(nn.Module):
    def __init__(s, k=K, d=DIM, decay=0.99, eps=1e-5):
        super().__init__(); s.k, s.d, s.decay, s.eps = k, d, decay, eps
        e = torch.randn(k, d) * 0.1
        s.register_buffer("emb", e); s.register_buffer("ema_n", torch.ones(k)); s.register_buffer("ema_w", e.clone()); s.register_buffer("usage", torch.zeros(k))
    def forward(s, z):                                        # z (B,D,T)
        zf = z.permute(0, 2, 1).reshape(-1, s.d)
        dist = zf.pow(2).sum(1, keepdim=True) - 2 * zf @ s.emb.t() + s.emb.pow(2).sum(1)[None]
        idx = dist.argmin(1); q = s.emb[idx]
        if s.training:
            with torch.no_grad():
                oh = F.one_hot(idx, s.k).float(); n = oh.sum(0); w = oh.t() @ zf
                s.ema_n.mul_(s.decay).add_(n, alpha=1 - s.decay); s.ema_w.mul_(s.decay).add_(w, alpha=1 - s.decay)
                nn_ = (s.ema_n + s.eps) / (s.ema_n.sum() + s.k * s.eps) * s.ema_n.sum(); s.emb.copy_(s.ema_w / nn_[:, None])
                s.usage.mul_(0.99).add_(n / n.sum(), alpha=0.01)
                dead = s.usage < 1e-4                          # restart dead codes from random encoder outputs
                if dead.any():
                    rnd = zf[torch.randint(0, len(zf), (int(dead.sum()),), device=zf.device)]
                    s.emb[dead] = rnd; s.ema_w[dead] = rnd; s.ema_n[dead] = 1.0; s.usage[dead] = 1.0 / s.k
        commit = F.mse_loss(zf, q.detach()); q = zf + (q - zf).detach()
        return q.reshape(z.shape[0], -1, s.d).permute(0, 2, 1), idx.reshape(z.shape[0], -1), commit
class VQVAE(nn.Module):
    def __init__(s): super().__init__(); s.enc, s.vq, s.dec = Enc(), VQ(), Dec()
    def forward(s, x): z = s.enc(x); q, idx, commit = s.vq(z); return s.dec(q), idx, commit
    @torch.no_grad()
    def encode(s, x): return s.vq(s.enc(x))[1]

def stft_loss(x, y):
    l = 0
    for n in (32, 64, 128):
        X = torch.stft(x.squeeze(1), n, hop_length=n // 4, window=torch.hann_window(n, device=x.device), return_complex=True).abs()
        Y = torch.stft(y.squeeze(1), n, hop_length=n // 4, window=torch.hann_window(n, device=x.device), return_complex=True).abs()
        l = l + F.l1_loss(torch.log1p(X), torch.log1p(Y))
    return l / 3
def batch(recs, bs=BS, crop=CROP):
    xs = []
    for _ in range(bs):
        x = sig[recs[np.random.randint(len(recs))]]; s0 = np.random.randint(0, len(x) - crop); xs.append(x[s0:s0 + crop])
    return torch.tensor(np.stack(xs), device=dev)[:, None]

if __name__ == "__main__":
    import sys
    ENCODE_ONLY = "--encode-only" in sys.argv and os.path.exists(f"{OUT}/tokenizer.pt")
    m = VQVAE().to(dev); opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    log(f"params {sum(p.numel() for p in m.parameters())/1e3:.0f}k  device {dev}")
    use_stft = True
    try: stft_loss(torch.zeros(2, 1, 100, device=dev), torch.zeros(2, 1, 100, device=dev))
    except Exception as e: use_stft = False; log("stft loss disabled:", repr(e))
    t = time.time(); hist = []
    if ENCODE_ONLY: m.load_state_dict(torch.load(f"{OUT}/tokenizer.pt", map_location=dev)); hist = json.load(open(f"{OUT}/tokenizer_hist.json")); log("loaded trained tokenizer, encoding only")
    for step in range(1, 0 if ENCODE_ONLY else STEPS + 1):
        m.train(); x = batch(split["train"]); y, idx, commit = m(x)
        rec = F.mse_loss(y, x); loss = rec + 0.25 * commit + (0.5 * stft_loss(x, y) if use_stft else 0)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == 1:
            m.eval()
            with torch.no_grad():
                xv = batch(split["heldout"], 64); yv, iv, _ = m(xv); rv = F.mse_loss(yv, xv).item(); snr = 10 * math.log10(xv.var().item() / rv)
                p = torch.bincount(iv.flatten(), minlength=K).float(); p = p / p.sum(); perp = math.exp(-(p[p > 0] * p[p > 0].log()).sum().item())
            hist.append(dict(step=step, train_rec=rec.item(), heldout_rec=rv, snr_db=snr, code_perplexity=perp, codes_used=int((p > 0).sum())))
            log(f"step {step:5d} rec {rec.item():.4f} | heldout rec {rv:.4f} SNR {snr:5.2f} dB | code perplexity {perp:6.1f} used {int((p>0).sum())}/{K} | {(time.time()-t)/step:.3f}s/step")
    torch.save(m.state_dict(), f"{OUT}/tokenizer.pt"); json.dump(hist, open(f"{OUT}/tokenizer_hist.json", "w"), indent=1)
    # ---- encode all nights
    m.eval(); stats = {}
    for r, x in sig.items():
        toks = []
        for s0 in range(0, len(x), 30000):                    # 5 min segments (MPS conv limit)
            seg = torch.tensor(x[s0:s0 + 30000], device=dev)[None, None]; toks.append(m.encode(seg)[0].cpu().numpy())
        toks = np.concatenate(toks).astype(np.int16); np.save(f"{D}/tokens/{r}.npy", toks)
    # ---- held-out evaluation: recon, token-stage mutual information, unigram/bigram entropies
    m.eval(); res = dict(history=hist[-1], K=K, stride_ms=STRIDE * 10)
    with torch.no_grad():
        xv = torch.cat([torch.tensor(sig[r][100000 + i * 30000:130000 + i * 30000], device=dev)[None, None] for r in split["heldout"] for i in range(5)]); yv, _, _ = m(xv)
        res["heldout_recon_mse"] = F.mse_loss(yv, xv).item(); res["heldout_snr_db"] = 10 * math.log10(xv.var().item() / res["heldout_recon_mse"])
        # band-wise power preservation (delta/theta/alpha/sigma/beta) on held-out
        X = torch.stft(xv.squeeze(1), 200, hop_length=100, window=torch.hann_window(200, device=dev), return_complex=True).abs().pow(2).mean((0, 2)).cpu().numpy()
        Y = torch.stft(yv.squeeze(1), 200, hop_length=100, window=torch.hann_window(200, device=dev), return_complex=True).abs().pow(2).mean((0, 2)).cpu().numpy()
        f = np.arange(101) * 0.5; bands = dict(delta=(0.5, 4), theta=(4, 8), alpha=(8, 12), sigma=(12, 16), beta=(16, 30))
        res["band_power_ratio_recon_over_orig"] = {b: float(Y[(f >= lo) & (f < hi)].sum() / X[(f >= lo) & (f < hi)].sum()) for b, (lo, hi) in bands.items()}
    # token/stage MI on held-out nights (tokens per 30 s epoch = 300)
    joint = np.zeros((5, K))
    for r in split["heldout"]:
        t = np.load(f"{D}/tokens/{r}.npy"); h = hyp[r]; n = min(len(t) // 300, len(h))
        for e in range(n):
            if h[e] >= 0: joint[h[e]] += np.bincount(t[e * 300:(e + 1) * 300], minlength=K)
    p = joint / joint.sum(); ps, pt = p.sum(1, keepdims=True), p.sum(0, keepdims=True); nz = p > 0
    res["token_stage_MI_bits"] = float((p[nz] * np.log2(p[nz] / (ps @ pt)[nz])).sum()); res["stage_entropy_bits"] = float(-(ps[ps > 0] * np.log2(ps[ps > 0])).sum())
    tr = np.concatenate([np.load(f"{D}/tokens/{r}.npy").astype(np.int64) for r in split["train"]]); ho = np.concatenate([np.load(f"{D}/tokens/{r}.npy").astype(np.int64) for r in split["heldout"]])
    uni = np.bincount(tr, minlength=K) + 1; uni = uni / uni.sum(); res["heldout_unigram_bits"] = float(-np.log2(uni[ho]).mean())
    big = np.ones((K, K)); np.add.at(big, (tr[:-1], tr[1:]), 1); big = big / big.sum(1, keepdims=True); res["heldout_bigram_bits"] = float(-np.log2(big[ho[:-1], ho[1:]]).mean())
    res["n_train_tokens"] = int(len(tr)); res["n_heldout_tokens"] = int(len(ho))
    json.dump(res, open(f"{OUT}/tokenizer_eval.json", "w"), indent=1); log(json.dumps(res, indent=1))
    # figure: original vs reconstruction, 5 s of a held-out night
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = split["heldout"][0]; x = sig[r][2000000:2000500]
    with torch.no_grad(): y = m(torch.tensor(x, device=dev)[None, None])[0][0, 0].cpu().numpy()
    fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True); tt = np.arange(500) / 100
    ax[0].plot(tt, x, lw=0.8, label="original"); ax[0].plot(tt, y, lw=0.8, label="VQ-VAE reconstruction"); ax[0].legend(loc="upper right"); ax[0].set_ylabel("z (robust)")
    ax[1].plot(tt, x - y, lw=0.8, color="k"); ax[1].set_ylabel("residual"); ax[1].set_xlabel("s"); fig.tight_layout(); fig.savefig(f"{OUT}/tokenizer_recon.png", dpi=150)
    log("done")
