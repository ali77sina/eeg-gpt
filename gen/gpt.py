"""Decoder-only transformer, next-token prediction on EEG tokens (100 ms each). Context 512 tokens = 51.2 s."""
import os, json, time, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
ROOT = os.path.expanduser("~/morph-net"); D = f"{ROOT}/data/gen"; OUT = f"{ROOT}/results/gen"
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
K, CTX, L, DM, H, BS, STEPS, LR = 512, 512, 6, 256, 4, 32, int(os.environ.get("GPT_STEPS", 6000)), 3e-4
split = json.load(open(f"{D}/split.json"))
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(DM), nn.LayerNorm(DM)
        s.attn = nn.MultiheadAttention(DM, H, dropout=0.1, batch_first=True); s.mlp = nn.Sequential(nn.Linear(DM, 4 * DM), nn.GELU(), nn.Linear(4 * DM, DM), nn.Dropout(0.1))
    def forward(s, x, mask):
        h = s.ln1(x); x = x + s.attn(h, h, h, attn_mask=mask, need_weights=False)[0]; return x + s.mlp(s.ln2(x))
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.tok, s.pos = nn.Embedding(K, DM), nn.Embedding(CTX, DM); s.blocks = nn.ModuleList([Block() for _ in range(L)]); s.ln = nn.LayerNorm(DM); s.head = nn.Linear(DM, K, bias=False)
        s.register_buffer("mask", torch.triu(torch.ones(CTX, CTX, dtype=torch.bool), 1))
    def forward(s, idx, return_hidden=False):
        T = idx.shape[1]; x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device)); m = s.mask[:T, :T]
        for b in s.blocks: x = b(x, m)
        h = s.ln(x); return (s.head(h), h) if return_hidden else s.head(h)
def load_tokens(recs): return {r: np.load(f"{D}/tokens/{r}.npy").astype(np.int64) for r in recs}
def batch(toks, bs=BS):
    recs = list(toks); xs = []
    for _ in range(bs):
        t = toks[recs[np.random.randint(len(recs))]]; s0 = np.random.randint(0, len(t) - CTX - 1); xs.append(t[s0:s0 + CTX + 1])
    x = torch.tensor(np.stack(xs), device=dev); return x[:, :-1], x[:, 1:]
@torch.no_grad()
def eval_nll(m, toks, n=64):
    m.eval(); tot = 0
    for _ in range(n // 16):
        x, y = batch(toks, 16); tot += F.cross_entropy(m(x).reshape(-1, K), y.reshape(-1)).item()
    return tot / (n // 16) / math.log(2)

if __name__ == "__main__":
    tr, ho = load_tokens(split["train"]), load_tokens(split["heldout"])
    m = GPT().to(dev); opt = torch.optim.AdamW(m.parameters(), LR, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)
    log(f"params {sum(p.numel() for p in m.parameters())/1e6:.2f}M  device {dev}  train tokens {sum(len(t) for t in tr.values())/1e6:.2f}M")
    t0 = time.time(); hist = []
    for step in range(1, STEPS + 1):
        m.train(); x, y = batch(tr); loss = F.cross_entropy(m(x).reshape(-1, K), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % 250 == 0 or step == 1:
            ho_bits = eval_nll(m, ho); hist.append(dict(step=step, train_bits=loss.item() / math.log(2), heldout_bits=ho_bits))
            log(f"step {step:5d} train {loss.item()/math.log(2):.3f} bits | heldout {ho_bits:.3f} bits/token | {(time.time()-t0)/step:.3f}s/step")
            torch.save(m.state_dict(), f"{OUT}/gpt.pt")
    json.dump(hist, open(f"{OUT}/gpt_hist.json", "w"), indent=1); log("done")
