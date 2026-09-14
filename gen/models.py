"""Model definitions shared by the DOD-H prototype and the corpus-scale runs."""
import math, torch, torch.nn as nn, torch.nn.functional as F

# ------------------------------------------------------------------ VQ-VAE tokenizer (one code per 100 ms @100 Hz)
class Enc(nn.Module):
    def __init__(s, dim=64, w=1):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(1, 32 * w, 7, padding=3), nn.GELU(), nn.Conv1d(32 * w, 64 * w, 5, stride=2, padding=2), nn.GELU(),
                              nn.Conv1d(64 * w, 128 * w, 5, stride=5, padding=0), nn.GELU(), nn.Conv1d(128 * w, 128 * w, 3, padding=1), nn.GELU(), nn.Conv1d(128 * w, dim, 1))
    def forward(s, x): return s.net(x)
class Dec(nn.Module):
    def __init__(s, dim=64, w=1):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(dim, 128 * w, 3, padding=1), nn.GELU(), nn.ConvTranspose1d(128 * w, 64 * w, 5, stride=5), nn.GELU(),
                              nn.ConvTranspose1d(64 * w, 32 * w, 4, stride=2, padding=1), nn.GELU(), nn.Conv1d(32 * w, 32 * w, 7, padding=3), nn.GELU(), nn.Conv1d(32 * w, 1, 1))
    def forward(s, z): return s.net(z)
class VQ(nn.Module):
    def __init__(s, k=512, d=64, decay=0.99, eps=1e-5):
        super().__init__(); s.k, s.d, s.decay, s.eps = k, d, decay, eps
        e = torch.randn(k, d) * 0.1
        s.register_buffer("emb", e); s.register_buffer("ema_n", torch.ones(k)); s.register_buffer("ema_w", e.clone()); s.register_buffer("usage", torch.zeros(k))
    def forward(s, z):
        zf = z.permute(0, 2, 1).reshape(-1, s.d)
        dist = zf.pow(2).sum(1, keepdim=True) - 2 * zf @ s.emb.t() + s.emb.pow(2).sum(1)[None]
        idx = dist.argmin(1); q = s.emb[idx]
        if s.training:
            with torch.no_grad():
                oh = F.one_hot(idx, s.k).float(); n = oh.sum(0); w = oh.t() @ zf
                s.ema_n.mul_(s.decay).add_(n, alpha=1 - s.decay); s.ema_w.mul_(s.decay).add_(w, alpha=1 - s.decay)
                nn_ = (s.ema_n + s.eps) / (s.ema_n.sum() + s.k * s.eps) * s.ema_n.sum(); s.emb.copy_(s.ema_w / nn_[:, None])
                s.usage.mul_(0.99).add_(n / n.sum(), alpha=0.01)
                dead = s.usage < 1e-4
                if dead.any():
                    rnd = zf[torch.randint(0, len(zf), (int(dead.sum()),), device=zf.device)]
                    s.emb[dead] = rnd; s.ema_w[dead] = rnd; s.ema_n[dead] = 1.0; s.usage[dead] = 1.0 / s.k
        commit = F.mse_loss(zf, q.detach()); q = zf + (q - zf).detach()
        return q.reshape(z.shape[0], -1, s.d).permute(0, 2, 1), idx.reshape(z.shape[0], -1), commit
class VQVAE(nn.Module):
    def __init__(s, k=512, dim=64, w=1): super().__init__(); s.enc, s.vq, s.dec = Enc(dim, w), VQ(k, dim), Dec(dim, w)
    def forward(s, x): z = s.enc(x); q, idx, commit = s.vq(z); return s.dec(q), idx, commit
    @torch.no_grad()
    def encode(s, x): return s.vq(s.enc(x))[1]
    @torch.no_grad()
    def decode_tokens(s, tok): return s.dec(s.vq.emb[tok].permute(0, 2, 1))

def stft_loss(x, y):
    l = 0
    for n in (32, 64, 128):
        win = torch.hann_window(n, device=x.device)
        X = torch.stft(x.squeeze(1), n, hop_length=n // 4, window=win, return_complex=True).abs()
        Y = torch.stft(y.squeeze(1), n, hop_length=n // 4, window=win, return_complex=True).abs()
        l = l + F.l1_loss(torch.log1p(X), torch.log1p(Y))
    return l / 3

# ------------------------------------------------------------------ decoder-only transformer
class Block(nn.Module):
    def __init__(s, dm, h, drop=0.1):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(dm), nn.LayerNorm(dm)
        s.attn = nn.MultiheadAttention(dm, h, dropout=drop, batch_first=True); s.mlp = nn.Sequential(nn.Linear(dm, 4 * dm), nn.GELU(), nn.Linear(4 * dm, dm), nn.Dropout(drop))
    def forward(s, x, mask):
        h = s.ln1(x); x = x + s.attn(h, h, h, attn_mask=mask, need_weights=False, is_causal=True)[0]; return x + s.mlp(s.ln2(x))
class GPT(nn.Module):
    """vocab = K signal codes + n_tags tag tokens (dataset/channel tag is prepended as token 0 of each sequence)"""
    def __init__(s, k=512, n_tags=0, ctx=512, L=6, dm=256, h=4, drop=0.1):
        super().__init__(); s.k, s.ctx = k, ctx
        s.tok, s.pos = nn.Embedding(k + n_tags, dm), nn.Embedding(ctx, dm); s.blocks = nn.ModuleList([Block(dm, h, drop) for _ in range(L)]); s.ln = nn.LayerNorm(dm); s.head = nn.Linear(dm, k, bias=False)
        s.register_buffer("mask", torch.triu(torch.ones(ctx, ctx, dtype=torch.bool), 1))
    def forward(s, idx, return_hidden=False):
        T = idx.shape[1]; x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device)); m = s.mask[:T, :T]
        for b in s.blocks: x = b(x, m)
        h = s.ln(x); return (s.head(h), h) if return_hidden else s.head(h)
