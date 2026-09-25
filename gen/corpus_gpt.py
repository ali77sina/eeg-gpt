"""Corpus-scale next-token pretraining. Each training sequence = [tag] + CTX tokens from one stream,
where tag = dataset__channel id (so the model knows the derivation). Held-out subjects never seen.
Outputs /workspace/gen/gpt.pt, gpt_hist.json, tags.json
"""
import os, sys, json, csv, time, math, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import GPT
torch.manual_seed(0); np.random.seed(0); random.seed(0)
OUT = os.environ.get("GEN_OUT", "/workspace/gen"); TOK = os.environ.get("TOK_DIR", OUT); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN = os.environ.get("RUN_NAME", "gpt"); EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 1000)); N_EVAL = int(os.environ.get("N_EVAL", 256))
K, CTX = 512, int(os.environ.get("CTX", 1024))
L, DM, H = int(os.environ.get("L", 8)), int(os.environ.get("DM", 384)), int(os.environ.get("H", 6))
BS, STEPS, LR = int(os.environ.get("BS", 64)), int(os.environ.get("GPT_STEPS", 40000)), 3e-4
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

streams = list(csv.DictReader(open(f"{TOK}/streams.csv")))
tags = sorted({f"{s['dataset']}__{s['chan']}" for s in streams}); tag_id = {t: K + i for i, t in enumerate(tags)}
json.dump(dict(tags=tags, K=K, CTX=CTX, L=L, DM=DM, H=H), open(f"{OUT}/tags.json", "w"))
def load_split(split):
    """pack all streams of a split into one int16 memmap (avoids thousands of open files); returns (tags, offsets, lengths, memmap)"""
    ss = [s for s in streams if s["split"] == split and int(s["n_tokens"]) >= CTX + 2]
    packed, idx = f"{TOK}/{split}_tokens.bin", f"{TOK}/{split}_index.npz"
    if not os.path.exists(idx):
        lens = np.array([len(np.load(s["path"], mmap_mode="r")) for s in ss]); offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
        mm = np.memmap(packed, dtype=np.int16, mode="w+", shape=(int(lens.sum()),))
        for s, o, l in zip(ss, offs, lens): mm[o:o + l] = np.load(s["path"])
        mm.flush(); del mm
        np.savez(idx, tags=np.array([tag_id[f"{s['dataset']}__{s['chan']}"] for s in ss]), offs=offs, lens=lens)
    d = np.load(idx); mm = np.memmap(packed, dtype=np.int16, mode="r")
    return d["tags"], d["offs"], d["lens"], mm
tr, ho = load_split("train"), load_split("heldout")
ntr = int(tr[2].sum()); log(f"{len(tr[0])} train streams ({ntr/1e9:.2f}B tokens), {len(ho[0])} held-out streams, {len(tags)} tags, ctx {CTX}, model L{L} d{DM}")
wtr = tr[2].astype(np.float64); wtr /= wtr.sum()
def batch(pool, w, bs):
    tags_, offs, lens, mm = pool; xs = []
    for i in np.random.choice(len(tags_), bs, p=w):
        s0 = offs[i] + np.random.randint(0, lens[i] - CTX - 1); seq = np.concatenate([[tags_[i]], np.asarray(mm[s0:s0 + CTX])]).astype(np.int64); xs.append(seq)
    x = torch.tensor(np.stack(xs), device=dev); return x[:, :-1], x[:, 1:]
who = ho[2].astype(np.float64); who /= who.sum()
# fixed held-out evaluation set: the same N_EVAL windows every time (seeded), so the curve is comparable across runs and low-noise
_rs = np.random.RandomState(123); _tags, _offs, _lens, _mm = ho
_ei = _rs.choice(len(_tags), N_EVAL, p=who); _es = [int(_offs[i] + _rs.randint(0, _lens[i] - CTX - 1)) for i in _ei]
EVAL_X = torch.tensor(np.stack([np.concatenate([[_tags[i]], np.asarray(_mm[s0:s0 + CTX])]).astype(np.int64) for i, s0 in zip(_ei, _es)]))   # [tag]+CTX tokens, like training batches
@torch.no_grad()
def eval_bits(m, pool=None, w=None, n=None):
    m.eval(); tot = 0; nb = 0
    for b in range(0, len(EVAL_X), 32):
        x = EVAL_X[b:b + 32].to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = m(x[:, :-1])
        tot += F.cross_entropy(lg.float().reshape(-1, K), x[:, 1:].reshape(-1)).item(); nb += 1
    return tot / nb / math.log(2)

if __name__ == "__main__":
    m = GPT(K, len(tags), CTX, L, DM, H).to(dev); n_params = sum(p.numel() for p in m.parameters()); log(f"params {n_params/1e6:.1f}M on {dev}")
    wb = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb; wb = wandb.init(project=os.environ.get("WANDB_PROJECT", "eeg-gpt-scaling"), name=RUN, id=RUN, resume="allow",
                                      config=dict(params=n_params, L=L, DM=DM, H=H, ctx=CTX, bs=BS, steps=STEPS, lr=LR, train_tokens=ntr, n_eval_windows=N_EVAL))
    opt = torch.optim.AdamW(m.parameters(), LR, betas=(0.9, 0.95), weight_decay=0.1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.03)
    start = 1
    UPLOAD_EVERY = int(os.environ.get("UPLOAD_EVERY", 5000)); ART = f"{RUN}-ckpt"
    def upload(step, final=False):   # push full checkpoint (model+opt+sched) and config to W&B as a versioned model artifact
        if not wb: return
        try: _upload(step, final)
        except Exception as e: log(f"W&B upload failed at step {step}: {e!r}")   # never let an upload kill training
    def _upload(step, final):
        import wandb
        a = wandb.Artifact(ART, type="model", metadata=dict(step=step, params=n_params, L=L, DM=DM, H=H, ctx=CTX, heldout_bits=hist[-1]["heldout_bits"] if hist else None))
        a.add_file(f"{OUT}/gpt_ckpt.pt"); a.add_file(f"{OUT}/gpt.pt"); a.add_file(f"{OUT}/tags.json")
        wb.log_artifact(a, aliases=["latest", f"step{step}"] + (["final"] if final else []))
    if not os.path.exists(f"{OUT}/gpt_ckpt.pt") and wb:   # pod disk lost: pull the latest checkpoint back from W&B
        try:
            d = wb.use_artifact(f"{ART}:latest").download(root=OUT); log(f"restored checkpoint from W&B artifact {ART}:latest")
        except Exception as e: log(f"no W&B checkpoint to restore ({type(e).__name__})")
    if os.path.exists(f"{OUT}/gpt_ckpt.pt"):
        ck = torch.load(f"{OUT}/gpt_ckpt.pt", map_location=dev); m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start = ck["step"] + 1; log(f"resumed at step {start}")
    t0 = time.time(); hist = json.load(open(f"{OUT}/gpt_hist.json")) if os.path.exists(f"{OUT}/gpt_hist.json") and start > 1 else []
    for step in range(start, STEPS + 1):
        m.train(); x, y = batch(tr, wtr, BS)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss = F.cross_entropy(m(x).float().reshape(-1, K), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % EVAL_EVERY == 0 or step == 1:
            hb = eval_bits(m); tb = loss.item() / math.log(2); hist.append(dict(step=step, train_bits=tb, heldout_bits=hb))
            log(f"step {step:6d} train {tb:.3f} | heldout {hb:.3f} bits/token | {(time.time()-t0)/(step-start+1):.3f}s/step")
            if wb: wb.log(dict(step=step, train_bits=tb, heldout_bits=hb, lr=sched.get_last_lr()[0], tokens_seen=step * BS * CTX, sec_per_step=(time.time()-t0)/(step-start+1)), step=step)
            torch.save(m.state_dict(), f"{OUT}/gpt.pt"); torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step), f"{OUT}/gpt_ckpt.pt")
            json.dump(hist, open(f"{OUT}/gpt_hist.json", "w"), indent=1)
            if step % UPLOAD_EVERY == 0: upload(step)
    upload(STEPS, final=True)
    if wb: wb.summary["final_heldout_bits"] = hist[-1]["heldout_bits"]; wb.finish()
    open(f"{OUT}/DONE", "w").write(str(hist[-1])); log("done")
