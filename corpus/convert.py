"""Unified sleep-EEG corpus builder.
Every recording -> /workspace/corpus/<dataset>/<rec_id>.npz with
  x        : float16 (n_chan, T)  EEG @100 Hz, 0.5-40 Hz bandpass, robust-scaled per channel (median/IQR), clipped +-10
  chans    : list of channel names (derivations as recorded, e.g. "F3-M2", "Fpz-Cz", "HB_1")
  hyp      : int8 (n_epochs,) 0=W 1=N1 2=N2 3=N3 4=R  -1=unscored  (30 s epochs, R&K S4 merged into N3)
  meta     : json string {dataset, subject, session, fs_orig}
Manifest: /workspace/corpus/manifest.csv  (dataset, rec_id, subject, n_epochs, n_chan, chans, path)
Usage: python convert.py <dataset> [--workers N]   dataset in {bitbrain, sleepedf, hmc, dodo, dodh, cap, pn2018, isruc}
"""
import os, sys, json, glob, csv, re, traceback, numpy as np
from multiprocessing import Pool
from scipy.signal import butter, lfilter, resample_poly
from fractions import Fraction

RAW = os.environ.get("RAW", "/workspace/raw"); OUT = os.environ.get("CORPUS", "/workspace/corpus"); FS = 100
os.makedirs(OUT, exist_ok=True)
STAGE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 3, "R": 4}

def preprocess(x, fs):
    """x (n_chan, T) float -> 100 Hz bandpassed robust-scaled float16"""
    x = np.asarray(x, np.float64)
    nyq = 0.5 * fs; b, a = butter(5, [0.5 / nyq, min(40, 0.45 * fs) / nyq], btype="band")
    x = lfilter(b, a, x, axis=1)
    if fs != FS:
        fr = Fraction(FS, int(round(fs))).limit_denominator(1000); x = resample_poly(x, fr.numerator, fr.denominator, axis=1)
    med = np.median(x, axis=1, keepdims=True); iqr = np.subtract(*np.percentile(x, [75, 25], axis=1)).reshape(-1, 1)
    iqr[iqr == 0] = 1.0
    return np.clip((x - med) / (iqr / 1.349), -10, 10).astype(np.float16)

def save(dataset, rec_id, x, chans, hyp, meta):
    n = min(x.shape[1] // (30 * FS), len(hyp)); x = x[:, : n * 30 * FS]; hyp = np.asarray(hyp[:n], np.int8)
    d = f"{OUT}/{dataset}"; os.makedirs(d, exist_ok=True)
    np.savez(f"{d}/{rec_id}.npz", x=x, chans=np.array(chans), hyp=hyp, meta=json.dumps(meta))
    return dict(dataset=dataset, rec_id=rec_id, subject=meta.get("subject", rec_id), n_epochs=int(n), n_chan=len(chans), chans="|".join(chans), path=f"{d}/{rec_id}.npz")

def read_edf(path, pick=None, rename=None):
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="error")
    names = raw.ch_names; sel = [c for c in names if (pick is None or c in pick)]
    raw.pick(sel); raw.load_data(); x = raw.get_data() * 1e6   # uV
    chans = [rename.get(c, c) if rename else c for c in raw.ch_names]
    return x, chans, raw.info["sfreq"], raw

def hyp_from_annotations(raw, n_epochs, mapping):
    """annotation-based hypnogram (Sleep-EDF / HMC style): mapping desc-substring -> stage code"""
    hyp = -np.ones(n_epochs, np.int8)
    for on, du, de in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        code = next((v for k, v in mapping.items() if k in de), None)
        if code is None: continue
        e0, e1 = int(round(on / 30)), int(round((on + du) / 30)); hyp[max(e0, 0):min(e1, n_epochs)] = code
    return hyp

# ------------------------------------------------------------------ datasets
def bitbrain(sub_dir):
    sub = os.path.basename(sub_dir); ev = f"{sub_dir}/eeg/{sub}_task-Sleep_acq-psg_events.tsv"
    rows = list(csv.DictReader(open(ev), delimiter="\t")); hyp = np.array([int(r["stage_hum"]) for r in rows]); hyp[(hyp < 0) | (hyp > 4)] = -1
    out = []
    xp, cp, fs, _ = read_edf(f"{sub_dir}/eeg/{sub}_task-Sleep_acq-psg_eeg.edf", pick=["PSG_F3", "PSG_F4", "PSG_C3", "PSG_C4", "PSG_O1", "PSG_O2"])
    i3, i4 = cp.index("PSG_F3"), cp.index("PSG_F4")
    xp = np.vstack([xp, (xp[i3] - xp[i4])[None]]); cp = [c.replace("PSG_", "") for c in cp] + ["F3-F4"]
    out.append(save("bitbrain", f"{sub}_psg", preprocess(xp, fs), cp, hyp, dict(dataset="bitbrain", subject=sub, fs_orig=fs, source="psg")))
    xh, ch, fs, _ = read_edf(f"{sub_dir}/eeg/{sub}_task-Sleep_acq-headband_eeg.edf", pick=["HB_1", "HB_2"])
    xh = np.vstack([xh, (xh[0] - xh[1])[None]]); ch = ch + ["HB_1-HB_2"]
    out.append(save("bitbrain", f"{sub}_headband", preprocess(xh, fs), ch, hyp, dict(dataset="bitbrain", subject=sub, fs_orig=fs, source="headband")))
    return out

def sleepedf(psg_path):
    import mne
    base = os.path.basename(psg_path)[:6]; hyp_path = glob.glob(os.path.join(os.path.dirname(psg_path), base + "*-Hypnogram.edf"))[0]
    x, chans, fs, raw = read_edf(psg_path, pick=["EEG Fpz-Cz", "EEG Pz-Oz"], rename={"EEG Fpz-Cz": "Fpz-Cz", "EEG Pz-Oz": "Pz-Oz"})
    ann = mne.read_annotations(hyp_path); raw.set_annotations(ann)
    n = int(x.shape[1] / fs // 30)
    hyp = hyp_from_annotations(raw, n, {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4})
    # trim long wake at both ends to 30 min around sleep (standard Sleep-EDF practice)
    s = np.where((hyp > 0) & (hyp < 5))[0]
    if len(s): lo, hi = max(0, s[0] - 60), min(n, s[-1] + 61); x = x[:, lo * 30 * int(fs): hi * 30 * int(fs)]; hyp = hyp[lo:hi]
    sub = base[:5]
    return [save("sleepedf", base, preprocess(x, fs), chans, hyp, dict(dataset="sleepedf", subject=sub, fs_orig=fs, cassette="SC" in base))]

def hmc(edf_path):
    import mne
    rid = os.path.basename(edf_path)[:-4]; sc = edf_path[:-4] + "_sleepscoring.edf"
    x, chans, fs, raw = read_edf(edf_path, pick=["EEG F4-M1", "EEG C4-M1", "EEG O2-M1", "EEG C3-M2"], rename={"EEG F4-M1": "F4-M1", "EEG C4-M1": "C4-M1", "EEG O2-M1": "O2-M1", "EEG C3-M2": "C3-M2"})
    raw.set_annotations(mne.read_annotations(sc)); n = int(x.shape[1] / fs // 30)
    hyp = hyp_from_annotations(raw, n, {"Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2, "Sleep stage N3": 3, "Sleep stage R": 4})
    return [save("hmc", rid, preprocess(x, fs), chans, hyp, dict(dataset="hmc", subject=rid, fs_orig=fs))]

def dreem(h5_path, dataset):
    import h5py
    rid = os.path.basename(h5_path)[:-3]
    with h5py.File(h5_path, "r") as f:
        hyp = np.array(f["hypnogram"]).astype(np.int8); eeg = f["signals/eeg"]
        names = ["F3_M2", "F4_M1", "C3_M2", "F3_F4", "FP1_F3", "F3_O1"]; names = [c for c in names if c in eeg]
        x = np.stack([np.array(eeg[c]) for c in names])
    return [save(dataset, rid, preprocess(x, 250), [c.replace("_", "-") for c in names], hyp, dict(dataset=dataset, subject=rid, fs_orig=250))]

def pn2018(rec_dir):
    """PhysioNet/CinC 2018: WFDB .mat/.hea (200 Hz), *-arousal.mat (v7.3) with data/sleep_stages/{wake,nonrem1,nonrem2,nonrem3,rem,undefined} per sample"""
    import wfdb, h5py
    rid = os.path.basename(rec_dir); rec = wfdb.rdrecord(f"{rec_dir}/{rid}")
    want = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1"]; idx = [rec.sig_name.index(c) for c in want if c in rec.sig_name]
    x = rec.p_signal[:, idx].T; fs = rec.fs; chans = [rec.sig_name[i] for i in idx]
    with h5py.File(f"{rec_dir}/{rid}-arousal.mat", "r") as f:
        ss = f["data"]["sleep_stages"]; per = {k: np.array(ss[k]).ravel() for k in ["wake", "nonrem1", "nonrem2", "nonrem3", "rem", "undefined"]}
    n = int(x.shape[1] / fs // 30); hyp = -np.ones(n, np.int8); L = int(30 * fs)
    for code, k in enumerate(["wake", "nonrem1", "nonrem2", "nonrem3", "rem"]):
        v = per[k][: n * L].reshape(n, L).mean(1) > 0.5; hyp[v] = code
    return [save("pn2018", rid, preprocess(x, fs), chans, hyp, dict(dataset="pn2018", subject=rid, fs_orig=fs))]

def cap(edf_path):
    """CAP sleep database: EDF + .txt with 'Sleep Stage' column (W,S1..S4,R) per 30 s epoch"""
    rid = os.path.basename(edf_path)[:-4]; txt = edf_path[:-4] + ".txt"
    if not os.path.exists(txt): return []
    import mne
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="error")
    cand = [c for c in raw.ch_names if re.match(r"^(Fp[12z]|F[34z78]|C[34z]|O[12]|A[12]|M[12])(-|$)", c, re.I) and "EOG" not in c.upper()]
    cand = [c for c in cand if not any(k in c.upper() for k in ("ECG", "EMG", "EOG", "SAO2", "HR"))][:6]
    if not cand: return []
    x, chans, fs, _ = read_edf(edf_path, pick=cand)
    lines = open(txt, errors="ignore").read().splitlines(); hdr = next(i for i, l in enumerate(lines) if l.startswith("Sleep Stage"))
    cols = lines[hdr].split("\t"); ist, itime, idur, iev = cols.index("Sleep Stage"), [i for i, c in enumerate(cols) if c.startswith("Time")][0], [i for i, c in enumerate(cols) if c.startswith("Duration")][0], cols.index("Event")
    n = int(x.shape[1] / fs // 30); hyp = -np.ones(n, np.int8)
    import datetime as dt
    t0 = raw.info["meas_date"]; e = 0
    m = {"W": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 3, "R": 4, "REM": 4}
    for l in lines[hdr + 1:]:
        p = l.split("\t")
        if len(p) <= max(ist, itime, idur, iev) or not p[iev].startswith("SLEEP"): continue
        try: hh, mm, ss = [int(v) for v in re.split("[:.]", p[itime].strip())[:3]]
        except Exception: continue
        t = t0.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        if t < t0: t += dt.timedelta(days=1)
        e0 = int(round((t - t0).total_seconds() / 30)); ne = max(1, int(round(float(p[idur]) / 30)))
        if p[ist] in m and 0 <= e0 < n: hyp[e0:min(n, e0 + ne)] = m[p[ist]]
    if (hyp >= 0).sum() < 100: return []
    return [save("cap", rid, preprocess(x, fs), chans, hyp, dict(dataset="cap", subject=rid, fs_orig=fs))]

def isruc(rec_dir):
    """ISRUC: <n>/<n>.rec (EDF) + <n>_1.txt (scorer 1, one stage per 30 s line: 0=W 1=N1 2=N2 3=N3 5=R)"""
    rid = os.path.basename(rec_dir); recs = glob.glob(f"{rec_dir}/*.rec") + glob.glob(f"{rec_dir}/*.edf")
    if not recs: return []
    rec = recs[0]; txt = sorted(glob.glob(f"{rec_dir}/*_1.txt"))
    if not txt: return []
    if rec.endswith(".rec"): os.symlink(rec, rec[:-4] + "_isruc.edf") if not os.path.exists(rec[:-4] + "_isruc.edf") else None; rec = rec[:-4] + "_isruc.edf"
    import mne
    raw = mne.io.read_raw_edf(rec, preload=False, verbose="error")
    cand = [c for c in raw.ch_names if re.match(r"^(F3|F4|C3|C4|O1|O2)-?(A1|A2|M1|M2)?$", c.strip(), re.I)][:6]
    x, chans, fs, _ = read_edf(rec, pick=cand)
    st = np.array([int(v) for v in open(txt[0]).read().split()]); m = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}
    hyp = np.array([m.get(v, -1) for v in st], np.int8)
    return [save("isruc", rid, preprocess(x, fs), [c.strip().replace("A1", "M1").replace("A2", "M2") for c in chans], hyp, dict(dataset="isruc", subject=rid, fs_orig=fs))]

def dodo(p): return dreem(p, "dodo")
def dodh(p): return dreem(p, "dodh")

def dodh_npz(p):
    """DOD-H from the locally extracted npz (F3_M2, F3_F4 @250 Hz + hypnogram)"""
    d = np.load(p); rid = os.path.basename(p)[:-4]; x = np.stack([d["F3_M2"], d["F3_F4"]])
    return [save("dodh", rid, preprocess(x, 250), ["F3-M2", "F3-F4"], d["hypnogram"].astype(np.int8), dict(dataset="dodh", subject=rid, fs_orig=250))]

JOBS = {
    "dodh_npz": (dodh_npz, lambda: sorted(glob.glob(f"{RAW}/dodh_npz/*.npz"))),
    "bitbrain": (bitbrain, lambda: sorted(glob.glob(f"{RAW}/bitbrain/sub-*"))),
    "sleepedf": (sleepedf, lambda: sorted(glob.glob(f"{RAW}/sleep-edfx/**/*-PSG.edf", recursive=True))),
    "hmc": (hmc, lambda: sorted(glob.glob(f"{RAW}/hmc-sleep-staging/**/recordings/SN*.edf", recursive=True)) if False else sorted(p for p in glob.glob(f"{RAW}/hmc-sleep-staging/**/SN*.edf", recursive=True) if "sleepscoring" not in p)),
    "dodo": (dodo, lambda: sorted(p for p in glob.glob(f"{RAW}/dodo/**/*.h5", recursive=True) if "__MACOSX" not in p and not os.path.basename(p).startswith("._"))),
    "dodh": (dodh, lambda: sorted(p for p in glob.glob(f"{RAW}/dodh/**/*.h5", recursive=True) if "__MACOSX" not in p and not os.path.basename(p).startswith("._"))),
    "pn2018": (pn2018, lambda: sorted(d for d in glob.glob(f"{RAW}/challenge-2018/tr*") if os.path.isdir(d))),
    "cap": (cap, lambda: sorted(glob.glob(f"{RAW}/capslpdb/**/*.edf", recursive=True))),
    "isruc": (isruc, lambda: sorted(d for d in glob.glob(f"{RAW}/isruc/*/*") if os.path.isdir(d)) or sorted(d for d in glob.glob(f"{RAW}/isruc/*") if os.path.isdir(d))),
}
def _run(args):
    fn, item = args
    try: return fn(item), None
    except Exception as e: return [], f"{item}: {e!r}\n{traceback.format_exc()[-600:]}"

if __name__ == "__main__":
    ds = sys.argv[1]; workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    fn, lister = JOBS[ds]; items = lister()
    if "--limit" in sys.argv: items = items[: int(sys.argv[sys.argv.index("--limit") + 1])]
    print(f"{ds}: {len(items)} items", flush=True)
    rows, errs = [], []
    with Pool(workers) as pool:
        for i, (r, e) in enumerate(pool.imap_unordered(_run, [(fn, it) for it in items])):
            rows += r
            if e: errs.append(e)
            if (i + 1) % 20 == 0 or i + 1 == len(items): print(f"  {i+1}/{len(items)} done, {len(rows)} recordings, {len(errs)} errors", flush=True)
    with open(f"{OUT}/manifest_{ds}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "rec_id", "subject", "n_epochs", "n_chan", "chans", "path"]); w.writeheader(); w.writerows(rows)
    open(f"{OUT}/errors_{ds}.log", "w").write("\n".join(errs))
    tot = sum(r["n_epochs"] for r in rows); print(f"{ds}: {len(rows)} recordings, {tot} epochs = {tot/120:.0f} h, {len(errs)} errors", flush=True)
