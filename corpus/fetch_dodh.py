"""Stream DOD-H recordings from the Zenodo zip one at a time (HTTP range requests),
keep only the channels MorpheusNet needs (F3_M2, F3_F4) + consensus hypnogram, and
delete the ~900 MB h5 right after. Keeps disk footprint under ~2 GB total."""
import os, sys, time, traceback
import numpy as np, h5py
from remotezip import RemoteZip

URL = "https://zenodo.org/api/records/15900394/files/dodh.zip/content"
OUT = os.path.expanduser("~/morph-net/data/dodh_npz")
TMP = os.path.expanduser("~/morph-net/data/tmp")
CHANS = ["F3_M2", "F3_F4"]

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

with RemoteZip(URL) as z:
    members = sorted(i.filename for i in z.infolist()
                     if i.filename.startswith("dodh/") and i.filename.endswith(".h5"))
    log(f"{len(members)} recordings in zip")
    for k, name in enumerate(members):
        rid = os.path.basename(name)[:-3]
        out = os.path.join(OUT, rid + ".npz")
        if os.path.exists(out):
            log(f"[{k+1}/{len(members)}] {rid} already done"); continue
        for attempt in range(4):
            try:
                t = time.time()
                z.extract(name, TMP)
                p = os.path.join(TMP, name)
                log(f"[{k+1}/{len(members)}] {rid} downloaded {os.path.getsize(p)/1e6:.0f} MB in {time.time()-t:.0f}s")
                with h5py.File(p, "r") as f:
                    if k == 0:
                        f.visit(lambda n: log("   h5:", n, getattr(f[n], 'shape', '')))
                        log("   attrs:", dict(f.attrs))
                    hyp = np.array(f["hypnogram"]).astype(np.int8)
                    arrs = {c: np.array(f["signals/eeg"][c]).astype(np.float32) for c in CHANS}
                    fs = {c: dict(f["signals/eeg"][c].attrs) for c in CHANS}
                np.savez(out, hypnogram=hyp, **arrs)
                log(f"   hyp {hyp.shape} uniq {np.unique(hyp).tolist()} | " +
                    " | ".join(f"{c} {arrs[c].shape} attrs {fs[c]}" for c in CHANS))
                os.remove(p)
                break
            except Exception as e:
                log(f"   attempt {attempt} failed: {e!r}"); traceback.print_exc()
                time.sleep(10)
        else:
            log(f"GIVING UP on {rid}"); sys.exit(1)
log("ALL DONE")
