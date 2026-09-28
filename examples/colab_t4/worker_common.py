"""Shared helpers for the Colab workers (each case runs in a fresh process)."""

import gc
import json
import os
import resource
import sys
import threading
import time
import traceback

import torch

DEVICE = os.environ.get("MP_DEVICE") or (
    "cuda" if torch.cuda.is_available()
    else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else "cpu"
)
RESULT: dict = {"status": None, "device": DEVICE, "argv": sys.argv[1:]}
_T0 = time.time()


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


class DeviceMonitor:
    """Peak device memory in use (cudaMemGetInfo: includes the CUDA context and other processes'
    use, i.e. what nvidia-smi shows) sampled every 20 ms, plus the allocator's own peaks."""

    def __init__(self):
        self.peak_used = 0
        self._stop = threading.Event()
        if DEVICE == "cuda":
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            self.peak_used = max(self.peak_used, total - free)
            self._stop.wait(0.02)

    def reset(self):
        self.peak_used = 0
        if DEVICE == "cuda":
            sync()
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out = {"host_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024}
        if DEVICE == "cuda":
            sync()
            free, total = torch.cuda.mem_get_info()
            out.update(allocated_bytes=torch.cuda.memory_allocated(),
                       reserved_bytes=torch.cuda.memory_reserved(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       device_used_bytes=total - free,
                       peak_device_used_bytes=max(self.peak_used, total - free))
        elif DEVICE == "mps":
            out.update(allocated_bytes=torch.mps.current_allocated_memory(),
                       driver_bytes=torch.mps.driver_allocated_memory())
        return out


MON = DeviceMonitor()


def is_oom(e):
    text = f"{type(e).__name__} {e}".lower()
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in text


def cleanup():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


def memopro_report():
    try:
        import memopro

        return [{"technique": e.technique, "action": e.action, "detail": e.detail}
                for e in memopro.report().entries]
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]


def finish(status=None, **fields):
    RESULT.update(fields)
    if status:
        RESULT["status"] = status
    RESULT.setdefault("status", "ok")
    if RESULT["status"] is None:
        RESULT["status"] = "ok"
    RESULT["worker_s"] = round(time.time() - _T0, 1)
    RESULT["memory_end"] = MON.snapshot()
    RESULT["memopro_report"] = memopro_report()
    path = os.environ.get("MP_OUT")
    text = json.dumps(RESULT, default=str)
    if path:
        with open(path, "w") as f:
            f.write(text)
    print("RESULT " + text[:2000], flush=True)


def guarded(main):
    """Run main(); record an OOM or any other error as the case's result instead of crashing."""
    try:
        main()
    except Exception as e:  # noqa: BLE001
        status = "oom" if is_oom(e) else "error"
        finish(status, error=f"{type(e).__name__}: {e}"[:1500], trace=traceback.format_exc()[-3000:])
        return
    if RESULT.get("status") is None:
        finish()


def wikitext_ids(tokenizer, split, n_tokens):
    """WikiText-2 raw tokens (the revision pinned in 0017), joined with blank lines."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                      revision="b08601e04326c79dfdd32d625aee71d232d685c3", split=split)
    text = "\n\n".join(t for t in ds["text"][: 20000 if split == "train" else None])
    ids = tokenizer(text, return_tensors="pt", verbose=False).input_ids[0]
    if ids.numel() < n_tokens:
        reps = n_tokens // ids.numel() + 1
        ids = ids.repeat(reps)
    return ids[:n_tokens]


def model_path(model_id):
    return os.environ.get("MP_MODEL_PATH") or model_id
