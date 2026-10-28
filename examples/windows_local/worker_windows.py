"""E070 worker (docs/research/0364): one case in a fresh process on a Windows (or any) CPU machine.

    python worker_windows.py OUT.json KIND [args...]

Every result records the process's peak working set (psutil ``peak_wset`` on Windows, peak RSS
elsewhere) and the bytes it wrote (psutil I/O counters). Outputs of sorts and joins are hashed as
they stream (SHA-256 for sorts, an order-free row hash for joins), never written.
"""

import hashlib
import json
import os
import sys
import time
import traceback

OUT, KIND, ARGS = sys.argv[1], sys.argv[2], sys.argv[3:]
RESULT = {"kind": KIND, "args": ARGS, "status": None}
PROMPTS = [
    "Explain how a hash table handles collisions.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
]
TEXT = (
    "Memory is the scarcest resource on small machines. A program that needs more than the "
    "machine has either fails, swaps to the disk, or is changed to need less. Lossless methods "
    "keep the results exactly as they were: they read data again from its file, compress what "
    "waits, and recompute what can be recomputed. "
)


def proc_stats():
    """Peak memory and bytes written by this process (Windows: peak working set and I/O
    counters; macOS has no I/O counters in psutil)."""
    import psutil

    p = psutil.Process()
    peak = getattr(p.memory_info(), "peak_wset", None) or _maxrss()
    io = p.io_counters() if hasattr(p, "io_counters") else None
    return {
        "peak_bytes": peak,
        "written_bytes": io.write_bytes if io else None,
        "read_bytes": io.read_bytes if io else None,
    }


def _maxrss():
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


class RowHash:
    """Order-free hash of the rows it is fed, header (first line) left out: the sum of 64-bit
    row hashes mod 2**64, and the row count. Rows never hold line breaks here."""

    def __init__(self):
        self.carry, self.total, self.rows, self.header = b"", 0, 0, None

    def update(self, data):
        data = self.carry + bytes(data)
        cut = data.rfind(b"\n") + 1
        self.carry = data[cut:]
        lines = data[:cut].split(b"\n")[:-1]
        if self.header is None and lines:
            self.header, lines = lines[0], lines[1:]
        for line in lines:
            line = line.rstrip(b"\r")
            self.total = (
                self.total + int.from_bytes(hashlib.blake2b(line, digest_size=8).digest(), "little")
            ) % (1 << 64)
        self.rows += len(lines)

    def value(self):
        return f"{self.total:016x}:{self.rows}"


def tok_and_model_dir(path):
    import transformers

    return transformers.AutoTokenizer.from_pretrained(path)


def chat_ids(tok, prompt):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    )
    return ids["input_ids"] if hasattr(ids, "keys") else ids


def gen_plain(path, n):
    import torch
    import transformers

    tok = tok_and_model_dir(path)
    t = time.time()
    m = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    RESULT["load_s"] = time.time() - t
    texts, secs = [], []
    for p in PROMPTS:
        ids = chat_ids(tok, p)
        t = time.time()
        with torch.no_grad():
            out = m.generate(
                input_ids=ids,
                do_sample=False,
                max_new_tokens=n,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        secs.append(time.time() - t)
        texts.append(tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True))
    RESULT.update(texts=texts, seconds=secs, s_per_token=sum(secs) / _tokens(tok, texts))


def _tokens(tok, texts):
    return max(1, sum(len(tok(t, add_special_tokens=False).input_ids) for t in texts))


def gen_mp(path, n, ceiling):
    import memopro

    tok = tok_and_model_dir(path)
    texts, secs = [], []
    for p in PROMPTS:
        t = time.time()
        texts.append(
            memopro.generate(
                path, p, tokenizer=tok, budget=int(ceiling), device="cpu", max_new_tokens=n
            )
        )
        secs.append(time.time() - t)
    RESULT.update(
        texts=texts, seconds=secs, s_per_token=sum(secs) / _tokens(tok, texts), ceiling=int(ceiling)
    )


def gen_llama(gguf, n):
    from llama_cpp import Llama

    t = time.time()
    m = Llama(model_path=gguf, n_ctx=1024, n_threads=os.cpu_count(), verbose=False, seed=0)
    RESULT["load_s"] = time.time() - t
    texts, secs = [], []
    for p in PROMPTS:
        t = time.time()
        r = m.create_chat_completion(
            messages=[{"role": "user", "content": p}], max_tokens=n, temperature=0.0, top_k=1
        )
        secs.append(time.time() - t)
        texts.append(r["choices"][0]["message"]["content"])
        RESULT.setdefault("tokens", []).append(r["usage"]["completion_tokens"])
    RESULT.update(
        texts=texts,
        seconds=secs,
        s_per_token=sum(secs) / max(1, sum(RESULT["tokens"])),
        gguf=os.path.basename(gguf),
    )


def lora_texts(steps, seq):
    return [f"{i}. " + TEXT * (1 + seq // 60) for i in range(steps)]


def lora_plain(path, steps, seq):
    import peft
    import torch
    import transformers

    torch.manual_seed(0)
    tok = tok_and_model_dir(path)
    m = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    m = peft.get_peft_model(
        m,
        peft.LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
    )
    for p in m.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    ids = tok((tok.eos_token or "").join(lora_texts(steps, seq)), return_tensors="pt").input_ids[0]
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-4)
    losses, secs = [], []
    for i in range(steps):
        x = ids[i * seq : (i + 1) * seq][None]
        t = time.time()
        loss = m(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        secs.append(time.time() - t)
        losses.append(float(loss))
    RESULT.update(losses=losses, step_s=secs)


def lora_mp(path, steps, seq, ceiling):
    import memopro

    tok = tok_and_model_dir(path)
    texts = lora_texts(steps, seq)
    r = memopro.finetune(path, texts, tokenizer=tok, budget=int(ceiling), device="cpu", seq_len=seq)
    RESULT.update(
        losses=r.losses[:steps],
        loss_bits=[float(x).hex() for x in r.losses],
        step_s=r.step_seconds,
        ceiling=int(ceiling),
    )


def csv_gen(data, gib, dims):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import csvgen

    os.makedirs(data, exist_ok=True)
    rows = int(float(gib) * (1 << 30) / csvgen.ROW_BYTES)
    RESULT["fact_bytes"] = csvgen.write(
        os.path.join(data, "fact.csv"), csvgen.fact_blocks(rows, int(dims))
    )
    RESULT["dim_bytes"] = csvgen.write(os.path.join(data, "dim.csv"), csvgen.dim_blocks(int(dims)))
    RESULT.update(rows=rows, dims=int(dims))


def sort_mp(data, ceiling):
    import memopro

    h = hashlib.sha256()
    st = memopro.sort(os.path.join(data, "fact.csv"), "id", out=h.update, budget=int(ceiling))
    st.pop("runtime", None)
    RESULT.update(hash=h.hexdigest(), stats=st, ceiling=int(ceiling))


def join_mp(data, ceiling):
    import memopro

    h = RowHash()
    st = memopro.join(
        os.path.join(data, "fact.csv"),
        os.path.join(data, "dim.csv"),
        "dim_id",
        out=h.update,
        budget=int(ceiling),
    )
    RESULT.update(hash=h.value(), stats=st, ceiling=int(ceiling))


def _duck(query, limit, sink, data):
    """DuckDB with its memory limit; text columns joined by commas as the file's lines are."""
    import duckdb
    import pyarrow.compute as pc

    tmp = os.path.join(data, "duck_tmp")
    os.makedirs(tmp, exist_ok=True)
    con = duckdb.connect(
        config={
            "memory_limit": f"{int(limit) >> 20}MB",
            "temp_directory": tmp,
            "threads": os.cpu_count(),
        }
    )
    reader = con.execute(query).fetch_record_batch(1 << 18)
    sink(b"header\n")
    for batch in reader:
        line = pc.binary_join_element_wise(*batch.columns, ",")
        sink(("\n".join(line.to_pylist()) + "\n").encode())
    con.close()


def sort_duck(data, limit):
    h = hashlib.sha256()
    fact = os.path.join(data, "fact.csv").replace("\\", "/")
    with open(os.path.join(data, "fact.csv"), "rb") as f:
        header = f.readline()
    first = [True]

    def sink(b):
        if first[0]:  # the file's own header instead of the placeholder
            first[0] = False
            h.update(header)
            return
        h.update(b)

    _duck(
        f"SELECT * FROM read_csv('{fact}', all_varchar=true, header=true) "
        "ORDER BY CAST(id AS BIGINT)",
        limit,
        sink,
        data,
    )
    RESULT.update(hash=h.hexdigest(), memory_limit=int(limit))


def join_duck(data, limit):
    h = RowHash()
    fact = os.path.join(data, "fact.csv").replace("\\", "/")
    dim = os.path.join(data, "dim.csv").replace("\\", "/")
    _duck(
        f"SELECT l.id, l.dim_id, l.a, l.b, l.text, r.name, r.w FROM read_csv('{fact}', "
        f"all_varchar=true, header=true) l JOIN read_csv('{dim}', all_varchar=true, "
        "header=true) r ON CAST(l.dim_id AS BIGINT) = CAST(r.dim_id AS BIGINT)",
        limit,
        h.update,
        data,
    )
    RESULT.update(hash=h.value(), memory_limit=int(limit))


def sort_pandas(data):
    import pandas as pd

    t = time.time()
    df = pd.read_csv(os.path.join(data, "fact.csv"), dtype=str, keep_default_na=False)
    RESULT["read_s"] = time.time() - t
    df = df.iloc[df["id"].astype("int64").argsort(kind="stable")]
    h = hashlib.sha256()
    for i in range(0, len(df), 1 << 20):  # the same text as the file's lines, piece by piece
        h.update(
            df.iloc[i : i + (1 << 20)]
            .to_csv(index=False, header=i == 0, lineterminator="\n")
            .encode()
        )
    RESULT["hash"] = h.hexdigest()


def codec(mib):
    import numpy as np

    from memopro import rt

    n = int(mib) << 20
    rng = np.random.default_rng(0)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import csvgen

    def csv_text():
        text = bytearray()
        for b in csvgen.fact_blocks(n // csvgen.ROW_BYTES + 1, 1_000_000):
            text += b
            if len(text) >= n:
                break
        return np.frombuffer(bytes(text[:n]), np.uint8)

    x = np.arange(4096, dtype=np.float32) / 4096
    kinds = {  # made one at a time
        "CSV text": csv_text,
        "int64 keys": lambda: rng.permutation(n // 8).astype(np.int64),
        "uint16 image": lambda: (
            np.resize((2000 + 900 * np.sin(np.arange(1 << 20) / 997.0)).astype(np.uint16), n // 2)
            + rng.integers(0, 4, n // 2, dtype=np.uint16)
        ),
        "float32 field": lambda: np.resize(np.sin(9 * x)[None, :] * np.cos(7 * x)[:, None], n // 4),
        "random bytes": lambda: rng.integers(0, 256, n, dtype=np.uint8),
    }
    out = []
    for name, make in kinds.items():
        raw = make().view(np.uint8).reshape(-1)
        want = hashlib.sha256(raw).hexdigest()
        r = rt.Runtime(budget=3 * n, process_budget=False, min_saving=0.0)
        buf = r.alloc(raw.nbytes)
        with buf.pin(write=True) as v:
            np.frombuffer(v, np.uint8)[:] = raw
        buf.evict()
        s1 = r.stats()
        with buf.pin() as v:
            same = hashlib.sha256(np.frombuffer(v, np.uint8)).hexdigest() == want
        s2 = r.stats()
        c, d = s1["compress_seconds"], s2["decompress_seconds"]
        out.append(
            {
                "kind": name,
                "identical": same,
                "ratio": raw.nbytes / s1["compress_out"] if s1["compress_out"] else None,
                "compress_gbps": raw.nbytes / c / 1e9 if c else None,
                "decompress_gbps": raw.nbytes / d / 1e9 if d else None,
            }
        )
        del buf, r, raw
    RESULT["codec"] = out


def env():
    import platform
    import subprocess

    import psutil

    RESULT.update(
        platform=platform.platform(),
        python=platform.python_version(),
        processor=platform.processor(),
        cpus=os.cpu_count(),
        ram_bytes=psutil.virtual_memory().total,
        swap_bytes=psutil.swap_memory().total,
    )
    r = subprocess.run(
        [sys.executable, "-m", "memopro", "doctor", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    RESULT["doctor"] = r.stdout[-4000:] or r.stderr[-2000:]
    import memopro

    RESULT["memopro"] = memopro.__version__
    for name in ("torch", "transformers", "peft", "duckdb", "pandas", "llama_cpp"):
        try:
            RESULT[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - recorded as missing
            RESULT[name] = None


KINDS = {
    "env": env,
    "gen_plain": gen_plain,
    "gen_mp": gen_mp,
    "gen_llama": gen_llama,
    "lora_plain": lora_plain,
    "lora_mp": lora_mp,
    "csv_gen": csv_gen,
    "sort_mp": sort_mp,
    "join_mp": join_mp,
    "sort_duck": sort_duck,
    "join_duck": join_duck,
    "sort_pandas": sort_pandas,
    "codec": codec,
}

if __name__ == "__main__":
    t0 = time.time()
    try:
        KINDS[KIND](*[int(a) if a.isdigit() else a for a in ARGS])
        RESULT["status"] = "ok"
    except MemoryError as e:
        RESULT.update(status="oom", error=f"MemoryError: {e}"[:500])
    except Exception as e:  # noqa: BLE001 - recorded
        name = type(e).__name__
        RESULT.update(
            status="refused" if name == "BudgetExceeded" else "error",
            error=f"{name}: {e}"[:1500],
            trace=traceback.format_exc()[-3000:],
        )
    RESULT["wall_s"] = time.time() - t0
    try:
        RESULT.update(proc_stats())
    except Exception as e:  # noqa: BLE001
        RESULT["stats_error"] = str(e)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(RESULT, f, default=str)
