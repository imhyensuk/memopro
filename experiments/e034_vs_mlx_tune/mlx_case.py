"""E034 (docs/research/0150): one mlx-tune case, run by the isolated venv .cache/venv-mlx.

    .cache/venv-mlx/bin/python experiments/e034_vs_mlx_tune/mlx_case.py MODEL TEXTS_JSON OUT_DIR STEPS

Prints one JSON line: losses and timings parsed from mlx-lm's reports, peak footprint.
"""

import ctypes
import json
import os
import re
import sys
import time


class _RusageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(36)]


def footprint():
    info = _RusageV4()
    lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info))
    return info.f7, info.f28


def main():
    model_name, texts_json, out_dir, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    base, _ = footprint()
    from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer

    texts = json.load(open(texts_json))
    t0 = time.perf_counter()
    model, tok = FastLanguageModel.from_pretrained(model_name, max_seq_length=160)
    model = FastLanguageModel.get_peft_model(
        model, r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    load_s = time.perf_counter() - t0
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=[{"text": t} for t in texts],
        args=SFTConfig(
            output_dir=out_dir,
            per_device_train_batch_size=1,
            max_steps=steps,
            learning_rate=2e-4,
            logging_steps=1,
            grad_checkpoint=True,
        ),
    )
    log = open(os.path.join(out_dir, "mlx_stdout.txt"), "w")
    real = sys.stdout
    sys.stdout = log
    t = time.perf_counter()
    try:
        trainer.train()
    finally:
        sys.stdout = real
        log.close()
    train_s = time.perf_counter() - t
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(os.path.join(out_dir, "mlx_stdout.txt")).read())
    # mlx-lm's table rows: iter, train loss, arrow, tokens/s, tokens so far (validation rows skip)
    iters = [
        {"iter": int(m[0]), "loss": float(m[1]), "tok_s": float(m[2])}
        for m in re.findall(r"^\s*(\d+)\s+([\d.]+)\s+\S\s+([\d.]+)\s+[\d.]+k?\s*$", text, re.MULTILINE)
    ]
    _, peak = footprint()
    print(json.dumps({"load_s": load_s, "train_s": train_s, "iters": iters,
                      "base_footprint": base, "peak_footprint": peak}))


if __name__ == "__main__":
    main()
