"""Build the E053 notebooks (docs/research/0268): one self-contained notebook per test, for a
fresh Colab T4 runtime (nothing cached: memopro from PyPI, models from the Hugging Face Hub).

    python examples/colab_breadth/build.py          # writes examples/colab_breadth/notebooks/
    python examples/colab_breadth/build.py --check  # fails if a notebook is out of date

Each notebook has four cells: what it tests (markdown), settings, the shared code written to
/content (bench.py, worker.py and the workload), and the test itself.
"""

import argparse
import json
import pprint
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "notebooks"
MEMOPRO = "0.1.4"
GiB, MiB = 1 << 30, 1 << 20

# ---------------------------------------------------------------------------- the tests
LM = [
    # name, repo, weight patterns, host budgets, gen (tokens, prompts), LoRA (steps, seq), part
    (
        "10_slm_qwen2.5-1.5b",
        "Qwen/Qwen2.5-1.5B-Instruct",
        ["*.safetensors"],
        [768 * MiB, 1536 * MiB],
        (32, 2),
        (5, 256),
        "SLM",
        "대조군: 이미 검증한 Qwen2 계열",
    ),
    (
        "11_slm_smollm2-1.7b",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        ["*.safetensors"],
        [1 * GiB, 2 * GiB],
        (32, 2),
        (5, 256),
        "SLM",
        "Llama 구조(공개 모델)",
    ),
    (
        "12_slm_phi-3.5-mini",
        "microsoft/Phi-3.5-mini-instruct",
        ["*.safetensors"],
        [2 * GiB, 4 * GiB],
        (32, 2),
        (5, 256),
        "SLM",
        "Phi3: q/k/v가 하나로 합쳐진 qkv_proj, gate/up이 합쳐진 MLP",
    ),
    (
        "20_llm_mistral-7b",
        "mistralai/Mistral-7B-Instruct-v0.3",
        ["model-*.safetensors"],
        [4 * GiB, 6 * GiB],
        (16, 2),
        (3, 256),
        "LLM",
        "14.5GB: T4 GPU(15GB)와 호스트 RAM(12.7GB)에 그냥은 들어가지 않는 크기",
    ),
    (
        "21_llm_qwen2.5-14b",
        "Qwen/Qwen2.5-14B-Instruct",
        ["*.safetensors"],
        [4 * GiB, 6 * GiB],
        (8, 1),
        (2, 128),
        "LLM",
        "29.5GB: GPU와 호스트 RAM을 합쳐도(28GB) 모자라 디스크에서 흘려 읽어야 하는 크기",
    ),
]
VISION = [
    # name, repo, patterns, class, budgets, batch, size, LoRA, note
    (
        "30_vision_dinov2-giant",
        "facebook/dinov2-giant",
        ["*.safetensors"],
        "Dinov2Model",
        [1100 * MiB, 2200 * MiB],
        8,
        224,
        True,
        "대조군(0239 통과). LoRA 손실 차이의 원인 판별",
    ),
    (
        "31_vision_clip-vit-bigg",
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        ["pytorch_model-*.safetensors"],
        "CLIPVisionModel",
        [2 * GiB, 4 * GiB],
        4,
        224,
        False,
        "큰 비전 인코더(CLIP 전체 체크포인트에서 비전 탑만)",
    ),
    (
        "32_vision_siglip-so400m",
        "google/siglip-so400m-patch14-384",
        ["*.safetensors"],
        "SiglipVisionModel",
        [512 * MiB, 1 * GiB],
        8,
        384,
        False,
        "다른 구조(SigLIP)",
    ),
    (
        "33_vision_convnextv2-huge",
        "facebook/convnextv2-huge-22k-512",
        ["*.safetensors"],
        "ConvNextV2ForImageClassification",
        [512 * MiB, 1 * GiB],
        4,
        512,
        False,
        "CNN",
    ),
]
SCRIPTS = [
    # name, workload, args, extra pip, note
    ("40_lib_sklearn", "sklearn_pca_rf.py", [], [], "scikit-learn: PCA + 랜덤 포레스트"),
    (
        "41_lib_pandas_merge",
        "pandas_merge.py",
        [],
        [],
        "pandas 조인과 고객별 마지막 주문(무작위 접근)",
    ),
    ("42_lib_numpy_knn", "knn_numpy.py", [], [], "NumPy BLAS 최근접 이웃(순차 반복)"),
    ("43_lib_xgboost", "xgboost_train.py", [], ["xgboost"], "XGBoost(자체 할당기)"),
    ("44_lib_scipy", "scipy_sparse_fft.py", [], [], "SciPy 희소 행렬 + FFT"),
    ("45_lib_polars", "polars_agg.py", [], ["polars"], "Polars(Arrow, 자체 할당기)"),
    ("50_data_image_stack", "image_stack.py", [4096], [], "E027 영상 스택(2GiB)"),
    ("51_data_classify", "classify.py", [], [], "E040 scikit-learn 분류(희소)"),
    ("52_data_dataframe", "dataframe.py", [], [], "E040 pandas 표(무작위 접근)"),
    (
        "53_data_simulate",
        "simulate.py",
        [],
        [],
        "E040 시뮬레이션(압축해도 1/2 상한에 안 들어감, 0265)",
    ),
    ("54_data_csv", "csv_roundtrip.py", [], [], "CSV 쓰기·읽기·정리"),
    ("55_data_sort", "sort_large.py", [], [], "3억 개 float32 정렬"),
    ("56_data_imgprep", "imgprep.py", [], [], "Pillow 이미지 전처리 → float32 묶음"),
]

# ---------------------------------------------------------------------------- cell bodies
BODY_LM = r"""
import sys, os
sys.path.insert(0, CODE)
import bench
env = bench.setup(f"memopro[llm]=={MEMOPRO}", ["datasets"])
path, size = bench.fetch(SPEC["repo"], SPEC["patterns"])
gib = lambda b: f"{b / bench.GiB:.2f}GiB"
(tokens, prompts), (steps, seq) = SPEC["gen"], SPEC["lora"]
b1, b2 = SPEC["budgets"]
R, gen, lora = [], {}, {}
T = SPEC["timeout_s"]
gen["plain"] = bench.worker("gen_plain", ["lm_gen", path, "plain", 0, tokens, prompts], T)
for b in (b1, b2):
    gen[b] = bench.worker(f"gen_memopro_{gib(b)}", ["lm_gen", path, "memopro", b, tokens, prompts], T)
if SPEC["plain_lora"]:  # LLMs larger than the GPU: the plain generation case shows the failure
    lora["plain"] = bench.worker("lora_plain", ["lm_lora", path, "plain", 0, steps, seq, 1], T)
for b in (b1, b2):
    lora[b] = bench.worker(f"lora_memopro_{gib(b)}", ["lm_lora", path, "memopro", b, steps, seq, 1], T)
auto = bench.worker("lora_memopro_auto", ["lm_lora", path, "memopro", b2, steps, seq, "auto"], T)
R = [gen["plain"], gen[b1], gen[b2], *([lora["plain"]] if SPEC["plain_lora"] else []), lora[b1], lora[b2], auto]
ok = lambda r: r.get("status") == "ok"
checks = []
if ok(gen["plain"]):
    for b in (b1, b2):
        checks.append((f"생성: memopro({gib(b)}) 토큰 = 일반 실행", bench.same(gen[b], gen["plain"], "tokens"), ""))
else:
    checks.append(("일반 실행(생성)", None, f"{gen['plain'].get('status')}: {str(gen['plain'].get('error', ''))[:120]}"))
    checks.append((f"생성: 예산 {gib(b1)} 토큰 = {gib(b2)}", bench.same(gen[b1], gen[b2], "tokens"), ""))
checks.append((f"LoRA: 예산 {gib(b1)} 손실 = {gib(b2)} (비트 단위)", bench.same(lora[b1], lora[b2], "loss_bits"), str(lora[b1].get("losses"))[:120]))
checks.append(("memopro가 이 계열에서 동작(생성·LoRA)", all(ok(r) for r in (gen[b1], gen[b2], lora[b1], lora[b2])),
               "; ".join(f"{r['case']}: {r.get('error', '')[:80]}" for r in (gen[b1], gen[b2], lora[b1], lora[b2]) if not ok(r))))
if SPEC["plain_lora"]:
    checks.append(("일반 실행(LoRA)", None, lora["plain"].get("status")))
checks.append(('batch="auto"가 고른 batch', None, f"{auto.get('batch')} ({auto.get('status')})"))
for b in (b1, b2):
    checks.append((f"속도 {gib(b)}", None, f"{gen[b].get('s_per_token', 0):.2f} 초/토큰, LoRA 단계 {lora[b].get('step_s')}"))
if ok(gen["plain"]):
    checks.append(("속도 일반 실행", None, f"{gen['plain'].get('s_per_token', 0):.2f} 초/토큰"))
bench.verdict(SPEC["name"], env, R, checks)
"""

BODY_VISION = r"""
import sys, os
sys.path.insert(0, CODE)
import bench
env = bench.setup(f"memopro[llm]=={MEMOPRO}", [])
path, size = bench.fetch(SPEC["repo"], SPEC["patterns"])
gib = lambda b: f"{b / bench.GiB:.2f}GiB"
b1, b2 = SPEC["budgets"]
cls, batch, px, T = SPEC["cls"], SPEC["batch"], SPEC["size"], SPEC["timeout_s"]
run = lambda name, *a: bench.worker(name, list(a), T)
inf = {"plain": run("infer_plain", "vis_infer", path, cls, "plain", 0, batch, px)}
for b in (b1, b2):
    inf[b] = run(f"infer_memopro_{gib(b)}", "vis_infer", path, cls, "memopro", b, batch, px)
R = list(inf.values())
ok = lambda r: r.get("status") == "ok"
checks = []
for b in (b1, b2):
    if ok(inf["plain"]):
        checks.append((f"추론: memopro({gib(b)}) 출력 = 일반 실행 (비트 단위)", bench.same(inf[b], inf["plain"], "output_sha"), ""))
    else:
        checks.append((f"추론: memopro({gib(b)}) 동작", ok(inf[b]), inf[b].get("error", "")[:120]))
if not ok(inf["plain"]):
    checks.append(("일반 실행", None, inf["plain"].get("status")))
    checks.append((f"추론: {gib(b1)} 출력 = {gib(b2)}", bench.same(inf[b1], inf[b2], "output_sha"), ""))
for k, r in inf.items():
    checks.append((f"시간 {k if k == 'plain' else gib(k)}", None, f"첫 {r.get('first_s', 0):.2f}s, 둘째 {r.get('second_s', 0):.2f}s"))
if SPEC["lora"]:
    la = run(f"lora_memopro_{gib(b1)}_a", "vis_lora", path, cls, "memopro", b1, 5, batch, px)
    lb = run(f"lora_memopro_{gib(b1)}_b", "vis_lora", path, cls, "memopro", b1, 5, batch, px)
    lc = run(f"lora_memopro_{gib(b2)}", "vis_lora", path, cls, "memopro", b2, 5, batch, px)
    pa = run("lora_plain_a", "vis_lora", path, cls, "plain", 0, 5, batch, px)
    pb = run("lora_plain_b", "vis_lora", path, cls, "plain", 0, 5, batch, px)
    R += [la, lb, lc, pa, pb]
    checks.append((f"LoRA: 같은 예산 두 번 손실 같음 (결정성 켬)", bench.same(la, lb, "loss_bits"), str(la.get("losses"))[:100]))
    checks.append((f"LoRA: 예산 {gib(b1)} = {gib(b2)}", bench.same(la, lc, "loss_bits"), str(lc.get("losses"))[:100]))
    checks.append(("LoRA: 일반 실행 두 번 같음 (보고)", None, bench.same(pa, pb, "loss_bits")))
    checks.append(("LoRA: memopro = 일반 실행 (보고)", None, bench.same(la, pa, "loss_bits")))
bench.verdict(SPEC["name"], env, R, checks)
"""

BODY_SCRIPT = r"""
import sys, os
sys.path.insert(0, CODE)
import bench
env = bench.setup(f"memopro=={MEMOPRO}", SPEC["pip"])
wl = os.path.join(CODE, "workloads", SPEC["workload"])
SKIP = {"seconds", "peak_bytes", "case", "wall_s", "warned_ceiling", "status", "ceiling",
        "peak_footprint", "exceeded_bytes", "overruns", "stderr", "returncode", "timeout_s",
        "xgboost"}
result = lambda r: {k: v for k, v in r.items() if k not in SKIP}
plain = bench.script("plain", wl, SPEC["args"], SPEC["timeout_s"])
R, checks = [plain], []
if plain.get("status") != "crashed" and plain.get("peak_bytes"):
    P = plain["peak_bytes"]
    for frac in (0.5, 0.75):
        ceiling = int(P * frac)
        r = bench.script(f"memopro_{int(frac * 100)}pct", wl, SPEC["args"], 2 * SPEC["timeout_s"], budget=ceiling)
        R.append(r)
        done = "seconds" in r
        checks.append((f"결과 = 일반 실행 (상한 {frac:.0%})", done and result(r) == result(plain),
                       r.get("status") if not done else ""))
        peak, over = r.get("peak_footprint") or r.get("peak_bytes") or 0, r.get("exceeded_bytes") or 0
        if peak and peak <= ceiling + bench.TOLERANCE:
            c = (True, f"최대 {peak / bench.GiB:.2f} ≤ 상한 {ceiling / bench.GiB:.2f} GiB")
        elif r.get("warned_ceiling") and over:
            c = (True, f"상한을 지킬 수 없어 경고, +{over / bench.MiB:.0f} MiB 보고 (0265의 약속)")
        else:
            c = (False, f"최대 {peak / bench.GiB:.2f} GiB, 상한 {ceiling / bench.GiB:.2f}, 경고 {r.get('warned_ceiling')}")
        checks.append((f"상한 {frac:.0%}: 지킴 또는 경고·보고", *c))
        if done:
            checks.append((f"시간 배율 {frac:.0%}", None, f"{r['seconds'] / max(plain['seconds'], 1e-9):.2f}x ({r['seconds']:.0f}s / {plain['seconds']:.0f}s)"))
else:
    checks.append(("일반 실행", False, plain.get("status")))
bench.verdict(SPEC["name"], env, R, checks, [f"일반 실행 최대 메모리 {plain.get('peak_bytes', 0) / bench.GiB:.2f} GiB"])
"""

BODY_UNSLOTH = r"""
import sys, os, subprocess
sys.path.insert(0, CODE)
import bench
env = bench.setup(f"memopro[llm]=={MEMOPRO}", ["datasets"])
VENV = "/content/venv-unsloth"
py = os.path.join(VENV, "bin", "python")
if not os.path.exists(py):  # uv: Colab's Python has no ensurepip (0238)
    bench.log("installing Unsloth into its own environment with uv (~3-6 min)")
    for cmd in ([sys.executable, "-m", "pip", "install", "-q", "uv"],
                [sys.executable, "-m", "uv", "venv", "-q", VENV],
                [sys.executable, "-m", "uv", "pip", "install", "-q", "--python", py, "unsloth", "datasets"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            bench.log(f"failed: {(r.stdout + r.stderr)[-800:]}")
v = subprocess.run([py, "-c", "import unsloth; print(unsloth.__version__)"], capture_output=True, text=True)
env["unsloth"] = v.stdout.strip() or v.stderr[-300:]
bench.log(f"Unsloth {env['unsloth']}")
path, size = bench.fetch(SPEC["repo"], ["*.safetensors"])
steps, seq, T = SPEC["steps"], SPEC["seq"], SPEC["timeout_s"]
mp = bench.worker("memopro_16bit", ["lm_lora", path, "memopro", SPEC["budget"], steps, seq, 1], T)
u16 = bench.worker("unsloth_16bit", ["unsloth", path, 16, steps, seq], T, python=py)
path4, _ = bench.fetch(SPEC["repo4"], ["*.safetensors"])
u4 = bench.worker("unsloth_4bit", ["unsloth", path4, 4, steps, seq], T, python=py)
ok = lambda r: r.get("status") == "ok"
checks = [
    ("B1: Unsloth 16비트는 실패하고 memopro 16비트는 완주", (not ok(u16)) and ok(mp),
     f"Unsloth 16: {u16.get('status')} {str(u16.get('error', ''))[:80]}; memopro: {mp.get('status')}"),
    ("Unsloth 4비트(손실 있음)", None, f"{u4.get('status')}, 단계 {u4.get('step_s')}"),
    ("memopro 16비트 단계 시간", None, str(mp.get("step_s"))),
    ("첫 손실 비교", None, f"memopro {str(mp.get('losses'))[:40]} / Unsloth4 {str(u4.get('losses'))[:40]}"),
]
bench.verdict(SPEC["name"], env, [mp, u16, u4], checks)
"""

BODY_TESTS = r"""
import sys, os, subprocess, time, json
sys.path.insert(0, CODE)
import bench
env = bench.setup(f"memopro[llm]=={MEMOPRO}", ["pytest", "ipython"])
src = os.path.join(bench.ROOT, "memopro-src")
if not os.path.isdir(src):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", f"v{MEMOPRO}",
                    "https://github.com/imhyensuk/memopro.git", src], check=True)
bench.log("pytest: start (the repository's tests against the installed PyPI package)")
t = time.time()
p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", "tests"],
                   capture_output=True, text=True, cwd=src, timeout=SPEC["timeout_s"])
out = p.stdout + p.stderr
summary = [l for l in out.splitlines() if " passed" in l or " failed" in l or " error" in l]
failed = [l for l in out.splitlines() if l.startswith(("FAILED", "ERROR"))]
rec = {"case": "pytest", "status": "ok" if p.returncode == 0 else "failed", "wall_s": round(time.time() - t, 1),
       "summary": summary[-1] if summary else "", "failed": failed, "output_tail": out[-20000:]}
with open(os.path.join(bench.RESULTS, "pytest.json"), "w") as f:
    json.dump(rec, f, indent=1)
print(out[-2500:])
checks = [("pytest (설치한 PyPI 버전, T4)", p.returncode == 0, rec["summary"]),
          ("실패한 시험", None, "; ".join(failed)[:400] or "없음")]
bench.verdict(SPEC["name"], env, [rec], checks)
"""


# ---------------------------------------------------------------------------- notebook
def notebook(name, title, md, spec, body, files):
    settings = (
        "# ==== 설정 (필요하면 여기만 고치세요) ====\n"
        f'MEMOPRO = "{MEMOPRO}"  # PyPI에서 설치할 memopro 버전\n'
        "SAVE_TO_DRIVE = False  # True면 결과를 Google Drive(memopro_e053/)에도 복사합니다\n"
        f"SPEC = {pprint.pformat(spec, width=96, sort_dicts=False)}\n"
        "import os\n\n"
        'CODE = os.environ.get("MP_BENCH_CODE", "/content/memopro_bench/code")\n'
        'os.environ["MP_SAVE_TO_DRIVE"] = "1" if SAVE_TO_DRIVE else "0"\n'
    )
    writer = (
        "# 이 노트북이 쓰는 코드를 새 런타임에 파일로 씁니다 (bench.py, worker.py, 작업 스크립트)\n"
        "import os\n"
        f"FILES = {json.dumps(files, ensure_ascii=False)}\n"
        "for rel, text in FILES.items():\n"
        "    p = os.path.join(CODE, rel)\n"
        "    os.makedirs(os.path.dirname(p), exist_ok=True)\n"
        "    with open(p, 'w') as f:\n"
        "        f.write(text)\n"
        'print("wrote", ", ".join(FILES))\n'
    )

    def code(src):
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True),
        }

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": f"# {title}\n\n{md}".splitlines(keepends=True),
        },
        code(settings),
        code(writer),
        code(body),
    ]
    return {
        "cells": cells,
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
    }


HOW = (
    "**실행 방법**: Colab에서 이 노트북을 연 뒤 *런타임 → 런타임 유형 변경 → T4 GPU*를 고르고 "
    "*런타임 → 모두 실행*을 누르세요. 캐시가 없는 새 런타임을 가정합니다: memopro는 PyPI에서, "
    "모델은 Hugging Face에서 새로 받습니다(토큰 불필요). 각 경우는 따로 된 프로세스에서 돌고, "
    "끝에 경우별 표와 판정표가 출력됩니다. 결과 JSON은 `/content/memopro_bench/results/`에 "
    "남고, `SAVE_TO_DRIVE = True`면 Drive에도 복사됩니다.\n\n"
    "연구 기록: docs/research/0268 (E053)."
)


def build():
    shared = {
        "bench.py": (HERE / "bench.py").read_text(),
        "worker.py": (HERE / "worker.py").read_text(),
    }
    common = {"workloads/common.py": (HERE / "workloads" / "common.py").read_text()}
    books = {}
    for name, repo, pats, budgets, gen, lora, part, note in LM:
        spec = {
            "name": name,
            "repo": repo,
            "patterns": pats,
            "budgets": budgets,
            "gen": gen,
            "lora": lora,
            "plain_lora": part == "SLM",
            "timeout_s": 5400 if "14b" in name else 2400,
        }
        md = (
            f"**{part}** · `{repo}` — {note}\n\n"
            "**무엇을 보나**\n"
            "- 생성: 일반 실행(GPU에 그대로 올림) 대 memopro(호스트 예산 두 가지, 가중치를 흘려 씀). "
            "일반 실행이 되면 토큰이 같아야 하고, 안 되면 두 예산의 토큰이 같아야 합니다.\n"
            "- LoRA: memopro 두 예산의 손실이 비트 단위로 같아야 합니다. "
            '`batch="auto"`가 고른 batch도 기록합니다.\n'
            "- 이 계열에서 memopro가 동작하는지(구조 호환성).\n\n" + HOW
        )
        books[name] = notebook(name, f"E053 · {name}", md, spec, BODY_LM, shared)
    for name, repo, pats, cls, budgets, batch, size, lora, note in VISION:
        spec = {
            "name": name,
            "repo": repo,
            "patterns": pats,
            "cls": cls,
            "budgets": budgets,
            "batch": batch,
            "size": size,
            "lora": lora,
            "timeout_s": 1800,
        }
        md = (
            f"**비전** · `{repo}` ({cls}) — {note}\n\n"
            "**무엇을 보나**: 추론 출력이 일반 실행과 비트 단위로 같은지(호스트 예산 두 가지)"
            + (", LoRA 손실이 같은 예산 두 번·다른 예산에서 같은지(결정성 켬)" if lora else "")
            + ".\n\n"
            + HOW
        )
        books[name] = notebook(name, f"E053 · {name}", md, spec, BODY_VISION, shared)
    for name, workload, args, pip, note in SCRIPTS:
        spec = {"name": name, "workload": workload, "args": args, "pip": pip, "timeout_s": 1800}
        files = {
            **shared,
            **common,
            f"workloads/{workload}": (HERE / "workloads" / workload).read_text(),
        }
        md = (
            f"**{'고사양 라이브러리' if name.startswith('4') else '데이터 처리'}** · "
            f"`{workload}` — {note}\n\n"
            "**무엇을 보나**: 코드를 고치지 않은 프로그램을 일반 실행으로 한 번 돌려 최대 메모리 P를 "
            "잰 뒤, `memopro run --budget`으로 상한 P/2와 3P/4에서 다시 돌립니다.\n"
            "- 결과(해시)가 일반 실행과 같아야 합니다.\n"
            "- 최대 메모리가 상한 + 32MiB 안이거나, 지킬 수 없으면 **실행 중 경고와 초과량 보고**가 "
            "있어야 합니다(0.1.4의 약속, 0265).\n- 시간 배율은 보고만 합니다.\n\n" + HOW
        )
        books[name] = notebook(name, f"E053 · {name}", md, spec, BODY_SCRIPT, files)
    spec = {
        "name": "22_llm_qwen2.5-14b_vs_unsloth",
        "repo": "Qwen/Qwen2.5-14B-Instruct",
        "repo4": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "budget": 6 * GiB,
        "steps": 2,
        "seq": 128,
        "timeout_s": 5400,
    }
    md = (
        "**LLM 비교 (E049 W1)** · Qwen2.5-14B 16비트 LoRA: Unsloth(16비트·4비트) 대 memopro(16비트)\n\n"
        "**무엇을 보나**: 같은 T4에서 Unsloth 16비트는 14B를 담지 못해 실패하고, memopro는 16비트 "
        "그대로 완주하는지(B1). Unsloth 4비트(손실 있음)의 속도·손실도 함께 기록합니다. "
        "Unsloth는 uv로 만든 별도 환경에 설치합니다(0238).\n\n" + HOW
    )
    books[spec["name"]] = notebook(
        spec["name"], f"E053 · {spec['name']}", md, spec, BODY_UNSLOTH, shared
    )
    spec = {"name": "00_tests", "timeout_s": 2400}
    md = (
        "**시험 묶음** · PyPI에 배포한 memopro를 설치하고, 같은 버전 태그의 저장소 시험(pytest)을 "
        "T4(CUDA)에서 돌립니다. 0.1.4의 상한 초과 경고 시험이 Linux에서 통과하는지도 여기서 봅니다.\n\n"
        + HOW
    )
    books["00_tests"] = notebook("00_tests", "E053 · 00_tests", md, spec, BODY_TESTS, shared)
    return books


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    books = build()
    stale = []
    OUT.mkdir(exist_ok=True)
    for name, nb in sorted(books.items()):
        text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
        path = OUT / f"{name}.ipynb"
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(path.name)
        else:
            path.write_text(text)
    if stale:
        sys.exit(f"out of date (run build.py): {', '.join(stale)}")
    print(f"{len(books)} notebooks {'checked' if args.check else 'written'} in {OUT}")


if __name__ == "__main__":
    main()
