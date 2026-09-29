"""Assemble one Colab notebook per part (0092): examples/colab_t4_{train,infer,multi}.ipynb,
each a short guide and one self-contained code cell.

The code cell = its settings + common.py + the worker sources (written to disk at run time) +
its body. Usage: .venv/bin/python examples/colab_t4/build.py [--out-dir DIR for .py copies]
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORKER_FILES = ["worker_common.py", "worker_train.py", "worker_infer.py", "worker_multi.py",
                "app_naive_load.py"]

COMMON_PREP = """
**준비 (처음 한 번, 세 노트북 공통)**
1. 런타임 → 런타임 유형 변경 → **T4 GPU**.
2. Google Drive에 `memopro_colab/install/` 폴더를 만들고 memopro 소스 배포본 `memopro-0.1.0a1.tar.gz`를 올린다(Linux wheel이 있으면 그것을 쓴다). 소스 배포본은 첫 실행 때 Rust로 빌드한다(3~5분).
   - 또는 Colab 왼쪽 열쇠(보안 비밀)에 `GITHUB_TOKEN`을 **직접** 등록하면 저장소에서 설치한다.
3. 셀을 실행한다. Drive 연결 허용 창이 뜨면 허용한다. 설정은 셀 맨 위(`SETTINGS`)에서 바꾼다.

**결과**: `MyDrive/memopro_colab/results/{cell}/<실행 id>/`
- `summary.md`·`summary.csv`: 비교 표와 읽는 법
- `cases/*.json`: 경우마다 모든 측정값(메모리, 시간, memopro 보고, 오류와 추적)
- `logs/*.log`: 경우마다 프로세스 출력 전체
- `timelines/*.csv`: 0.5초 간격 GPU·호스트 메모리와 GPU 사용률
- `env.json`: GPU, 판, memopro 빌드 확인
- 같은 이름의 `.zip`: 폴더 전체(공유용)

**끊겼을 때**: 같은 셀을 다시 실행하면 끝난 경우는 건너뛰고 이어서 한다(`NEW_RUN = True`면 새로 시작). 모델은 Drive의 `hf_cache/`에 남아 다시 받지 않는다. 세 노트북은 같은 Drive 폴더와 모델 캐시를 함께 쓴다.

**공정성**: 모든 경우는 **새 프로세스**에서 돈다(앞 경우의 메모리가 섞이지 않게, 0055). 여러 모델을 **동시에** 한 GPU에서 재면 서로의 측정을 오염시키므로(0085), 여러 모델은 목록으로 차례로 잰다.
"""

NOTEBOOKS = {
    "train": ("colab_t4_train.ipynb", "cell_train.py", """# memopro Colab T4 재검증 ① 학습·개발 과정 (docs/research/0091, 0092)

GPT-2, Qwen2.5-0.5B·1.5B 전체 미세조정(fp32 AdamW, WikiText-2, 같은 시드·데이터·단계, dropout 끔)을 다섯 방식으로 비교한다: 일반 PyTorch(기준), fp16 AMP, HF gradient checkpointing, accelerate 배치 탐색, memopro `train_session`(+ lossless). 전체 T4와 메모리 상한(작은 GPU 흉내) 두 조건, `memopro.check` 예측 대 실측, 7B QLoRA(표준 방식 대 memopro).

- 예상 시간: 2~3시간(7B 첫 다운로드 약 15GB 포함)
- 판정(0091): T1 같은 모델 학습(손실 ≤ 1e-3, 파라미터 ≤ 1e-5), T2 작은 GPU에서 완주, T3 `check` ±15%, T4 안 될 때 제안과 함께 거절
"""),
    "infer": ("colab_t4_infer.ipynb", "cell_infer.py", """# memopro Colab T4 재검증 ② 로컬 AI 모델 구동 (docs/research/0091, 0092)

Qwen2.5 1.5B·3B·7B·14B Instruct를 흔한 방식(fp16 `device_map="auto"`, bitsandbytes 8비트·4비트)과 `memopro.load`(기본·`quality="low"`)로 불러와 비교한다: 불러오기 시간, 가중치 위치, GPU·호스트 최대 사용, 512·2048토큰 프롬프트 첫 토큰 시간, 디코딩 속도, WikiText-2 perplexity, 채팅 답변 일치(가장 정확한 방법 대비).

- 예상 시간: 2~3시간(첫 실행은 모델 다운로드 약 55GB 포함)
- 판정(0091): I1 memopro는 OOM으로 끝나지 않음(불러오거나 제안과 함께 거절), I2 품질·속도는 그대로 보고
"""),
    "multi": ("colab_t4_multi.ipynb", "cell_multi.py", """# memopro Colab T4 재검증 ③ 여러 모델·프로세스 수준 기능 (docs/research/0091, 0092)

doctor 정확도, 모델 3개(fp16 합 약 17GB > T4)를 한 GPU에서 번갈아 쓰기(모두 올려 두기 / 지우고 다시 불러오기 / memopro 동면), 수정하지 않은 스크립트를 `python` 대 `memopro run`, 최근 변경의 CUDA·Linux 회귀 검사 7종.

- 예상 시간: 약 1시간
- 판정(0091): R1 회귀 7종 통과(배포 전 필수), M1 동면 뒤 같은 답, M2·M3 보고
"""),
}


def commit():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout.strip() or "unknown"


def workers_block():
    lines = ["WORKERS = {"]
    for name in WORKER_FILES:
        src = (HERE / name).read_text()
        assert "'''" not in src, name
        lines.append(f"    {name!r}: r'''{src}''',")
    lines.append("}")
    return "\n".join(lines)


def compose(cell_file):
    text = (HERE / cell_file).read_text().replace("@@COMMIT@@", commit())
    head, body = text.split("# @@COMMON@@\n")
    common = (HERE / "common.py").read_text()
    return head + common + "\n\n" + workers_block() + "\n\n" + body


def notebook(title_md, src):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": title_md.splitlines(True)},
             {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": src.splitlines(True)}]
    return {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]) if "--out-dir" in sys.argv else None
    for name, (file, cell_file, head) in NOTEBOOKS.items():
        src = compose(cell_file)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{name}.py").write_text(src)
        nb = notebook(head + COMMON_PREP.replace("{cell}", name), src)
        out = ROOT / "examples" / file
        out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(out, len(nb["cells"][1]["source"]), "lines")


if __name__ == "__main__":
    main()
