"""Assemble examples/colab_t4_heavy.ipynb: one self-contained code cell per test.

Each code cell = its settings + common.py + the worker sources (written to disk at run time) +
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
CELLS = [("cell_train.py", "train"), ("cell_infer.py", "infer"), ("cell_multi.py", "multi")]

INTRO = """# memopro Colab T4 재검증: 무거운 모델, 학습·추론·여러 모델 (docs/research/0091)

**셀 3개는 서로 독립이다.** 어느 셀이든 하나만 실행해도 된다(각 셀이 Drive 연결, 설치, 이어하기를 스스로 한다).

| 셀 | 무엇을 재는가 | 예상 시간(T4) |
|---|---|---|
| 1. 학습·개발 | GPT-2, Qwen2.5-0.5B·1.5B 전체 미세조정(fp32 AdamW): 일반 PyTorch, fp16 AMP, HF gradient checkpointing, accelerate 배치 탐색, memopro `train_session`(+ lossless). 전체 T4와 작은 GPU 흉내(메모리 상한). `memopro.check` 예측 대 실측. 7B QLoRA(표준 방식 대 memopro) | 2~3시간 |
| 2. 로컬 AI 모델 | Qwen2.5 1.5B·3B·7B·14B: fp16 `device_map="auto"`, bitsandbytes 8비트·4비트, `memopro.load`(기본·`quality="low"`). 메모리, 불러오기 시간, 긴 프롬프트 첫 토큰 시간, 디코딩 속도, WikiText-2 perplexity, 답변 일치 | 2~3시간 (첫 실행은 모델 다운로드 약 55GB 포함) |
| 3. 여러 모델·프로세스 | doctor 정확도, 모델 3개를 한 GPU에서 번갈아 쓰기(모두 올려 두기 / 지우고 다시 불러오기 / memopro 동면), 수정하지 않은 스크립트를 `python` 대 `memopro run`, 최근 변경의 CUDA 회귀 검사 | 약 1시간 |

**준비 (처음 한 번)**
1. 런타임 → 런타임 유형 변경 → **T4 GPU**.
2. Google Drive에 `memopro_colab/install/` 폴더를 만들고 memopro 소스 배포본 `memopro-0.1.0a1.tar.gz`를 올린다(Linux wheel이 있으면 그것을 쓴다). 소스 배포본은 첫 실행 때 Rust로 빌드한다(3~5분).
   - 또는 Colab 왼쪽 열쇠(보안 비밀)에 `GITHUB_TOKEN`을 **직접** 등록하면 저장소에서 설치한다.
3. 셀을 실행한다. Drive 연결 허용 창이 뜨면 허용한다.

**결과**: `MyDrive/memopro_colab/results/<셀>/<실행 id>/`
- `summary.md`·`summary.csv`: 비교 표와 읽는 법
- `cases/*.json`: 경우마다 모든 측정값(손실 곡선, 단계 시간, 메모리, memopro 보고, 오류)
- `logs/*.log`: 경우마다 프로세스 출력 전체
- `timelines/*.csv`: 0.5초 간격 GPU·호스트 메모리와 GPU 사용률
- `env.json`: GPU, 판, memopro 빌드 확인
- 같은 이름의 `.zip`: 폴더 전체(공유용)

**끊겼을 때**: 같은 셀을 다시 실행하면 끝난 경우는 건너뛰고 이어서 한다(`NEW_RUN = True`면 새로 시작). 모델은 Drive의 `hf_cache/`에 남아 다시 받지 않는다.

**공정성**: 모든 경우는 **새 프로세스**에서 돈다(앞 경우의 메모리가 섞이지 않게, 0055). 같은 데이터·시드·단계 수를 쓰고, dropout은 끈다. 여러 모델을 **동시에** 한 GPU에서 재면 서로의 측정을 오염시키므로(0085), 여러 모델은 목록으로 차례로 잰다.
"""


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


def main():
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]) if "--out-dir" in sys.argv else None
    cells = [{"cell_type": "markdown", "metadata": {}, "source": INTRO.splitlines(True)}]
    titles = {"train": "## 1. 학습·개발 과정 통합 테스트",
              "infer": "## 2. 로컬 AI 모델 활용 통합 테스트",
              "multi": "## 3. 여러 모델·프로세스 수준 기능 통합 테스트"}
    for cell_file, name in CELLS:
        src = compose(cell_file)
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{name}.py").write_text(src)
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [titles[name]]})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": src.splitlines(True)})
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4",
                                                                        "provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "examples" / "colab_t4_heavy.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(out, [len(c["source"]) for c in cells if c["cell_type"] == "code"])


if __name__ == "__main__":
    main()
