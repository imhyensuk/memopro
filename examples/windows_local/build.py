"""Assemble examples/colab_windows_local.ipynb (E070, docs/research/0364): a guide and one
self-contained code cell (settings + body, with the worker and the CSV generator embedded).

    .venv/bin/python examples/windows_local/build.py [--nb-dir DIR]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "worker_windows.py": HERE / "worker_windows.py",
    "csvgen.py": ROOT / "experiments/e069_validation/csvgen.py",
}
GUIDE = """# memopro E070: Windows PC (CPU) 한계 시험과 비교 (docs/research/0364)

학교 PC 같은 **Windows + CPU만 있는 기계**에서 memopro를 1시간 안에 재고, 같은 작업을 Transformers·llama.cpp·DuckDB·pandas와 비교한다. Colab의 **로컬 런타임**으로 이 PC의 Python에서 돈다(GPU는 쓰지 않음: Intel UHD 그래픽은 PyTorch가 쓰지 못한다).

| 갈래 | 무엇을 보나 |
|---|---|
| 생성 | Qwen2.5-1.5B(가중치 3.1GB)를 프로세스 상한 2GiB·1.5GiB에서 `memopro.generate`: 글이 그냥 올린 모델과 같은가(W1), 최대 메모리가 상한 안인가(W3). llama.cpp 4비트·16비트와 속도 비교(보고) |
| 학습 | Qwen2.5-0.5B 16비트 LoRA를 상한 2·3GiB에서 `memopro.finetune`: 손실이 상한과 무관하게 같은가(W2) |
| 데이터 | 4GB CSV를 1.5GiB 상한에서 `memopro.sort`/`join`: 결과가 DuckDB와 같은가(W4), 디스크에 쓰지 않는가(W5). pandas는 메모리에 다 올림(보고) |
| 압축 | 런타임 무손실 압축의 비율·속도(보고) |

**준비 (처음 한 번)**
1. 이 PC에 Python 3.11~3.13과 Jupyter를 설치한다(`pip install notebook`). 디스크 여유 약 **20GB**(모델 6GB, CSV 4GB, DuckDB 임시 파일 포함).
2. 명령 프롬프트에서 실행한다:
   `jupyter notebook --NotebookApp.allow_origin="https://colab.research.google.com" --port=8888 --NotebookApp.port_retries=0`
3. Colab에서 이 노트북을 열고 **연결 → 로컬 런타임에 연결**을 고른 뒤, 2번이 출력한 주소(토큰 포함)를 붙여 넣는다.
4. 셀을 실행한다. 필요한 패키지(memopro, torch CPU, DuckDB 등)는 셀이 설치한다. 다른 프로그램은 닫아 두는 것이 좋다.

**결과**: `%USERPROFILE%\\memopro_e070\\runs\\<시각>\\summary.md`와 같은 이름의 `.zip`(마지막 줄에 경로가 나온다). zip을 보내 주면 된다.
"""


def main():
    cell = (HERE / "cell_windows.py").read_text(encoding="utf-8")
    workers = {name: path.read_text(encoding="utf-8") for name, path in SOURCES.items()}
    marker = "WORKERS = {}  # @@WORKERS@@"
    assert marker in cell
    cell = cell.replace(marker, "WORKERS = " + repr(workers))
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": GUIDE.splitlines(keepends=True)},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_dir = (
        Path(sys.argv[sys.argv.index("--nb-dir") + 1])
        if "--nb-dir" in sys.argv
        else ROOT / "examples"
    )
    out = nb_dir / "colab_windows_local.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
