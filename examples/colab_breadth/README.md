# E053: Colab T4에서 메모리를 크게 요구하는 작업 전반 시험

memopro가 **여러 모델 계열(SLM·LLM)·비전 모델·고사양 라이브러리·데이터 처리**에서도 결과를 바꾸지 않고 메모리 상한 안에서 도는지 확인하는 노트북 묶음입니다. 연구 기록: docs/research/0268 (사전 등록).

- **캐시가 없는 새 Colab 런타임을 가정합니다.** memopro는 PyPI에서, 모델은 Hugging Face에서 새로 받습니다. 공개 모델만 쓰므로 토큰이 필요 없습니다.
- **시험(모델)마다 노트북이 하나씩** 있습니다. 서로 독립적이라 순서 상관없이, 무료 Colab에서 하루에 몇 개씩 돌려도 됩니다.
- 각 노트북은 경우마다 따로 된 프로세스를 띄우고, 끝에 **경우별 표와 판정표(PASS/FAIL/report)** 를 출력합니다.

## 실행 방법

1. 아래 표의 링크로 노트북을 엽니다(Colab에서 바로 열림).
2. *런타임 → 런타임 유형 변경 → T4 GPU*를 고릅니다.
3. *런타임 → 모두 실행*을 누릅니다.
4. 결과 JSON은 `/content/memopro_bench/results/`에 남습니다. 설정 셀에서 `SAVE_TO_DRIVE = True`로 바꾸면 Google Drive의 `memopro_e053/`에도 복사됩니다.

## 노트북 목록

| 노트북 | 분류 | 내용 | 예상 시간 | 내려받는 모델 |
|---|---|---|---|---|
| [00_tests](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/00_tests.ipynb) | 시험 묶음 | PyPI 0.1.4 + 같은 태그의 pytest(CUDA 포함) | 약 10분 | - |
| [10_slm_qwen2.5-1.5b](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/10_slm_qwen2.5-1.5b.ipynb) | SLM | Qwen2.5-1.5B (대조군) | 약 15분 | 3GB |
| [11_slm_smollm2-1.7b](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/11_slm_smollm2-1.7b.ipynb) | SLM | SmolLM2-1.7B (Llama 구조) | 약 15분 | 3.5GB |
| [12_slm_phi-3.5-mini](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/12_slm_phi-3.5-mini.ipynb) | SLM | Phi-3.5-mini (합쳐진 qkv_proj) | 약 25분 | 7.6GB |
| [20_llm_mistral-7b](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/20_llm_mistral-7b.ipynb) | LLM | Mistral-7B (GPU·RAM에 그냥은 안 들어감) | 약 40~60분 | 14.5GB |
| [21_llm_qwen2.5-14b](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/21_llm_qwen2.5-14b.ipynb) | LLM | Qwen2.5-14B (디스크에서 흘려 읽음) | 약 70~90분 | 29.5GB |
| [22_llm_qwen2.5-14b_vs_unsloth](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/22_llm_qwen2.5-14b_vs_unsloth.ipynb) | LLM 비교 | 14B 16비트 LoRA: Unsloth 16/4비트 대 memopro (E049) | 약 60~90분 | 39GB |
| [30_vision_dinov2-giant](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/30_vision_dinov2-giant.ipynb) | 비전 | DINOv2-giant 추론 + LoRA 결정성 | 약 15분 | 4.5GB |
| [31_vision_clip-vit-bigg](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/31_vision_clip-vit-bigg.ipynb) | 비전 | CLIP ViT-bigG 비전 탑 | 약 15분 | 10GB |
| [32_vision_siglip-so400m](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/32_vision_siglip-so400m.ipynb) | 비전 | SigLIP so400m | 약 10분 | 3.5GB |
| [33_vision_convnextv2-huge](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/33_vision_convnextv2-huge.ipynb) | 비전 | ConvNeXtV2-Huge (CNN) | 약 10분 | 2.6GB |
| [40_lib_sklearn](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/40_lib_sklearn.ipynb) | 라이브러리 | scikit-learn PCA + 랜덤 포레스트 | 약 10~20분 | - |
| [41_lib_pandas_merge](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/41_lib_pandas_merge.ipynb) | 라이브러리 | pandas 조인·무작위 접근 | 약 10~30분 | - |
| [42_lib_numpy_knn](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/42_lib_numpy_knn.ipynb) | 라이브러리 | NumPy 최근접 이웃 | 약 10~20분 | - |
| [43_lib_xgboost](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/43_lib_xgboost.ipynb) | 라이브러리 | XGBoost (자체 할당기) | 약 10~20분 | - |
| [44_lib_scipy](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/44_lib_scipy.ipynb) | 라이브러리 | SciPy 희소 행렬 + FFT | 약 10~20분 | - |
| [45_lib_polars](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/45_lib_polars.ipynb) | 라이브러리 | Polars (Arrow, 자체 할당기) | 약 5~15분 | - |
| [50_data_image_stack](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/50_data_image_stack.ipynb) | 데이터 | 영상 스택 2GiB (E027) | 약 5~10분 | - |
| [51_data_classify](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/51_data_classify.ipynb) | 데이터 | 희소 분류 (E040) | 약 10분 | - |
| [52_data_dataframe](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/52_data_dataframe.ipynb) | 데이터 | pandas 표 (E040, 무작위 접근) | 약 10~40분 | - |
| [53_data_simulate](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/53_data_simulate.ipynb) | 데이터 | 열 확산 시뮬레이션 (E040, 압축해도 안 들어감) | 약 15~40분 | - |
| [54_data_csv](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/54_data_csv.ipynb) | 데이터 | CSV 쓰기·읽기·정리 | 약 10분 | - |
| [55_data_sort](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/55_data_sort.ipynb) | 데이터 | 3억 개 정렬 | 약 10~20분 | - |
| [56_data_imgprep](https://colab.research.google.com/github/imhyensuk/memopro/blob/main/examples/colab_breadth/notebooks/56_data_imgprep.ipynb) | 데이터 | Pillow 이미지 전처리 | 약 10~20분 | - |

## 판정 기준

| 분류 | 반드시 통과해야 하는 것 |
|---|---|
| SLM | memopro 생성 토큰 = 일반 실행, LoRA 손실이 두 예산에서 비트 단위로 같음, 이 계열에서 동작 |
| LLM | 일반 실행이 못 하는 크기를 memopro가 완주, 두 예산의 토큰·손실이 같음 |
| LLM 비교 | Unsloth 16비트는 14B에서 실패하고 memopro 16비트는 완주(E049 B1) |
| 비전 | 추론 출력이 일반 실행과 비트 단위로 같음 (DINOv2는 LoRA 결정성도) |
| 라이브러리·데이터 | 결과 해시 = 일반 실행, 최대 메모리 ≤ 상한 + 32MiB **또는** 상한을 지킬 수 없을 때 실행 중 경고 + 초과량 보고(0.1.4) |

속도는 모두 보고만 합니다(T4에는 bf16 텐서 코어가 없음, docs/research/0239).

## 소스

노트북은 생성물입니다. 고칠 때는 아래 파일을 고친 뒤 `python examples/colab_breadth/build.py`로 다시 만드세요(`tests/test_colab_breadth.py`가 확인합니다).

- `bench.py`: 설치, 모델 받기, 경우 실행(프로세스 그룹, 시간 제한), 판정표
- `worker.py`: 언어·비전 모델 경우(생성, LoRA, 추론, Unsloth)
- `workloads/`: 라이브러리·데이터 작업(결과 해시와 VmHWM을 JSON 한 줄로 출력)
- `build.py`: 시험 목록과 노트북 생성
