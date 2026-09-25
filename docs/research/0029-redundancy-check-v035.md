# 0029. 중복성 검사 3차: 최종 설계 v0.3.5와 실험 결과 주장

- **날짜**: 2026-09-25
- **유형**: survey
- **상태**: 확정 (조사 결과) / 제안 C1~C5 채택(→ 0030)
- **관련 기록**: 0002 (선행 조사), 0011 (D1~D7), 0013 (D8~D13), 0018 (α 기각), 0019 (E005), 0024 (E007), 0025·0027 (Rust 방출 엔진), 0028 (현황)

## 배경 / 동기

> "중복성 검사를 진행해."

마지막 중복성 검사(0013, D8~D13)는 설계 v0.3.1 기준이었다. 그 뒤로 설계와 연구 재료가 크게 바뀌었다.
- α 기각
- β 기본값 변경(무압축 방출)
- Rust 방출 엔진(RS1~RS5)
- Muon 평가
- 실험 결과 주장(E005, E007, E008)

또 2026년에 새 도구와 논문이 나왔다. 이번 검사는 두 대상을 본다.
- **(가) 최종 라이브러리의 기능별 중복**
- **(나) 논문에 쓸 연구 결과 주장의 선행 연구 중복**

## 방법

- 기간: 2026-09-25 하루. 웹 검색 약 15회와 주요 문헌·저장소 원문 확인.
- 기능마다 "같은 문제를 같은 대상(PyTorch 개발자)에게 같은 방식으로 푸는 도구나 논문"이 있는지 찾았다.
- 등급
  - 🔴 거의 같음
  - 🟠 상당 부분 겹침
  - 🟡 개념만 겹침
  - 🟢 조사 범위에서 찾지 못함
- 표현 원칙: "새롭다"가 아니라 "우리 조사 범위에서 찾지 못했다"로 쓴다.

## 결과 (가) — 최종 라이브러리 기능별 중복

| # | 기능 | 새로 찾은 것(굵게) / 기존에 알던 것 | 등급 | 판단 |
|---|---|---|---|---|
| D14 | 이름 `memopro` | PyPI와 crates.io 모두 등록되지 않음(API 응답 404, 2026-09-25 확인) | 🟢 | 사용 가능. 선점은 사용자 확인 후 |
| D15 | `doctor` | llmfit, `accelerate env`, nvitop 등 (0013 D8) | 🟠 | 변동 없음. 기반 기능이며 차별화 요소가 아니다 |
| D16 | `check` (적합 판정·메모리 예측) | **vram-check**(PyPI, 2026-07. PyTorch 모델의 학습·추론 VRAM을 정적 분석. NVIDIA·AMD만 지원, 활성값은 경험식). llmfit, LM Studio, accelerate estimate-memory (0011 D1) | 🟠 → **높아짐** | "PyTorch 모델이 들어가는가"를 답하는 도구가 새로 생겼다. memopro의 차이는 네 가지이다. ① MPS·통합 메모리 한도와 cgroup ② 장치·호스트·디스크 풀별 판정 ③ 작은 배치 시험 실행으로 실측 외삽 ④ 판정 결과를 바로 구성으로 적용. 신규성은 주장하지 않는다 |
| D17 | `load`·`optimize` (추론 구성 자동 선택) | HF `device_map="auto"` + 양자화 설정, torchao autoquant, llama.cpp `--fit` (0011 D2, 0013 D13) | 🟠 | 변동 없음 |
| D18 | `train_session` (학습 구성 자동 선택) | **ProTrain**(arXiv 2406.08334. 모델·옵티마이저만 감싸면 런타임 프로파일로 메모리 관리 전략을 자동 선택, 학습 루프 수정 없음). **AutoCheckpoint**(GitHub, 초기 단계. 메모리 목표에 맞춰 체크포인팅할 층 선택). torchtune 메모리 최적화 모음, Unsloth (0011 D3) | 🟠 → **높아짐** | ProTrain은 "감싸면 자동으로 맞춰 준다"는 접근이 같다. 차이는 세 가지이다. ① CUDA 대규모 LLM 학습 대상 대 memopro는 MPS·소형 환경 포함 ② 기존 기법 연동과 충실도 3분류 ③ 실패 시 대체와 실측 보고. **관련 연구에 반드시 인용**하고 신규성은 주장하지 않는다 |
| D19 | **β 노트북 유휴 텐서 동면** | **TensorNVMe**(CPU 텐서 ↔ NVMe 비동기 I/O, Linux 전용), **DeepNVMe**(DeepSpeed), vDNN·SwapAdvisor·Capuchin(학습 그래프 안의 자동 스와핑), ipyexperiments(셀 단위 변수 해제), JupyterLab 유휴 커널 정리, pytorch_memlab CPU 이동 (0013 D11) | 🟢 유지 | 두 가지를 모두 하는 도구는 **찾지 못했다**. ① 노트북·REPL 네임스페이스에서 유휴 텐서·모듈을 참조 그래프로 안전하게 골라 방출 ② 다시 접근할 때 투명 복원. 스와핑 시스템은 학습 계산 그래프 안의 텐서를 다루고, 노트북 도구는 변수를 **삭제**만 한다 |
| D20 | **Rust 방출 엔진** (RS1~RS5) | **TensorNVMe**(비동기, 동기·배치 API, Linux 전용, libaio·io_uring), **DeepNVMe**, kvikio(GPU Direct Storage), Colossal-AI의 읽기·계산·방출 파이프라인 | 🔴 개념 | "텐서를 SSD로 비동기·파이프라인 방출"은 이미 있는 기술이다. **신규성 없음.** 존재 이유는 세 가지이다. ① macOS 지원(기존 도구는 Linux 전용) ② RS4 메모리 상한 보장 ③ β와의 결합. → 제안 C2 |
| D21 | census 빠른 모드 (범주별 중복도) | 가중치·체크포인트·KV 캐시의 무손실 압축성 분석은 논문에 있다: ZipNN, DFloat11, **Heilper & Singer 2025**(arXiv 2508.19263, 가중치·체크포인트·KV만), LEXI (0013 D10) | 🟢 유지 | **학습 중 상태(그래디언트·옵티마이저 모멘트·저장 활성값)**를 범주별로 측정하는 **도구**는 찾지 못했다. Heilper & Singer도 학습 중 텐서는 다루지 않는다 |
| D22 | census 정밀 모드 (민감 비트) | **AIMET QuantAnalyzer**(층별 양자화 민감도), NVIDIA pytorch-quantization, Intel Neural Compressor, HAWQ 계열 | 🟠 **새로 확인** | "어느 층이 양자화에 민감한가"는 기존 도구가 추론 가중치·활성값에 대해 이미 제공한다. memopro의 차이는 대상과 목적에 있다. **학습 상태의 메모리 조사**의 일부로, 범주별 "필요한 비트 대비 실제 저장 비트"를 본다. → 제안 C3 |
| D23 | **γ OS 압박 탄력 런타임** | Tri-Accel(VRAM 피드백 배치 조절), 배치 적응 알고리즘, **SiliconBench**(2609.19169, 2026-09. Apple Silicon 서빙 엔진 9종 벤치마크) | 🟡 유지 | OS 메모리 압박 신호(macOS memory pressure, Linux PSI)로 PyTorch 실행 구성을 바꾸는 런타임은 찾지 못했다. SiliconBench는 "명시한 메모리 예산이 여유 공간을 보장하지 않는다"는 문제를 확인했을 뿐 해결책은 제안하지 않았다. γ의 동기를 뒷받침하는 근거로 인용한다 |
| D24 | `memopro run` (코드 수정 없는 실행) | ProTrain(감싸기 필요), HF `from_pretrained` 옵션 | 🟡 | 스크립트를 수정 없이 감싸 메모리 기능을 적용하는 명령은 찾지 못했다. 검색이 어려운 영역이므로 v0.3 착수 전에 다시 조사한다 |

## 결과 (나) — 연구 결과 주장의 선행 연구 중복

| # | 주장 (기록) | 선행 연구 | 등급 | 영향 |
|---|---|---|---|---|
| D25 | **α 기각**: 사전학습 GPT-2 블록은 비수축(σ 17~139)이고, 고정점 역산이 실패한다 (0018) | **Kim, Papamakarios & Mnih 2021**(ICML): 내적 셀프 어텐션은 입력 영역이 무한하면 **Lipschitz가 아님**을 증명. i-ResNet(2019): 고정점 역산에는 잔차 함수의 Lipschitz < 1이 필요. **No Free Swap**(2605.16234, 2026): 사전학습 GPT-2-Medium·Pythia-410M 층 전체 사상의 야코비안 노름을 측정했고 깊은 층이 확장적(부록 O) | 🟠 **새로 확인** | α의 전제(Pre-LN이면 수축적)는 **기존 이론으로 이미 의심할 수 있었다.** 0006의 신규성·타당성 검토가 Kim 2021을 놓쳤다. 부정적 결과의 기여는 좁아진다: ① 사전학습 모델 잔차 가지의 실측값 ② 힌트 교정이 오차를 **키운다**는 체인 누적 현상. → 제안 C1 |
| D26 | E005: fp32 학습 텐서 무손실 압축 약 1.16배(≈14%) (0019) | **ZipNN**: fp32 **모델 가중치**는 보통 17% 절감, bf16은 33%. 일반 무손실 압축기는 가수 비트의 무작위성 때문에 부동소수를 크게 줄이지 못한다는 보고는 오래되었다(DeepSZ 등) | 🟠 가중치 / 🟢 학습 상태 | 가중치에 대한 수치는 **기존 보고와 일치**하므로 새 발견이 아니다. 새로운 부분은 **학습 중 상태 전체를 범주별로** 측정하고 **OS 방식(16KB 페이지 LZ4)**과 비교한 것이다. 논문에서는 "기존 가중치 결과를 학습 상태로 확장"으로 쓴다 |
| D27 | E005: OS 방식 페이지 압축은 fp32 텐서에 무효(1.00배) | 부동소수에 범용 압축이 약하다는 것은 알려져 있다. 학습 텐서에 macOS·Linux 메모리 압축(zram 등) 방식을 직접 측정한 보고는 찾지 못했다 | 🟡 | 결과 자체는 예상 범위 안이다. 측정 보고로서의 가치는 작지만 β 설계 근거로는 유효하다 |
| D28 | E007: Adam 사전학습 모델의 Muon 전체 파인튜닝이 열세 (0024) | **"Can Muon Fine-tune Adam-Pretrained Models?"**(2605.10468, 2026)가 같은 현상과 원인(옵티마이저 불일치), LoRA 완화를 보고. 0024에서 이미 인용했다 | 🔴 | 새 주장 아님(0024에서도 "문헌과 같은 방향"으로 기록). 우리 쪽의 새 관찰은 **M1 MPS에서 bf16 Newton–Schulz 오버헤드가 소배치일수록 커진다**는 공학적 사실뿐이다 |
| D29 | E008: 단순 Rust는 Python 스레드보다 느리고, 설계한 Rust는 1.49배 (0025) | 일반적인 성능 공학 지식 | — | 논문 기여로 주장하지 않는다. System 절의 설계 근거로만 쓴다 |

## 결론

1. **memopro 고유 영역은 세 가지로 유지된다.** 조사 범위에서 찾지 못한 것은 다음과 같다.
   - β: 노트북 유휴 텐서를 안전하게 골라 방출하고 투명하게 복원
   - census: 학습 상태의 범주별 중복도 측정 도구
   - γ: OS 압박 신호에 따른 PyTorch 실행 구성 변경
2. **나머지는 공학적 통합이다.** doctor, check, load·optimize, train_session, Rust 방출 엔진은 모두 선행 도구가 있다. 그중 check(vram-check)와 train_session(ProTrain)은 **0013보다 중복도가 높아졌다.** 가치는 신규성이 아니라 다섯 가지에 있다.
   - 통합
   - 소형·통합 메모리 환경 지원
   - 실패 시 대체
   - 충실도 3분류
   - 실측 보고
3. **연구 주장은 좁혀야 한다.**
   - α 부정적 결과는 기존 이론(Kim 2021)으로 예측할 수 있었다.
   - E005 가중치 수치는 ZipNN과 일치한다.
   - E007 파인튜닝 결과는 기존 논문과 같다.
   - 남는 연구 기여는 **학습 상태의 범주별 중복·정밀도 측정(센서스 연구)**이다. 이는 0021 Q2 권고(센서스 연구를 주력으로)를 뒷받침한다.
4. 이름 `memopro`는 두 레지스트리 모두에서 사용할 수 있다.

## 제안 (사용자 승인 필요)

| # | 제안 | 이유 |
|---|---|---|
| C1 | **α 부정적 결과의 주장 범위 축소**: 단독 논문이 아니라 센서스 논문의 Discussion 한 절로 둔다. Kim 2021·i-ResNet·No Free Swap을 인용하고 "이론이 예측하는 비수축성을 사전학습 모델에서 실측했으며, 힌트 교정이 오차를 키우는 체인 누적을 확인했다"로 쓴다. 0018 상태 줄에 이 기록을 표시한다(본문 보존) | D25 |
| C2 | **E009에 기존 도구 기준선을 추가**한다. macOS에서는 `torch.save`·`numpy.tofile`, Linux에서는 TensorNVMe를 기준으로 삼는다. 방출 엔진의 존재 이유(macOS, 메모리 상한, β 결합)를 설계 문서에 명시한다. Linux에서 TensorNVMe를 선택 백엔드로 연동할지는 E009 결과를 보고 판단한다 | D20, 재구현 금지 원칙(0010) |
| C3 | **census 정밀 모드를 "양자화 민감도 분석"과 구분**해 정의한다. 추론 층별 민감도(AIMET 등 기존 도구 영역)가 아니라 **학습 상태 범주별로 필요한 비트 대비 실제 저장 비트**를 본다. 관련 연구에 AIMET·HAWQ를 추가한다 (0021 Q1·0022 R2와 함께 결정) | D22 |
| C4 | **check와 train_session의 비교 대상을 명시**한다. vram-check·ProTrain·AutoCheckpoint를 관련 연구와 README "기존 도구와의 차이" 절에 넣고, A2 완료 조건에 "같은 과제에서 기존 도구와 비교"를 추가한다 | D16, D18 |
| C5 | **`memopro run`의 중복 재조사**를 v0.3 착수 조건에 넣는다 | D24 검색 한계 |

## 한계

- 하루 동안의 웹 검색이며, 영어 자료 위주이다. 비공개·사내 도구와 최근 몇 주 안에 나온 자료는 빠졌을 수 있다.
- 일부 문헌은 초록과 공개 페이지만 확인했다(No Free Swap 부록 O의 수치, ProTrain 본문). 논문에 쓰기 전에 원문을 다시 확인한다.
- "찾지 못했다"는 존재하지 않는다는 뜻이 아니다.

## 논문 매핑

- **Related Work**: D16·D18·D19·D20·D22·D23의 도구를 범주별로 정리한다(적합 판정, 자동 학습 구성, 텐서 스와핑·NVMe 방출, 양자화 민감도, 메모리 압박 적응).
- **Discussion (α 절)**: D25. "이론으로 예측 가능했던 실패를 사전 등록 실험으로 확인한 사례"라는 방법론적 교훈을 쓴다. 신규성 검토에서 선행 이론을 놓친 점도 투명하게 기술한다.
- **센서스 논문 Motivation**: D21·D26. 가중치 무손실 압축성은 알려져 있으나(ZipNN), 학습 상태 전체의 범주별 측정은 조사 범위에서 찾지 못했다.

## 출처 (URL)

- vram-check: https://pypi.org/project/vram-check/
- ProTrain: https://arxiv.org/abs/2406.08334
- AutoCheckpoint: https://github.com/archakamk/autocheckpoint
- torchtune 메모리 최적화: https://meta-pytorch.org/torchtune/stable/tutorials/memory_optimizations.html
- TensorNVMe: https://github.com/hpcaitech/TensorNVMe · Colossal-AI NVMe offload: https://colossalai.org/docs/features/nvme_offload/
- DeepNVMe: https://www.deepspeed.ai/tutorials/deepnvme/
- Capuchin (ASPLOS 2020): https://dl.acm.org/doi/10.1145/3373376.3378505 · SwapAdvisor (ASPLOS 2020): https://dl.acm.org/doi/10.1145/3373376.3378530
- ipyexperiments: https://pypi.org/project/ipyexperiments/
- ZipNN: https://arxiv.org/abs/2411.05239 · Heilper & Singer: https://arxiv.org/abs/2508.19263
- AIMET QuantAnalyzer: https://quic.github.io/aimet-pages/releases/1.33.5/user_guide/quant_analyzer.html
- Tri-Accel: https://arxiv.org/abs/2508.16905 · SiliconBench: https://arxiv.org/abs/2609.19169
- Kim et al. 2021: https://arxiv.org/abs/2006.04710 · i-ResNet: https://proceedings.mlr.press/v97/behrmann19a/behrmann19a.pdf · No Free Swap: https://arxiv.org/abs/2605.16234
- Muon 파인튜닝 불일치: https://arxiv.org/abs/2605.10468 · Muon is Scalable: https://arxiv.org/abs/2502.16982

## 참고문헌

- [@kim2021lipschitz] [@garcia2026nofreeswap] [@heilper2025lossless] [@protrain2024] [@vramcheck] [@autocheckpoint] [@tensornvme] [@deepnvme] [@capuchin2020] [@swapadvisor2020] [@aimet] [@siliconbench2026] [@muonft2026] [@liu2025muon] — references.bib
