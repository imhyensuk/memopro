# 0013. 수정안 v0.3 검증: 중복성과 논리성

- **날짜**: 2026-09-25
- **유형**: survey + decision
- **상태**: 확정 — 수정 사항을 설계 v0.3.1에 반영
- **관련 기록**: 0012 (검증 대상), 0011 (이전 검증)

## 방법

- 중복성: v0.1 구성 요소(doctor, census, β, codec)와 v0.2 접근 계층을 S1 대상(PyTorch 개발자) 기준으로 재조사 (2026-09-25).
- 논리성: 목표-범위 정합성, v0.1 구성 요소의 기술적 실현성, 원칙 간 충돌을 점검.

## 결과 1 — 중복성

| # | 구성 요소 | 기존 도구 | 중복도 | 판단 |
|---|---|---|---|---|
| D8 | **doctor** (환경·풀별 예산) | 하드웨어 감지: llmfit, nvidia-smi·nvitop·gpustat, `torch.utils.collect_env`, `accelerate env`. 컨테이너 인식 메모리: PyTorch(테스트용, PR #198327), SGLang, tylertoo가 **각자 따로 구현**. Rust `sysinfo` 크레이트가 cgroup 한도 제공 | 🟠 부분별로 높음 | 전용 라이브러리는 없음. "PyTorch용 풀별 예산 벡터(MPS 권장 한도 + cgroup + 디스크)"라는 조합은 미발견이지만 **신규성은 낮은 유틸리티**. 차별화 요소가 아니라 기반 기능으로 취급 |
| D9 | **census 기본** (메모리가 어디에 쓰이나) | PyTorch Profiler(`profile_memory`), 메모리 스냅샷 시각화, pytorch_memlab, memray, Scalene(OSDI 2023), IPyExperiments(셀 단위) | 🔴 높음 | 기존 도구와 같은 수준이면 가치가 없음 |
| D10 | **census 중복도** (실제 엔트로피, 그래디언트 민감 비트) | 논문 속 분석으로만 존재: LEXI(가중치·활성값·캐시의 비트 수준 엔트로피 분석), ZipNN·DFloat11의 압축성 분석 | 🟢 도구로는 미발견 | **census의 차별점은 중복도 측정**. 기본 기능은 기존 도구 수준을 따라가되 중복도를 핵심으로 둔다 |
| D11 | **β** | Jupyter 유휴 변수 자동 압축·방출 확장은 **미발견**(커널 정리, 변수 조회 확장만 존재). 부분 유사: **pytorch_memlab의 "모든 CUDA 텐서를 CPU로 일시 이동"**(수동, 압축·방출 없음), Dask·Ray 방출(프레임워크 관리 객체 한정) | 🟢 미발견 | 유지. 관련 연구에 pytorch_memlab 추가 |
| D12 | **codec** (바이트 셔플 + 엔트로피 부호화) | blosc2(셔플 + zstd), ZipNN, zstd | 🔴 높음 | 재구현하면 0010의 **재구현 금지 원칙 위반** → 수정 V11 |
| D13 | **v0.2 접근 계층** (PyTorch 개발자 대상) | HF `device_map="auto"` + 양자화 설정, Accelerate 대형 모델 추론, torchao `autoquant`(층별 양자화 자동 선택, 속도 기준), Unsloth(LLM 파인튜닝) | 🟠 중간~높음 | 공학적 가치(여러 백엔드의 예산 기반 자동 선택, HF 외 모델 지원, 통합 보고)로 유지. 신규성 주장 안 함 |

**종합**: v0.1의 고유 핵심은 **β**와 **census 중복도**이다. doctor는 필수 기반 유틸리티이다. codec은 기존 라이브러리 위에 얇게 구현해야 한다.

## 결과 2 — 논리성

| # | 심각도 | 문제 | 수정 (v0.3.1 반영) |
|---|---|---|---|
| V1 | 🟠 | S1으로 대상이 좁아졌는데 궁극적 목표 문구("사용자 누구나")가 그대로 남아 있다 | 목표 문구를 "**PyTorch 개발자** 누구나"로 정정하고 README 표어도 수정 |
| V2 | 🟠 | v0.1(doctor + census + β)은 "큰 모델을 돌리게 한다"를 **직접** 달성하지 않는다 | 사용자 결정(S3)이므로 유지. 대신 v0.1의 가치를 정직하게 표기: "개발 단계 메모리 회수 + 낭비 진단". README 예시에 버전 표기(`load`, `train_session`은 v0.2) |
| V3 | 🟡 | S2 기각에 따라 U6의 외부 도구 안내 문구가 남아 있다 | 삭제 |
| V4 | 🟠 | β의 참조 수 검사가 IPython 출력 캐시(`Out[n]`, `_`, `__`, `___`)의 참조 때문에 실패한다 | 출력 캐시 참조는 "네임스페이스 참조"로 함께 집계하고, 동면 시 캐시 항목도 프록시로 교체 |
| V5 | 🔴 | 옵티마이저가 모델 파라미터를 참조한다. 모델만 동면하면 옵티마이저가 빈 파라미터를 가리키게 된다 | `gc.get_referrers`로 외부 참조자를 찾는다. 참조자가 옵티마이저뿐이면 **모델 + 옵티마이저를 한 묶음**으로, 둘 다 유휴일 때만 동면한다. 그 외 참조자가 있으면 동면하지 않는다 |
| V6 | 🔴 | 동면 과정 자체가 메모리를 늘린다(장치 → 호스트 복사, 압축 버퍼). 메모리가 빠듯할 때 오히려 OOM을 부를 수 있다 | **청크 단위 스트리밍 압축**(버퍼 상한, 기본 64MB). 여유가 버퍼보다 작으면 압축 없이 디스크 방출을 우선 |
| V7 | 🔴 | 텐서를 해제해도 메모리가 OS로 돌아간다는 보장이 없다. PyTorch 캐싱 할당자(MPS·CUDA)와 malloc이 보유할 수 있다 | 해제 후 `empty_cache()`를 호출한다. 절감량은 논리 바이트가 아니라 **실측 회수량**(프로세스 RSS, `torch.mps.driver_allocated_memory()` 등)으로 보고한다. 완료 조건도 실측 기준으로 한다 |
| V8 | 🟡 | 프록시의 투명성 한계: `isinstance`, `id()`, C 확장에 직접 넘기는 경우 | `__torch_function__`과 속성 접근 시 복원. 한계를 문서화하고, 기본값은 보수적으로(tensor와 `nn.Module`만) |
| V9 | 🟠 | census가 `saved_tensors_hooks`만 쓰면 `no_grad` 추론 텐서(KV 캐시 등), 할당자 캐시, 런타임 컨텍스트를 놓친다 | 할당자 통계와 대조해 **"미분류" 항목**을 명시한다. 목표: 할당자 바이트의 90% 이상을 분류 |
| V10 | 🟠 | 그래디언트 민감 비트 측정은 비용이 크다(텐서 수 × 비트 수 × 역전파) | census를 **2단계**로: 빠른 모드(기본, 표본 엔트로피·생애 주기) / 정밀 모드(선택, 표본 텐서에 교란 실험) |
| V11 | 🟠 | codec 재구현은 재구현 금지 원칙과 충돌한다 | codec은 기존 크레이트(zstd) + 바이트 셔플만 얇게 구현한다. 신규성 주장 없음. 성능이 부족하면 blosc2 연동을 검토 |
| V12 | 🟡 | torch 없는 기본 설치에서 doctor는 MPS 권장 한도를 알 수 없다(Metal API는 torch를 통해 조회) | 기본 설치: OS 수준 풀(RAM, cgroup, 디스크)만 보고하고 "MPS 한도: torch 설치 필요"로 표시. `[torch]` 설치 시 `torch.mps.recommended_max_memory()` 사용 |
| V13 | 🟡 | v0.1 Rust 크레이트(hwinfo, codec, ledger, spill)는 모두 기존 크레이트를 얇게 감싼 것이라 단독 가치가 작다 | 정직하게 인정한다. crates.io 배포는 최종 목표와 이름 확보를 위한 것이며, 크레이트 설명에 범위를 명시한다 |
| V14 | 🟢 | `pressure`(OS 메모리 압박)가 v0.1에 필요한가 | β는 유휴 기준으로 동작하면 충분하다 → `pressure`는 v0.4(γ)로 이동 |
| V15 | 🟢 | v0.2 추론 완료 조건이 비어 있다(0011 L4에서 폐기) | 초안: M1 8GB에서 **기본 fp32 로드 시 MPS 한도를 넘는 3B급 HF 모델**을 `optimize`/`load`로 예산 안에서 실행. 이미지 생성 파이프라인은 선택. v0.2 착수 전 확정 |

## 결론

- 수정안 v0.3은 **목표-범위 정합성(V1~V3)**을 문서 수정으로 해결할 수 있다.
- **β에 치명 3건(V5~V7)**이 있다. 모두 설계 수정으로 해결 가능하며, v0.1 완료 조건에 반영한다.
- census는 기본 기능이 기존 도구와 겹치므로 **중복도 측정을 핵심 기능**으로 둔다.
- 설계 문서를 v0.3.1로 갱신한다.

## 논문 매핑

- **System**: β 설계(참조 그래프 기반 동면 대상 선정, 청크 스트리밍, 실측 회수량 보고)는 시스템 논문의 설계 절 재료이다.
- **Related Work**: D8~D12 (프로파일러, 컨테이너 인식 메모리, pytorch_memlab, blosc2·ZipNN).

## 출처 (URL)

- PyTorch Profiler: https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html · 메모리 시각화: https://huggingface.co/blog/train_memory
- pytorch_memlab: https://github.com/Stonesjtu/pytorch_memlab · memray: https://bloomberg.github.io/memray/index.html · Scalene: https://github.com/plasma-umass/scalene
- LEXI(비트 수준 엔트로피 분석): https://arxiv.org/abs/2603.15589
- Jupyter 확장 현황: https://github.com/jupyterlab-contrib/jupyterlab-variableinspector , https://pub.towardsai.net/your-jupyterlab-is-hoarding-dead-sessions-heres-how-i-fixed-it-a7543bba6def
- 컨테이너 인식 메모리: https://github.com/pytorch/pytorch/pull/198327 , https://github.com/sgl-project/sglang/pull/40066 , https://github.com/geoparquet-io/tylertoo/pull/485 , https://github.com/dask/distributed/issues/1712
- PyTorch MPS 메모리 API: https://docs.pytorch.org/docs/stable/generated/torch.mps.recommended_max_memory.html , https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html
