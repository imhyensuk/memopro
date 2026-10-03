# 0139. G4 E4 제작: 상주 int4 초안 + 흘려 쓰는 bf16 검증 (HF 보조 생성 감싸기)

- **날짜**: 2026-10-03
- **유형**: implementation (+ 선행 확인, 개발 확인)
- **상태**: 확정. E033(0140)으로 검증
- **관련 기록**: 0125 (G4 설계 E4), 0127 (오진법 조사: int4 초안이 유력, E032 1등 일치 72.5%), 0069·0085 (MPS int4pack), 0130·0136 (MPS 무복사 스트리밍), 0138 (G4-B1 통과)

## 선행 확인 (신규성 규칙)

| 선행 | 내용 | E4와의 관계 |
|---|---|---|
| 추측 디코딩 (Leviathan 외 2023, Chen 외 2023) | 작은 초안이 k토큰을 제안하고 큰 모델이 한 번에 검증한다. 탐욕이면 출력이 큰 모델 단독과 같다 | E4의 기본 원리 그대로 |
| HF 보조 생성(`generate(assistant_model=...)`, Gante 2023) | 위 방법의 공개 구현. 초안 길이를 수용 결과에 따라 조절하고, 초안 확신도가 낮으면 일찍 멈춘다 | **감싸서 쓴다**(0010: 다시 구현하지 않음) |
| SubSpec(arXiv 2509.18344), ML-SpecQD(arXiv 2503.13565) | 오프로딩된 큰 모델의 저비트 사본을 초안으로 쓰는 무손실 추측 디코딩 | 같은 생각. 0127에 기록 |
| QuantSpec(arXiv 2502.10424) | 양자화 KV 캐시로 자기 추측 | 가까움 |

- **결론**: E4는 기존 기법의 결합이다. **신규성 주장은 없다.**
- memopro의 몫은 두 가지를 잇는 것이다.
  - 흘려 쓰는 무손실 bf16 대상(런타임, 예산 상한, 쓰기 없음).
  - 장치에 상주하는 int4 초안(torch int4pack).
- 흘려 쓰는 대상은 한 번 검증할 때마다 가중치 전체를 읽어야 한다. 그래서 검증 한 번당 토큰 수가 곧 속도 배수가 된다.

## 제작

- `memopro.rt.torch.draft_model(name_or_dir, *, target=None, device="mps", revision=None)`
  - HF 모델을 CPU에 bf16으로 메모리 매핑해 읽는다. `int4pack.convert`로 선형층을 int4(그룹 32)로 바꿔 MPS에 둔다.
  - 임베딩, 정규화, 출력 머리는 bf16으로 남긴다.
  - 크기(저장소 단위, 중복 없이)는 `memopro_draft_bytes`다. **스트리밍 예산 밖의 장치 메모리**라고 문서에 적었다.
  - `target`을 주면 어휘 크기가 같은지 확인한다. 다르면 `InvalidArgument`를 낸다(토크나이저를 공유해야 함).
  - MPS가 아니면 `ModeUnavailable`을 내고, 대안으로 "작은 모델을 그냥 불러 assistant_model로"를 알려 준다.
- 사용법: `model.generate(..., do_sample=False, assistant_model=draft)`. 탐욕 검증은 HF가 한다. 남는 토큰은 언제나 흘려 쓰는 모델이 고른 것이다.
- **시험**(`tests/test_rt_torch.py`)
  - `test_int4_draft_keeps_greedy_generation_of_the_streamed_model`: 작은 Qwen2(bf16, 약 12MB, 예산 9MiB, MPS)에서 자기 초안과 다른 초안 둘 다 보조 생성 결과가 단독 탐욕 생성과 같다. 어휘가 다르면 거부한다.
  - `test_int4_drafts_need_an_apple_gpu`.
- `references.bib`
  - 0127에서 `subspec2025`·`mlspecqd2025`가 같은 키로 두 번 들어간 것을 발견해 뒤의 중복을 지웠다.
  - `leviathan2023speculative`, `chen2023speculative`, `gante2023assisted`를 추가했다.

## 개발 확인 (결과 아님, 스크래치 실행)

대상은 Qwen2.5-3B bf16을 MPS로 흘려 썼다(예산 1GiB). 프롬프트 하나("hash table collisions")로 확인했다.

| 초안 | 초안 크기(MPS 할당) | 변환 시간 | 대상 단독 | 보조 생성 32토큰 | 최대 footprint 증가 |
|---|---|---|---|---|---|
| Qwen2.5-3B int4(자기 초안) | 2,267MiB | 17.1초 | 3.99초/토큰(8토큰) | 1.48초/토큰 | 3,606MiB |
| Qwen2.5-1.5B int4 | 1,249MiB | 7.7초 | 4.21초/토큰(8토큰) | **0.89초/토큰** | 2,488MiB |

- 두 경우 모두 앞 8토큰이 대상 단독과 같았다.
- 작은 초안이 더 빠르다. 초안이 싸야 초안을 길게 쓸 수 있고, 검증 비용은 초안 길이와 거의 무관하다(가중치 읽기가 주도).
- HF 기본값(`GenerationConfig`)은 보조 토큰 수·일정·확신도 문턱이 비어 있다. 실제 값은 초안 모델의 생성 설정에서 온다. memopro는 바꾸지 않는다.

## 논문 매핑

- **System (G4 E4)**: 흘려 쓰는 무손실 대상 + 상주 저비트 초안. 검증 한 번에 가중치를 한 번 읽어 여러 토큰을 낸다.
- **Related Work**: 추측 디코딩, HF 보조 생성, SubSpec, ML-SpecQD, QuantSpec. 신규성 주장 없음.
