# 0213. CITATION.cff 추가

- **날짜**: 2026-10-06
- **유형**: decision (사용자 지시) + implementation
- **상태**: 확정
- **관련 기록**: 0212 (공개와 라이선스 검토)

## 내용

- 사용자 지시: "CITATION.cff 추가하고 깃허브 공개를 진행해." (2026-10-06). 공개는 0212에서 이미 끝났다(`imhyensuk/memopro` PUBLIC). 라이선스는 MIT OR Apache-2.0 그대로(0212의 제안 A).
- 저장소 뿌리에 `CITATION.cff`(CFF 1.2.0): 제목 memopro, 저자 임현석(한신대학교 공공인재빅데이터융합학), 버전 0.1.0a1, 라이선스 MIT·Apache-2.0, 저장소 주소, 핵심어. 영문 이름 표기는 사용자 확인 전이라 한글로만 넣었다.
- 검증: `cffconvert --validate` → "valid according to schema version 1.2.0". GitHub 저장소 화면에 "Cite this repository"가 생긴다.

## 논문 매핑

- 각 논문의 **Availability** 문단과 소프트웨어 인용에 이 메타데이터를 쓴다.
