# DMFC 공개 이름과 익명 심사본 변경

> Historical revision record (2026-10-20). Names and archive filenames below describe that revision. The current method and library are **DMGFC** / `dmgfc.DMGFC`, with `dmgfc-eval` as the current command. See the [current naming guide](dmgfc-naming-2026-10-24.md) and [API migration guide](dmgfc-public-api-migration-2026-10-24.md).

2026년 10월 20일 추가 요청에 따라 논문과 공개 라이브러리의 이름을 **Decision-Model-Guided Function Calling (DMFC)**로 통일했습니다. 이 변경은 새로운 검증 모델 비교 실험을 뜻하지 않습니다. 실제 검증 모델은 Jev 1.13.0 하나이며, 생성 모델 비교와 구분합니다.

| 이전 공개 이름·형식 | 현재 이름·형식 | 확인한 동작 |
|---|---|---|
| `import jbfc`, `jbfc.JBFC` | `import dmfc`, `dmfc.DMFC` | 기존 import와 하위 모듈은 같은 구현을 가리키는 호환 경로로 유지합니다. |
| 설치 메타데이터 `jev-backed-function-calling` | `dmfc` | 저장소에서 설치하는 패키지와 배포 파일에 `dmfc`, `jbfc`, `jev_fc`를 함께 포함합니다. |
| `jbfc`, `jbfc-latest`, `jev-function`, `jev-arguments` | `dmfc`, `dmfc-latest`, `dmfc-f`, `dmfc-a` | 이전 옵션도 받습니다. `Result.method`와 표시 문자열은 현재 DMFC 이름을 반환합니다. |
| `jev-fc` 평가 명령 | `dmfc-eval` 별칭 추가 | 두 명령은 같은 기존 실험 실행기를 사용합니다. 실험 옵션·기록 형식은 그대로입니다. |
| README·그림·표 생성기의 JBFC 표시 | DMFC·DMFC-F/A/L 표시 | 원고를 다시 생성할 때도 현재 이름을 사용합니다. 실제 Jev 모델·서비스 이름은 유지합니다. |
| 저자 표시 ACM 원고 | 익명 `sigconf,anonymous,review` 원고 | 저자 정보·감사의 글을 제외하고 참고문헌 포함 총 4쪽입니다. 이전 원고는 별도 버전으로 보존합니다. |

기존 JSON 키와 `jev_joint`, `jev_joint_latest`, `jev_function`, `jev_arguments` 실험 ID, 저장 경로는 기록 대조를 위해 보존했습니다. `src/jev_fc/`의 11개 실험 코드 파일은 이전 커밋과 바이트 단위로 같습니다. 수정 한도 3회, 임계값 0.5, 예산 소진 시 최초 후보 유지, no-call 인자 검사 생략, 누적 거절 피드백의 순서도 유지했습니다.

전체 오프라인 테스트 **768개**와 코드 검사를 통과했습니다. 새·기존 API의 동일 구현, 함수 메타데이터 호환, F/A/L 매핑, 기본 검증기와 수락·예산 소진 동작을 검사했고, 독립 설치한 배포 파일에서도 세 import 경로를 확인했습니다. 원고의 실험 절·참고문헌·그림 소스와 측정 수치는 이전 저자 표시 버전과 같습니다. 익명화를 위한 `main.tex`와 첫 페이지 트랙 관련성을 위한 `sec/introduction.tex`만 이 단계에서 다시 바꿨습니다.

Overleaf에 업로드할 필수 10개 파일은 [업로드 체크리스트](overleaf-upload-checklist-2026-10-20.md)에 있습니다. 제공된 `acmart.cls`와 `ACM-Reference-Format.bst`는 최초 ZIP과 바이트 단위로 같으며, 업로드용 ZIP만으로 컴파일한 네 페이지가 저장소 PDF와 일치합니다. 현재 트랙 문구는 연구 내용에 가까운 **Web Infrastructure and Agentic Systems**를 작업 기준으로 사용했습니다.
