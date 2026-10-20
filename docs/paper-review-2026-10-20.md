# 논문 피드백 후속 액션

확인 기준: 2026년 10월 20일. 이번 수정에서는 요청대로 **초록과 서론을 유지**합니다. 아래 항목은 추가 실험이나 저자 검토가 필요한 후속 작업이며, 새 실험을 수행했다는 의미는 아닙니다.

## 이번에 반영한 내용

초록과 서론은 수정 전 원문과 바이트 단위로 일치합니다. 나머지 부분은 관련 문헌을 Methodology 초반에 통합하고, 별도 Related Work 절과 중복 Final Proposal 설명을 정리했습니다. 첫 페이지에 기존 방식과 DMFC의 비교 그림을 배치했고, 주 실험과 ablation 표를 분리했습니다. 저장된 실제 실행 기록의 정확도·평균 API 시간·p95를 이용한 line plot을 추가했습니다. 새 모델 실험이나 실행 결과는 만들지 않았습니다. 본문 3쪽과 참고문헌 1쪽으로 총 4쪽입니다.

DMFC는 일반 방법 이름이고, 현재 실험에서 사용한 검증 모델은 jev-1.13.0 하나임을 구분했습니다. F는 함수 판정만, A는 인자 판정만 피드백과 수락 조건에 사용하는 실험이며, A의 no-call은 schema 검사만으로 수락됩니다. L은 최근 거절 기록 하나만 보며, 전체 DMFC는 모든 기록을 봅니다. 둘 다 최초 후보·임계값·수정 예산은 같습니다.

## 우선순위

| 우선순위 | 다음 작업 | 완료 기준과 근거 |
|---|---|---|
| 1 | **검증 모델 계열 비교** | 생성 모델을 고정하고 검증 역할만 바꿔 비교합니다. 현재 결과의 검증 모델은 `jev-1.13.0` 하나입니다. 일반 LLM을 검증기로 사용하는 Qwen·Llama 계열을 비교 후보로 두고, 이를 전문 decision model과 구분해 표기합니다. 서로 다른 전문 decision model만 비교하려면 실제로 사용할 수 있는 다른 제품·모델부터 확인해야 합니다. |
| 2 | **평가 데이터와 튜닝 데이터 분리** | 임계값과 fallback 규칙을 별도 개발 세트에서 결정한 뒤 평가 설정을 고정합니다. 현재 방법 절에는 평가 사례를 보며 설정을 선택했고 별도 튜닝 세트가 없었다고 명시되어 있습니다. 이 한계를 제거하는 재평가가 필요합니다. |
| 3 | **비교 실험과 ablation을 세 번씩 반복** | 각 반복에서 같은 최초 호출 후보와 같은 사례 ID를 모든 방법에 사용합니다. 같은 질문, 수락 규칙, 최대 세 번의 수정 예산을 유지합니다. 현재 주 실험은 세 번의 평균이지만 F/A/L ablation은 한 번입니다. 평균·표준편차와 사례 ID별 paired 차이의 신뢰구간을 보고합니다. |
| 4 | **지연시간 실험 보강** | 호출 정확도, 잘못된 호출의 수락률, 불필요한 호출, 수정 횟수, 토큰·비용, 평균/p50/p95 지연시간을 함께 기록합니다. API 처리시간과 재시도 대기 등을 포함한 실제 경과시간을 구분합니다. 수정 횟수 또는 입력 context 길이를 x축으로 한 line plot을 반복 측정으로 보강합니다. 같은 실행 경로를 잘라 얻은 결과를 독립 반복 실험으로 표현하지 않습니다. |
| 5 | **4쪽 제출 형식 최종 확인** | 현재 검토본은 참고문헌을 포함해 총 4쪽으로 정리했습니다. 최종 익명 review 형식으로 바꿀 때도 참고문헌과 부록까지 4쪽을 유지해야 합니다. 별도 Related Work 절 없이 Methodology 초반에 관련 문헌을 넣는 구조는 전달받은 피드백과 맞습니다. 공식 규정은 아래 링크에서 확인했습니다. |
| 6 | **초록·서론의 남은 주장 검토** | 서론의 `xxx`와 “diverse decision-model verifier”는 현재 한 검증 모델만 사용한 실험과 맞지 않습니다. 추가 benchmark·검증 모델 실험을 실제로 수행하거나, 다음 저자 검토에서 문구를 실제 평가 범위에 맞춰 수정해야 합니다. 초록·서론의 빨간 문구와 문법도 제출 전 검토 대상입니다. 이번 수정 범위에서는 그대로 둡니다. |

## 검증 모델 비교 후보

| 후보 | 비교에서의 역할 | 확인한 근거 |
|---|---|---|
| `jev-1.13.0`, `jev-1.14.0` | 같은 Jev 계열의 **버전 비교** | TypeSafe 공식 문서에서 두 고정 버전 ID와 같은 System One endpoint를 확인했습니다. 두 버전을 서로 다른 모델 series로 세면 안 됩니다. 변경 가능한 alias 대신 고정 버전 ID를 사용합니다. |
| `Qwen/Qwen2.5-7B-Instruct` | 일반 LLM을 이진 질문에 답하는 검증기로 사용하는 baseline | 공식 모델 카드가 JSON 등 structured output 지원 개선을 설명합니다. 이 논문의 호출 검증 정확도는 아직 측정하지 않았습니다. |
| `Meta-Llama-3.1-8B-Instruct` | 다른 일반 LLM 계열의 검증기 baseline | 공식 모델 카드에 API-Bank/BFCL 등 tool-use 평가가 있습니다. 이 논문의 이진 검증 성능과 동일한 평가가 아닙니다. |

현재 Noul 출력은 yes 확률이며 별도 confidence 필드는 없습니다. 이 확률과 일반 LLM이 텍스트로 출력한 confidence 값을 동일한 보정된 확률로 취급하지 않습니다. 먼저 같은 이진 판정 인터페이스로 비교하고, 확률을 비교하려면 별도 개발 세트에서 보정 및 신뢰성 평가를 해야 합니다.

모델 근거: [TypeSafe Models](https://docs.typesafe.ai/models), [Qwen 공식 모델 카드](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Llama 공식 모델 카드](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md).

## 관련 문헌에서 유지할 구분

- [Toolformer](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html) (Schick 외, NeurIPS 2023)는 API 선택·호출 시점·인자를 자기지도학습으로 학습합니다. 기존 연구 전체를 “형식만 검사한다”거나 “의미 검증을 다루지 않는다”고 일반화하면 안 됩니다.
- [τ-bench](https://arxiv.org/abs/2406.12045) (Yao 외, 2024)는 정책을 따르는 사용자–에이전트 대화와 반복 실행의 일관성을 평가합니다. 현재의 단일 호출·실행 전 평가 범위가 실환경 신뢰성의 일부라는 점을 설명하는 근거입니다.
- [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) (Lightman 외, 2023 preprint)는 수학 추론의 단계별 과정 감독을 연구합니다. 검증 모델의 관련 연구로 인용할 수 있지만, 이 논문의 이진 함수 호출 decision model과 같은 종류의 모델로 설명하지 않습니다.

## WWW 2027 제출 규정

[공식 Research Track 안내](https://www2027.thewebconf.org/research-track-papers/)의 Formatting Requirements에는 **“For short papers: 4 (four) pages, including references and optional Appendix.”**라고 명시되어 있습니다. 권장 형식은 `sigconf, anonymous, review`이며, 첫 페이지에 Web 및 선택한 track과의 관련성을 설명해야 합니다. 저자·소속·감사의 글 등은 제출용 익명본에서 점검합니다.

[공식 홈페이지](https://www2027.thewebconf.org/)에 안내된 short paper 일정은 초록 2026년 11월 9일, 논문 11월 16일이며 마감은 AoE 기준입니다. 투고 직전에 최신 공지를 다시 확인합니다.

## Overleaf 코멘트 가져오기

**피드백이 달린 원래 프로젝트의 Review 패널에서 가져와야 합니다.** 프로젝트 복사본의 Comments·Chat·History·Tracked Changes는 비어 있다는 것이 [Overleaf 공식 복사 안내](https://docs.overleaf.com/managing-projects-and-files/copying-a-project)에 명시되어 있습니다.

1. 원래 프로젝트를 열고 Review 패널에서 코멘트와 답글을 확인합니다. 해결된 코멘트도 패널의 resolved comments 목록에서 확인합니다.
2. 각 스레드의 **파일 이름, 대상 문장 또는 주변 원문, 작성자, 코멘트와 답글, 해결 여부**를 텍스트로 복사해 한 문서에 모읍니다. 원문 위치가 함께 있어야 개정본에서 대응할 부분을 찾을 수 있습니다. 텍스트 선택이 어려우면 코멘트와 해당 원문이 함께 보이는 화면을 저장합니다.
3. 논문 소스는 **File → Download → Download as source (.zip)**, PDF는 **Download as PDF**로 따로 저장합니다. 이는 원문 전달 방법이며 Review 코멘트 목록을 내보낸 결과로 취급하지 않습니다.

[공식 Commenting 안내](https://docs.overleaf.com/collaborating/commenting)는 코멘트 열람·답글·해결 및 해결된 스레드 보기를 설명합니다. [공식 다운로드 안내](https://docs.overleaf.com/managing-projects-and-files/downloading-a-project)는 소스 ZIP과 PDF 다운로드를 설명합니다. **확인한 공식 문서에서는 코멘트 전체를 한 번에 파일로 내보내는 기능을 확인하지 못했습니다.** 따라서 “ZIP이나 PDF를 받으면 Overleaf Review 코멘트도 함께 들어 있다”고 안내할 수 없습니다. PDF에 별도로 추가된 하이라이트 주석은 PDF 자체에서 추출할 수 있습니다.


## PDF 코멘트별 반영표

아래 위치는 원본 주석 PDF의 페이지입니다. 초록·서론의 16개 코멘트는 요청에 따라 수정 대상에서 제외했습니다. 제목의 Jev-Backed → DMFC 변경은 이전 형식 교체에서 이미 반영되어 있습니다. 남은 20개 코멘트 중 문장·구조·그림 수정은 반영했고, 추가 모델 실험을 요구한 두 항목은 후속 과제로 남겼습니다.

| 원본 위치·대상 | 코멘트 요지 | 이번 처리 |
|---|---|---|
| p.1 Tool use and benchmarks | 연구 흐름을 읽기 쉽게 정리 | Methodology 2.1에 호출 문제 정의 → Toolformer·Gorilla·ToolLLM → 평가 범위의 순서로 통합했습니다. |
| p.1 These suggest broader tests | 기존 연구에서 제안 방법으로 연결 | 단일 호출의 실행 전 검증 범위와 아직 확정할 수 없는 accuracy–latency 관계를 설명했습니다. |
| p.1 Checks and feedback | 앞 코멘트와 같은 문헌 정리 | Self-Refine·ToolVerifier·ToolCritic·ReAct의 역할을 구분했습니다. |
| p.1 Accumulated prompts | 문헌과 역할 설명 | many-shot demonstration과 한 요청의 거절 기록을 구분하고 해당 문헌을 인용했습니다. |
| p.2 Figure 1 | 그림 디자인 개선 | 첫 페이지에 Direct/schema repair, generic self-review, DMFC 비교 도식을 새로 그렸습니다. 구현 차이를 보여주는 도식이며 측정 결과로 표시하지 않았습니다. |
| p.2 Jev as a judge | decision model로 일반화 | 2.2 Decision-Model Verification으로 정리하고 과정 검증·confidence escalation 문헌의 차이도 설명했습니다. |
| p.2 proposal | 더 적절한 용어 검토 | 생성된 호출은 candidate, 그 후보에 대한 판정은 feedback으로 구분했습니다. |
| p.2 single-call-or-no-call | 각주 대신 본문 | 병렬·multi-turn 제외 범위를 문제 정의의 본문에 넣었습니다. |
| p.2 S(c)=1 | 문장 다듬기 | local validation 항목을 밝히고 success indicator로 정의했습니다. |
| p.2 one Jev | 일반화·여러 모델·구현 상세 | 일반 G/J 인터페이스와 실제 모델·임계값·예산을 분리했습니다. 다른 검증 모델 실험은 아직 수행하지 않았습니다. |
| p.2 Noul | 의미 설명 | yes 확률을 반환하는 API primitive라고 정의하고 confidence와 구분했습니다. |
| p.2 3.1 | 모듈에 맞는 제목 | Decision-Model Verification으로 수정했습니다. |
| p.2 3.2 | 모듈에 맞는 제목 | Acceptance and Cumulative Repair로 수정했습니다. |
| p.3 Algorithm 1 반복 설명 | short용 중복 삭제 | 알고리즘을 줄이고 긴 줄별 해설을 삭제했습니다. 구현상 중요한 예외는 본문에 남겼습니다. |
| p.3 Final Proposal | 중복 절 정리 | 독립 절을 없애고 수락·예산 소진 시 처음 후보를 반환하는 규칙에 통합했습니다. |
| p.3 repository revision | 불필요한 상세 삭제 | 본문의 긴 revision/hash 각주를 삭제했습니다. 재현 정보와 기록은 저장소에 유지했습니다. |
| p.3 Table 1 | 다른 decision model 2~3개 비교 | 현재 두 생성 모델 결과만 간결하게 제시했습니다. 다른 검증 모델 비교는 새 실험 후 추가해야 합니다. |
| p.3 Models and Methods | polishing | Evaluation and Implementation과 Compared methods로 정리했습니다. |
| p.3 seven methods | 각 비교군 목적 설명 | Direct/schema/self-review와 F/A/L의 차이, A의 no-call 처리, L의 같은 예산을 명시했습니다. |
| p.4 Results | 주요 실험·ablation·latency 분리 | Main Experiment와 Ablation and Latency로 나누고 정확도·평균·p95 line plot을 추가했습니다. |

## 검증 및 보존

- 수정 전 ACM 원고의 source/PDF/bibliography/section files는 저장소 paper/versions/2026-10-20-acm-import에 보관했고, commit 8e37143와 일치함을 검사했습니다. 이전 JBFC 원고는 paper/versions/2026-10-18-jbfc에 그대로 있습니다.
- 초록과 서론의 동일성, 총 4쪽, 주요 절, 표 2개, 첫 페이지 비교 그림, 정확도·API 시간 그림, 알고리즘, 숫자 인용을 확인했습니다. 표·그림 자동 갱신과 형식 검사 관련 오프라인 테스트 81개가 통과했습니다.
- 표 수치는 기존 final summaries와 대조했고, p95는 기록된 API 시간에서 확인했습니다. 실험 구현과 원시 기록을 수정하지 않았습니다.
- 현재 PDF는 저자 검토본입니다. 초록·서론의 미완성 부분, 익명 제출 형식, Web/track relevance는 제출 전 후속 검토가 필요합니다.
- 내장 LaTeX 미리보기는 제공된 다중 파일 ACM 프로젝트와 class 버전을 처리하지 못했습니다. 저장되어 있던 LaTeX 환경으로 PDF를 생성하고 모든 페이지를 화면으로 점검했습니다.
