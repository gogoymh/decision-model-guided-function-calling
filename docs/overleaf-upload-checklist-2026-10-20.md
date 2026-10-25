# Overleaf 업로드 파일과 검증 결과

> Historical revision record (2026-10-20). Names and archive filenames below describe that revision. The current method and library are **DMGFC** / `dmgfc.DMGFC`, with `dmgfc-eval` as the current command. See the [current naming guide](dmgfc-naming-2026-10-24.md) and [API migration guide](dmgfc-public-api-migration-2026-10-24.md).

> **2026-10-21 후속 원문 복원:** 아래는 10월 20일 수정본의 이력입니다. 현재 초록·서론은 최초 ZIP 원문으로 돌아갔으며 총 4쪽입니다. 현재 업로드 파일은 [원문 복원 체크리스트](overleaf-polished-text-restore-2026-10-21.md)를 확인하세요.

2026년 10월 20일에 현재 저장소 원고를 직접 검사했습니다. 업로드용 ZIP은 **`DMFC-Overleaf-anonymous-review-2026-10-20.zip`**이며, 아래 필수 소스 10개만 포함합니다. ZIP 최상위에 `main.tex`가 있고, 절 파일은 `sec/` 안에 있습니다. PDF, 이전 버전, ACM 예제, 컴파일 부산물은 넣지 않았습니다.

## 파일별 변경 확인

비교 기준은 사용자가 처음 전달한 `_WWW_27__Decision_Model_Guided_Function_Calling_with_Cumulative_Verification_Feedback.zip`입니다. “추가 코멘트 전 대비”는 저장소에 보존한 `paper/versions/2026-10-20-before-overleaf/`의 원고, 기준 커밋 `927e895`와의 비교입니다. 두 기준의 파일 내용을 직접 대조했습니다.

| Overleaf 업로드 경로 | 최초 ZIP 대비 | 추가 코멘트 전 대비 | 실제 내용·역할 확인 |
|---|---|---|---|
| `main.tex` | 변경 | 변경 | 초록이 이 파일 안에 있습니다. Claude·API-Bank 결과, 한 검증 모델이라는 범위, 현재 절 연결을 확인했습니다. |
| `references.bib` | 변경 | 동일 | 앞선 PDF 피드백 단계에서 추가한 관련 연구를 포함합니다. 현재 인용 16개가 모두 정의되어 있습니다. |
| `sec/introduction.tex` | 변경 | 변경 | 세 연구 질문, 방법 요약, 추가 평가와 Claude 결과를 반영했습니다. `xxx`와 미실행 다중 검증 모델 주장은 없습니다. |
| `sec/methodology.tex` | 변경 | 변경 | 과제 정의, 함수·인자 검사, no-call 예외, 누적 피드백, 수락 조건과 알고리즘을 확인했습니다. |
| `sec/experiment_setting_and_results.tex` | 변경 | 변경 | BFCL/API-Bank × 세 생성 모델의 6행 주 실험 표, F/A/L 설명, ablation·시간·비용·한계를 확인했습니다. |
| `sec/conclusion.tex` | 변경 | 변경 | Claude의 BFCL 정확도 감소와 API-Bank의 탐색 평가 범위를 반영했습니다. |
| `sec/framework.tex` | 새로 추가 | 동일 | 앞선 PDF 피드백 단계에서 반영한 첫 페이지 비교 그림입니다. |
| `sec/repair_tradeoff.tex` | 새로 추가 | 동일 | 앞선 PDF 피드백 단계에서 반영한 정확도·평균/p95 API 시간 line plot입니다. 실험 절에서 이 파일을 읽습니다. |
| `acmart.cls` | 동일 | 보존본에 미포함 | 제공된 ACM 클래스 그대로입니다. 변경할 필요가 없는 필수 형식 파일입니다. |
| `ACM-Reference-Format.bst` | 동일 | 보존본에 미포함 | 제공된 ACM 참고문헌 스타일 그대로입니다. 변경할 필요가 없는 필수 형식 파일입니다. |

최초 ZIP 대비 **변경 6개 + 추가 2개 + 동일한 형식 파일 2개 = 10개**입니다. 익명 심사 형식으로 바꾸면서 `main.tex`와 `sec/introduction.tex`를 다시 수정했습니다. 감사의 글 파일은 현재 원고에서 읽지 않고 업로드 ZIP에서도 제외했습니다.

## 수행한 검증

- `main.tex`부터 중첩 입력 파일, 참고문헌, 클래스와 스타일을 따라가 10개의 필수 파일을 확인했습니다. 실제 컴파일 입력 기록도 대조했습니다.
- ZIP을 다시 열어 10개 파일의 내용이 현재 저장소 소스와 바이트 단위로 같은지 확인했습니다. 압축 손상 검사도 통과했습니다.
- ZIP을 별도 빈 작업 폴더에 풀어 **그 파일들만으로** pdfLaTeX/BibTeX 컴파일에 성공했습니다. 기존 저장소의 보조 파일에 의존하지 않습니다.
- 새 컴파일 PDF는 참고문헌을 포함해 **총 4쪽**입니다. 현재 `paper/main.pdf`와 추출 본문이 같고, 동일 해상도로 렌더링한 4개 페이지의 픽셀이 모두 일치했습니다. 네 페이지를 화면으로도 검사했습니다. 공개 이름·호환 API·문서 생성기·익명 원고 검사까지 포함한 전체 오프라인 테스트 768개와 코드 검사도 통과했습니다.
- 본문에 인용한 참고문헌 **16개**가 모두 정의되어 있고, `xxx` 및 빨간 초안 문구는 없습니다. 심사용 빨간 줄 번호는 ACM `review` 옵션이 생성한 정상 표시입니다. 표 2개, 첫 페이지 비교 그림, 정확도/API 시간 그래프, 알고리즘을 확인했습니다.
- 핵심 실험이나 측정 수치를 이번 업로드 준비 과정에서 변경하지 않았습니다. 기존 실험 기록과 핵심 실험 코드 파일 해시를 유지했습니다. 공개 API는 `import dmfc`, `dmfc.DMFC`로 바꾸고 `jbfc` 호환을 보존했습니다.

이 검증은 업로드할 파일과 원고의 일치를 확인한 것입니다. **서로 다른 검증 모델 계열 비교는 아직 수행하지 않았고**, 화면 2·3의 `show more` 뒤 코멘트 본문도 미확인입니다. Claude는 생성 모델 비교입니다. 해당 항목까지 모두 해결됐다는 의미로 표시하지 않습니다.

## 익명 심사본으로 바꾼 부분

- 저자·소속·이메일을 활성 소스와 PDF에서 제거했고, 감사의 글을 제외했습니다. 익명화 전 원고는 `paper/versions/2026-10-20-author-review/`에 보존했습니다.
- ACM `sigconf,anonymous,review`와 줄 번호를 사용합니다. CCS 항목과 ACM Reference Format도 넣었습니다. 제공된 `acmart.cls`와 `.bst`는 그대로입니다.
- 2018년 예제 학회 정보 대신 공식 WWW 2027 일정·장소를 넣었습니다. DOI/ISBN은 아직 배정되지 않아 비워 두었습니다.
- 첫 페이지에 Web agents의 실행 전 호출 검증이라는 연구 문제와 **Web Infrastructure and Agentic Systems** 트랙 관련성을 적었습니다. 현재 연구에 가장 가까운 트랙을 작업 기준으로 사용했으며, 저자가 다른 트랙을 정하면 해당 문구를 맞춰야 합니다.
- 참고문헌 포함 4쪽, 저자 정보의 본문/PDF 정보 누출, 두 표·두 그림·알고리즘·인용, ZIP 독립 컴파일을 확인합니다.

## 기존 Overleaf 프로젝트에 반영하는 순서

1. 위 이름의 ZIP을 컴퓨터에서 압축 해제합니다.
2. **피드백이 달린 원래 프로젝트**에서 같은 경로에 10개 파일을 업로드하고, 같은 이름의 파일은 덮어씁니다. 최상위에는 `main.tex`, `references.bib`, `acmart.cls`, `ACM-Reference-Format.bst` 4개를, `sec/`에는 절 파일 6개를 둡니다. `paper/` 폴더를 새로 만들어 넣지 않습니다.
3. 프로젝트 설정의 **Main document를 `main.tex`**, **Compiler를 pdfLaTeX**로 선택하고 다시 컴파일합니다. 소스의 형식은 `sigconf,anonymous,review`입니다. 기존 `sigconf.tex`를 메인으로 두면 ACM 예제가 출력됩니다.
4. 1쪽 비교 그림과 초록의 Claude/API-Bank 문장, 3쪽 주 실험 표의 6행과 평균/p95 그래프, 총 4쪽을 확인하면 이번 업로드본과 일치합니다.

초록은 `sec/abstract.tex`가 아니라 **`main.tex`**에 있습니다. `sec/abstract.tex`, `sec/related_work.tex`, `sec/acknowledgements.tex`, `sigconf*.tex`, `sample*`, `acmguide.pdf`, `versions/`는 이번 논문이 읽지 않으므로 업로드할 필요가 없습니다. 그림은 TikZ/PGFPlots 소스에서 생성되어 별도 PNG/PDF 그림 파일이 필요하지 않습니다. Overleaf의 코멘트 스레드에 Reply/Resolve를 수행한 것은 아닙니다.
