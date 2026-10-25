# 최초 ZIP의 확정 문구 복원 및 업로드 파일

> Historical revision record (2026-10-21). Names and archive filenames below describe that revision. The current method and library are **DMGFC** / `dmgfc.DMGFC`, with `dmgfc-eval` as the current command. See the [current naming guide](dmgfc-naming-2026-10-24.md) and [API migration guide](dmgfc-public-api-migration-2026-10-24.md).

사용자의 2026년 10월 21일 요청에 따라 최초 ZIP에 이미 작성되어 있던 논문 문구를 복원했습니다. 앞선 추가 피드백 반영과 4쪽 압축 과정에서 초록과 서론을 다시 쓰고 방법 절 제목을 바꿨던 변경을 되돌린 것입니다. 이 요청이 이전 문장 수정 허용보다 우선합니다.

| 파일 | 복원한 부분 | 검증 |
|---|---|---|
| `main.tex` | 최초 ZIP의 `abstract` 환경 전체. 빈 `sec/abstract` 입력과 원래 빨간 문구를 포함합니다. | 원본 환경과 바이트 단위로 같은 문자열입니다. 제목도 원본과 같습니다. |
| `sec/introduction.tex` | 문단·강조·세 기여 항목을 포함한 파일 전체. | 최초 ZIP 파일과 바이트 단위로 같습니다. |
| `sec/methodology.tex` | 첫 제목 `Task Definition`. | 원본 파일의 비어 있지 않은 제목을 정확히 복원했습니다. 뒤에 추가했던 방법 본문·수식·알고리즘은 유지했습니다. |

참고문헌의 원래 `Lamport1994` 항목과 제공된 `acmart.cls`, `ACM-Reference-Format.bst`는 원본과 같습니다. 최초 ZIP에서 실험·결론 파일은 비어 있었으므로 현재 추가된 본문, Sonnet/API-Bank 실험 결과, 두 그림과 표를 유지했습니다. 실험 기록·코드·공개 DMFC API는 바꾸지 않았습니다.

익명 심사 옵션·현재 ACM CCS·WWW 2027 메타데이터는 유지하고, ACM 예제의 2018년 학회 정보·야구 그림·임시 CCS를 되살리지 않았습니다. 이번 요청은 확정 논문 문구의 복원으로 처리했습니다.

## 현재 상태

초록과 서론을 원문 그대로 복원했습니다. 앞서 추가했던 참고문헌 앞 강제 페이지 나눔만 제거해, 문장을 압축하지 않고 참고문헌 포함 총 **4쪽**을 유지했습니다. 원문의 빨간 문구, `xxx`, 실제로 실행하지 않은 다중 검증 모델 문장도 그대로입니다. 실제 검증 모델은 Jev 1.13.0 하나입니다. 앞서 추가했던 첫 페이지 Web/track 문장도 전체 서론 복원에 따라 제거됐습니다. 따라서 현재 파일은 **원문 복원 초안**이며, 이전의 4쪽 제출 검증을 통과했다고 표시하지 않습니다. 기존 `make verify`의 엄격한 기준은 유지합니다. 4쪽 기준은 통과하지만, 원문 서론에는 앞서 추가했던 Web/track 관련성 문장이 없어 전체 제출 검사는 실패합니다.

수정 직전의 원고와 PDF는 `paper/versions/2026-10-21-before-polished-text-restore/`에 보존했습니다. 원래 전달한 ZIP은 그대로입니다. 최신 PDF와 업로드 ZIP은 같은 복원 소스를 사용합니다.

## 업로드할 11개 소스

새 ZIP은 **`DMFC-Overleaf-original-text-restored-2026-10-21.zip`**입니다. 이전 10월 20일 익명 ZIP과 비교해 `main.tex`, `sec/introduction.tex`, `sec/methodology.tex`가 바뀌었고, 빈 `sec/abstract.tex`가 입력 의존성으로 추가됐습니다. 나머지 8개는 같습니다.

1. `main.tex`
2. `references.bib`
3. `acmart.cls`
4. `ACM-Reference-Format.bst`
5. `sec/abstract.tex` — 원본의 빈 파일이며 `main.tex`에서 다시 읽습니다.
6. `sec/introduction.tex`
7. `sec/methodology.tex`
8. `sec/experiment_setting_and_results.tex`
9. `sec/conclusion.tex`
10. `sec/framework.tex`
11. `sec/repair_tradeoff.tex`

같은 경로의 파일을 덮어쓰고 Main document는 `main.tex`, Compiler는 pdfLaTeX로 둡니다. `paper/` 폴더를 ZIP 위에 추가하지 않습니다. 감사의 글·이전 버전·예제 파일·컴파일 부산물은 넣지 않았습니다.

## 수행한 확인

- 최초 ZIP의 초록·서론·방법 제목, 원래 참고문헌 항목 및 ACM 형식 파일과 직접 대조했습니다.
- 원래 비어 있던 곳에 추가한 실험·결론과 그림 소스는 수정 직전 커밋과 같습니다.
- 새 업로드 ZIP의 11개 파일을 현재 저장소 소스와 대조하고, 별도 빈 폴더에서 그 ZIP만으로 컴파일했습니다.
- PDF의 4쪽 전체를 렌더링해 확인하고, 독립 컴파일본과 저장소 PDF의 본문 및 렌더링을 대조했습니다.
- 문구 보존을 위해 원본 오탈자·미완성 표현도 자동 교정하지 않았습니다. 그 부분을 수정하려면 별도 명시적인 요청이 필요합니다. 서론 5행의 원래 끝 공백도 바이트 일치를 위해 유지했습니다.
