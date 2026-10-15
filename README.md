# Jev-Backed Function Calling (**JBFC**)

**JBFC**는 함수 호출 LLM의 제안을 **Jev가 함수 선택과 인자 정확성으로 나누어 검사**하고, 둘 다 통과할 때까지 최대 세 번 수정하는 방법입니다. 이 저장소에는 논문 원고와 결과표, BFCL 데이터 준비, 실행, 재개, 채점 및 표 반영 코드가 포함되어 있습니다.

현재 **실제 모델 실험은 수행하지 않았습니다.** 논문의 `--`는 미실행 값입니다. 로컬 테스트와 데모 결과를 모델 성능으로 사용하지 않습니다.

## 바로 시작하기

이미 이 환경에는 의존성과 BFCL 데이터가 준비되어 있습니다. 다른 환경에서는 Python 3.11 이상과 [uv](https://docs.astral.sh/uv/)를 준비한 뒤 다음을 실행합니다.

```bash
cd /Users/minhyeong/jev-backed-function-calling
make setup
make prepare
```

기본 설정은 제공해 주신 APIM 게이트웨이를 사용합니다. `.env`의 `APIM_SUBSCRIPTION_KEY`에 프로젝트 키 하나를 넣으면 생성기와 Jev가 같은 키로 연결됩니다. 공급자 원본 키를 넣는 항목이 아닙니다.

```dotenv
API_GATEWAY=apim
APIM_BASE_URL=https://enhans-aoai-prod-apim.azure-api.net
APIM_SUBSCRIPTION_KEY=your_project_subscription_key
LLM_API_STYLE=responses
```

기본 생성기는 `gpt-5.4-mini-2026-03-17`, Jev는 안정 버전인 `jev-1.13.0`으로 고정했습니다. APIM은 `api-key` 인증을 사용하며, OpenAI에는 `/llm/openai/v1/responses`, Jev에는 `/llm/typesafe/v1/systemone`으로 요청합니다. 해당 모델에 대한 프로젝트 접근 권한은 실제 연결 확인 단계에서 확인합니다.

`LLM_API_STYLE=anthropic_messages`로 바꾸면 `/llm/anthropic/v1/messages`를 사용합니다. 이때 `LLM_MODEL`에 사용할 Anthropic 모델 ID를 명시하고 가격 설정도 함께 지정해야 합니다. 가이드에 모델 목록은 없으므로 임의의 Claude 모델을 기본값으로 가정하지 않습니다. 공급자별 요청 형식과 응답을 자동 변환하며, BFCL의 원래 선택적 인자와 스키마는 유지합니다.

직접 연결은 `API_GATEWAY=direct`로 선택합니다. OpenAI는 `LLM_API_STYLE=responses` 또는 `chat_completions`와 `LLM_API_KEY`, Jev는 `TYPESAFE_API_KEY`를 사용합니다. APIM 모드에서는 직접 연결용 키와 URL을 사용하지 않습니다. 자세한 설정과 변경 근거는 `docs/apim-gateway.md`에 있습니다.

```bash
make check-config # API 호출 없이 경로·모델·키 설정 여부 확인
make test       # 키 불필요: 모의 HTTP 응답, 흐름, 데이터, 공식 채점기 검증
make demo       # 키 불필요: 인자 오류를 한 번 수정하는 합성 예제
make smoke      # APIM 키 필요: 개발 데이터 8개, 1회 실행, 실제 API 연결 확인
make experiment # APIM 키 필요: 전체 테스트 집합, 3회 반복, 6가지 방법 비교
```

`make experiment`는 **2,748개 × 3회 × 6방법 = 49,464개 방법별 기록**을 생성합니다. 최초 제안은 비교 방법끼리 공유하지만, 수정과 Jev 검증에는 별도 요청이 발생합니다. 실행 시간과 비용은 토큰 길이, 통과율 및 공급자에 따라 달라집니다. 먼저 `make smoke`로 계정과 모델 접근을 확인할 수 있습니다. 이미 완료한 기록은 재개할 때 다시 요청하지 않습니다.

## 실험 정의

```text
사용자 요청 + 함수 목록
  -> 함수 호출 LLM의 최초 제안
  -> Jev의 함수 선택 검사와 인자 검사 (한 API 요청)
  -> 두 검사 모두 통과하면 종료
  -> 하나라도 실패하면 결과를 수정 프롬프트에 추가
  -> 같은 LLM이 수정, 두 검사를 새 후보에 다시 수행
  -> 최대 3회 수정 이후 종료
  -> 최종 후보를 오프라인 채점기에 전달
```

- `max_repairs=3`은 **최초 1회 + 수정 최대 3회 = 제안 최대 4개**입니다.
- 기본 기준은 두 Noul 값이 각각 `0.8` 이상인 것입니다. 두 값의 독립성이나 결합 정확도 80%를 가정하지 않습니다.
- 기본 `execute_last`는 한도에 도달하면 Jev가 거부한 최종 후보도 채점합니다. 형식이 잘못된 후보는 오답으로 처리합니다.
- `abstain`은 한도에 도달해도 거부된 후보를 보류합니다. 이 정책의 수락 정확도에는 반드시 커버리지를 함께 보고합니다.
- Jev는 자유 형식 설명을 생성하지 않습니다. 수정 피드백은 실제 확률, 검사별 통과 여부, 로컬 스키마 오류로 구성합니다.
- 이전 검사 통과를 기억해 재사용하지 않습니다. 매번 같은 후보에 대해 두 검사를 다시 수행합니다.
- BFCL 함수는 실행하지 않습니다. 이 실험은 **호출 선택·인자 정확성**을 평가하며 실제 외부 API의 실행 성공률을 측정하지 않습니다. 응용 시스템은 `run_example(..., executor=callback)`으로 최종 실행 경계를 연결할 수 있습니다.

비교 방법은 직접 호출, 스키마 오류 수정, 일반적인 자기 재검토, 함수 검사만 사용하는 **JBFC-F**, 인자 검사만 사용하는 **JBFC-A**, 두 검사를 모두 사용하는 **JBFC**의 여섯 가지입니다. 논문과 표에는 이 명칭을 사용하며, 실행 옵션과 저장 기록의 내부 ID는 각각 `jev_function`, `jev_arguments`, `jev_joint`입니다. 자기 재검토는 모든 수정 예산을 사용합니다. 비교 방법의 **최대 제안 횟수**가 같으며 실제 토큰 수나 비용이 같다는 뜻은 아닙니다. Jev 단일 검사 비교도 동일 요청으로 두 확률을 구하지만, 선택한 의미 검사와 공통 스키마 검사만 통과 판단·피드백에 사용합니다.

## 데이터와 채점

BFCL v4의 고정된 단일 턴 Python 하위 집합 3,051개를 사용합니다. `simple_python`, `multiple`, `irrelevance`, `live_simple`, `live_multiple`, `live_irrelevance`, `live_relevance`를 포함합니다. 공식 전체 리더보드 점수는 아닙니다.

- 고정 버전: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`
- 개발 303개, 테스트 2,748개. 카테고리별 `sha256(seed:id)` 순위, seed 1729, 앞 10% 내림을 개발용으로 사용합니다.
- 본문 주 지표는 공식 AST 정확도와 no-call 정확도를 합친 사용자 정의 micro-average이며, `live_relevance` 15개를 제외한 2,733개가 분모입니다. AST와 no-call 지표도 별도로 저장합니다.
- `live_relevance`는 호출 여부만 평가할 수 있으며 함수·인자 정답을 제공하지 않습니다.
- 원본 답안의 모순 4개를 확인했습니다. 주 평가에서는 그대로 유지하며 자세한 근거는 `docs/benchmarks.md`와 준비 매니페스트에 있습니다.
- 다운로드 파일과 정규화 데이터의 체크섬을 검사하고, 공식 AST 평가기의 본문을 변경하지 않는 경량 어댑터로 채점합니다.
- 정답, 데이터 ID, 카테고리 및 정답이 포함될 수 있는 채점 오류는 **LLM/Jev 요청에 포함하지 않습니다.**

준비된 데이터는 `data/bfcl/`에 있으며, 외부 원본 데이터는 Git에서 제외했습니다. `make prepare`로 같은 버전을 재구성합니다. 원본 Apache-2.0 라이선스도 함께 저장합니다.

## 결과·재개·반복 예산

```bash
# 기본 비교. 동일 설정으로 중단 이후 재개할 수 있습니다.
uv run jev-fc run --output results/main --resume

# 별도 개발 실행. 프롬프트/임계값을 변경할 때에는 테스트 데이터를 사용하지 않습니다.
uv run jev-fc run --split dev --methods jev_joint --max-repairs 0 \
  --replicates 1 --output results/calibration --resume
uv run python scripts/calibrate_threshold.py results/calibration

# 직접 설정한 비교 예. 기본 논문과 다른 설정은 별도 경로에 저장합니다.
uv run jev-fc run --threshold 0.9 --output results/threshold_09 --resume

# 요약만 생성합니다.
uv run jev-fc report results/main
```

임계값 분석은 최초 개발 후보만 사용하며, 추천값을 사용하려면 `--threshold`로 명시합니다. 기본 `0.8`은 조정하지 않은 사전 설정입니다. 개발 기록은 주 논문 표로 반영할 수 없습니다.

`make experiment`는 설정이 기본 논문 프로토콜과 다르면 결과 요약까지 완료하고 논문 반영만 건너뜁니다. `--skip-paper`로 논문 반영을 명시적으로 생략할 수도 있습니다. 실제 세 번 수정 기록의 앞부분을 사용해 0·1·2회 제한도 평가합니다. 반복 한도를 모델에 알려주지 않고 중간에 함수를 실행하지 않으므로, 같은 후보에서 종료 조건만 바꾸는 비교가 가능합니다. 이 단계에는 추가 API 비용이 없습니다. 다른 임계값·프롬프트를 이 방법으로 재현할 수는 없습니다.

```bash
uv run python scripts/run_suite.py --include-abstain
```

실행 폴더에는 다음 파일이 생깁니다.

| 파일 | 내용 |
|---|---|
| `manifest.json` | 모델, 설정, 데이터/코드 지문, 대상 ID, 단가 |
| `initial.jsonl` | 비교 방법끼리 공유하는 최초 제안 |
| `records.jsonl` | 모든 후보, Jev 판단, 실제 피드백, 채점, 비용/지연 기록 |
| `errors.jsonl` | 키와 응답 내용을 제외한 오류 유형, 시각, HTTP 상태, 안전한 요청 ID와 시도 횟수 |
| `summary.json` | 카테고리별·반복별 지표, paired bootstrap, 수락·거절 진단 |
| `summary.csv` | 방법별 비교 요약 |

코드, 모델, API 형식·경로, 작업자 수, 재시도 설정, 데이터 또는 실행 조건이 바뀌면 기존 결과에 이어 쓰지 않고 새 출력 경로를 요구합니다. 키 값은 지문에 포함하지 않으므로 같은 프로젝트 키를 갱신해도 재개할 수 있습니다. 잘못된 응답을 정답으로 간주하거나 API 오류를 no-call 예측으로 바꾸지 않습니다. 완료된 기록의 중복과 설정 불일치도 보고 단계에서 검사합니다.

비용은 응답에 포함된 사용량과 설정된 단가로 추정합니다. Responses의 캐시 입력과 Anthropic의 캐시 읽기·5분/1시간 쓰기 토큰을 구분합니다. 필요한 단가나 사용량을 알 수 없으면 비용은 미상으로 남깁니다. 기본 단가는 공개 공급자 기준값이며 APIM 프로젝트의 가산 요금이나 실제 청구액은 아닙니다. 누락된 사용량은 0원으로 처리하지 않으며, 공급자 청구서·재시도 중 응답을 받지 못한 요청 비용까지 보장하지 않습니다. 비교용 지연 시간에는 공유 최초 제안의 관측 시간을 각 방법에 더합니다. 실제 벽시계 실행 시간과 이 논리적 지연 시간은 다릅니다.

## 논문

- `paper/main.tex`: 독립 실행형 영문 LaTeX, 본문 4쪽 + 참고문헌 1쪽
- `paper/main.pdf`: 확인된 PDF
- `paper/references.bib`: 별도 참고문헌 자료. 실제 원고는 내장 bibliography로 단일 파일 컴파일이 가능합니다.

Abstract, Introduction, Related Work, Method, Experiments, Conclusion, Acknowledgements가 본문에 포함됩니다. 표 3개에는 카테고리별 결과, 방법별 정확도/비용, 반복 한도 및 검증 진단을 준비했습니다. 저자는 익명으로 두었고, 특정 학회의 공식 스타일로 지정하지 않은 일반 2단 숏페이퍼입니다.

```bash
make report  # 검증된 기본 APIM/Responses 전체 테스트 실행의 세 표를 반영
make paper   # 설치된 pdflatex로 PDF 내보내기
make verify  # 본문 4쪽, References 5쪽 시작, 필수 절 확인
```

원고는 Codex의 내장 LaTeX 편집기에서도 그대로 열 수 있습니다. 실험 이후 표는 자동 반영되지만 **abstract·결론의 실증 해석은 실제 결과에 맞춰 수정해야 합니다.** 결과를 만들기 전부터 개선 효과를 주장하지 않습니다. 사용자 정의 모델·부분 데이터·다른 설정의 결과로 기본 원고를 채우려면 먼저 논문의 실험 조건을 함께 바꿔야 합니다.

## 확인한 공식 자료

- [TypeSafe 모델 및 버전](https://docs.typesafe.ai/models)
- [TypeSafe HTTP API](https://docs.typesafe.ai/api)
- [TypeSafe 함수 호출 예제](https://docs.typesafe.ai/cookbooks/function_calling)
- [TypeSafe 구조화 추출 검증 예제](https://docs.typesafe.ai/cookbooks/sde_cascade)
- [BFCL 논문](https://proceedings.mlr.press/v267/patil25a.html)
- [OpenAI 함수 호출 문서](https://developers.openai.com/api/docs/guides/function-calling)
- [기본 생성기 사양과 가격](https://developers.openai.com/api/docs/models/gpt-5.4-mini)

`docs/jev-research.md`와 `docs/literature.md`에는 조사 근거 및 ToolVerifier·ToolCritic 등 직접 관련된 선행연구와의 차이를 정리했습니다.

추가로 [JEV-as-a-Judge: Accept When Confident, Escalate When Unsure (v5)](https://arxiv.org/abs/2609.26550v5)의 본문과 부록을 검토해 논문에 인용했습니다. 방법·결과·한계와 이번 실험의 차이는 `docs/jev-as-a-judge.md`에 정리했습니다.
