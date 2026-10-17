# Jev-Backed Function Calling (**JBFC**)

**JBFC**는 함수 호출 LLM의 제안을 **Jev가 함수 선택과 인자 정확성으로 나누어 검사**하고, 필요한 검사를 통과할 때까지 정해진 한도 안에서 수정하는 방법입니다. no-call에서는 인자 검사를 생략합니다. **수정 한도는 하이퍼파라미터이며, 개발 데이터 실험으로 선택한 뒤 테스트 전에 고정합니다.** 이 저장소에는 논문 원고와 결과표, BFCL 데이터 준비, 실행, 재개, 채점 및 표 반영 코드가 포함되어 있습니다.

검사에 실패하면 **모든 이전 후보와 실제 Jev 피드백을 시간순으로 누적해 원래 사용자 요청 앞에 넣고**, 같은 LLM이 다시 제안하게 합니다. 최초 요청과 원본 대화의 내용은 유지하며, 수정 중에는 함수를 실행하지 않습니다. 함수와 인자는 LLM이 제안하고 Jev는 현재 후보의 정확성을 검사합니다.

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
make smoke      # APIM 키 필요: 개발 데이터 8개, 1회 실행, 수정 한도 1로 연결 확인
make tune       # APIM 키 필요: 전체 개발 집합에서 수정 한도 선택
make experiment # APIM 키 필요: 개발 실험·한도 선택 후 전체 테스트, 3회 반복·6방법 비교
```

`make experiment`는 먼저 **JBFC**로 **개발 303개 × 3회 = 909개 기록**을 만듭니다. 기본 수정 한도 후보는 `0,1,2,3,4,5`회입니다. 개발 정확도가 가장 높은 값을 선택하고, 동점이면 더 작은 값을 고릅니다. `live_relevance` 1개는 선택 점수에서 제외하므로 정확도 분모는 **302개 × 3회 = 906개**입니다. 선택한 한도를 여섯 비교 방법의 수정 예산으로 사용하며 Direct는 수정하지 않습니다.

선택을 검증한 뒤 **테스트 2,748개 × 3회 × 6방법 = 49,464개 방법별 기록**을 생성합니다. 여섯 방법은 사례·반복별로 같은 최초 LLM 제안 하나를 공유하고, 각 방법의 비용에는 이를 한 번씩 포함합니다. 실행 시간과 비용은 토큰 길이, 통과율 및 공급자에 따라 달라집니다. 먼저 `make smoke`로 계정과 모델 접근을 확인할 수 있습니다. 같은 설정의 완료된 기록은 재개할 때 다시 요청하지 않습니다.

## 실험 정의

**JBFC**(`jev_joint`)는 다음 순서로 진행합니다.

```text
사용자 요청 + 함수 목록
  -> 함수 호출 LLM의 최초 제안
  -> Jev의 함수 선택 검사와, 호출이 있으면 인자 검사 (한 API 요청)
  -> 필요한 검사와 스키마 검사가 모두 통과하면 종료
  -> 하나라도 실패하면 이전 후보와 Jev 판정을 시간순으로 누적
  -> 누적 기록을 원래 사용자 요청 앞에 넣음 (모든 후보는 미실행)
  -> 같은 LLM이 수정, 새 후보에 필요한 검사를 다시 수행
  -> 선택된 수정 한도 r에 도달하면 종료
  -> 최종 후보를 오프라인 채점기에 전달
```

최초 생성에는 수정 피드백을 넣지 않습니다. 수정 요청에서는 모든 이전 후보의 `round`, `calls`, `feedback`를 오래된 순서대로 JSON 목록에 담아 하나의 추가 사용자 메시지로 만듭니다. 이 메시지를 원본 대화 앞에 붙이며, 원래 메시지의 내용과 순서는 바꾸지 않습니다. 각 후보가 실행되지 않았다는 설명도 포함합니다. 최신 오류만 보내거나 기존 사용자 요청을 새 문장으로 대체하지 않습니다.

이전 단계의 관찰을 다음 제안에 반영한다는 점은 ReAct와 비슷합니다. 여기서 관찰은 실제 도구 실행 결과가 아니라 **실행 전 Jev 판정**이며, ReAct의 전체 에이전트 절차를 재현하는 실험은 아닙니다.

- 수정 한도 `r`이면 **최초 1회 + 수정 최대 r회 = 제안 최대 r+1개**입니다. 구현에는 3회라는 고정 상한이 없습니다.
- 개발 **JBFC** 실험에서 한도를 선택하고, 테스트에서는 모든 수정 방법에 같은 값을 적용합니다. `0`이 선택되면 수정하지 않지만 **JBFC**의 최초 제안 검사는 수행합니다.
- 후보 범위는 바꿀 수 있습니다. 기본 `0,1,2,3,4,5`는 탐색 설정이며, 5회를 방법의 최대 한도로 가정하지 않습니다.
- **JBFC**의 기본 수락 기준은 실제로 수행한 Noul 검사의 긍정 확률이 각각 `0.8` 이상이고 로컬 스키마 검사도 통과하는 것입니다. 함수 호출이 있으면 함수·인자 검사가 모두 필요합니다. 이 기준으로 정확도 80%를 보장하는 것은 아닙니다.
- 기본 `execute_last`는 한도에 도달하면 Jev가 거부한 최종 후보도 채점합니다. 형식이 잘못된 후보는 오답으로 처리합니다.
- `abstain`은 한도에 도달해도 거부된 후보를 보류합니다. 이 정책의 수락 정확도에는 반드시 커버리지를 함께 보고합니다.
- Jev는 자유 형식 설명을 생성하지 않습니다. 수정 피드백은 실제 확률, 검사별 통과 여부, 임계값과 로컬 스키마 오류로 구성합니다. 이전 후보와 이 피드백을 누적하되, Jev가 하지 않은 설명이나 정답 기반 힌트를 만들지 않습니다.
- **JBFC**는 이전 판정을 수정 문맥으로 제공하지만, 이전 검사 통과를 새 후보의 통과로 재사용하지 않습니다. 매번 현재 후보에 필요한 검사를 다시 수행합니다. 함수와 인자 모두 수정할 수 있습니다.
- no-call(`calls=[]`)에서는 인자 질문을 보내지 않습니다. 함수 질문으로 호출하지 않는 선택이 적절한지만 판단하고 함수 확률과 스키마 기준을 적용합니다. 기록에는 `arguments_skipped=true`, 인자 판정·확률은 `null`로 남기며 생략한 검사를 실패로 세지 않습니다. 함수를 선택했지만 필수 인자를 빠뜨린 경우에는 인자 검사를 생략하지 않습니다.
- BFCL 함수는 실행하지 않습니다. 이 실험은 **호출 선택·인자 정확성**을 평가하며 실제 외부 API의 실행 성공률을 측정하지 않습니다. 응용 시스템은 `run_example(..., executor=callback)`으로 최종 실행 경계에만 연결할 수 있습니다.

비교 방법은 다음 여섯 가지입니다.

| 방법 | 실행 ID | 역할 |
|---|---|---|
| Direct | `direct` | 최초 LLM 제안만 사용 |
| Schema-only | `schema_only` | 형식·타입·필수 인자 오류 수정 |
| Self-refine | `self_refine` | 같은 LLM의 일반적인 재검토와 수정 |
| **JBFC-F** | `jev_function` | Jev의 함수 선택 검사 사용 |
| **JBFC-A** | `jev_arguments` | Jev의 인자 검사 사용 |
| **JBFC** | `jev_joint` | Jev의 함수 선택·인자 검사 모두 사용 |

Direct는 수정하지 않고, 자기 재검토는 모든 수정 예산을 사용합니다. 나머지 방법은 통과하면 일찍 종료합니다. 같은 수정 한도를 적용해도 실제 제안 수·요청 수·토큰·시간·비용은 다를 수 있습니다. **JBFC-F**와 **JBFC-A**도 호출이 있으면 두 확률을 구하지만, 선택한 의미 검사와 공통 스키마 검사만 통과 판단·피드백에 사용합니다. no-call에서는 인자 질문을 생략하므로, 인자 전용 비교인 **JBFC-A**는 스키마 검사만 적용합니다. 함수 판정을 그 방법의 통과 기준이나 피드백으로 사용하지 않습니다.

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

## 수정 한도 선택과 테스트 실행

```bash
# 개발 실험과 한도 선택까지만 실행합니다.
make tune

# 완료된 개발 기록을 재사용하고, 선택한 한도로 전체 테스트를 실행합니다.
make experiment

# 후보 범위와 저장 경로를 바꾼 추가 실험입니다.
make experiment REPAIR_GRID=0,1,2,4,8 RESULTS_DIR=results/custom_budget

# 선택된 한도에서 abstain 정책도 기존 기록으로 평가합니다.
uv run python scripts/run_suite.py --include-abstain

# 이미 완료된 개발 기록으로 선택만 다시 계산합니다. API 호출은 없습니다.
uv run python scripts/tune_repair_budget.py results/tuned/dev_budget \
  --grid 0,1,2,3,4,5 --output results/tuned/repair_budget_selection.json

# 선택된 테스트 실행의 요약만 다시 생성합니다.
uv run jev-fc report results/tuned/test_selected
```

기본 저장 위치는 `results/tuned/dev_budget`, `results/tuned/repair_budget_selection.json`, `results/tuned/test_selected`입니다. `make tune`과 `make experiment`는 같은 설정이면 완료된 기록을 재사용합니다. 개발 기록은 **JBFC**로 후보 중 가장 큰 한도에서 한 번 생성하고, 더 작은 한도는 같은 기록의 앞부분을 잘라 평가합니다. 반복 한도를 모델에게 알려주지 않고 중간에 함수를 실행하지 않으므로 앞부분의 호출과 누적 피드백은 달라지지 않습니다. 이 재평가에는 추가 API 비용이 없습니다. 테스트 결과로 공통 한도를 다시 고르지 않습니다.

선택 파일에는 후보별 개발 성능, 선택한 `r`, 선택 규칙, 모델·API·임계값 등 설정, 개발 기록의 해시가 저장됩니다. 테스트 전에 원본 개발 기록에서 선택을 다시 검증합니다. 테스트·부분 실행·합성 데이터로 만든 선택, 변경된 기록, 서로 다른 설정은 받아들이지 않습니다. 이 파일도 테스트 실행 지문에 포함하므로 다른 한도로 기존 결과에 이어 쓸 수 없습니다. 선택 자료는 모델 프롬프트에 들어가지 않습니다.

논문의 주 결과표는 **전체 개발 데이터에서 선택한 한도와 연결된 전체 테스트 결과**만 받습니다. 직접 지정한 한도로 실행한 결과나 기본 논문과 다른 모델·후보 범위·설정은 별도 요약으로 남습니다. `--skip-paper`를 사용하면 논문 반영을 생략할 수 있습니다. 기본 테스트에서는 선택한 한도보다 큰 값을 탐색하지 않으며, 더 작은 한도나 `abstain`의 선택적 재평가 결과로 `r`을 다시 선택하지 않습니다.

```bash
# 논문용 선택 절차와 별개인 탐색 실행입니다. 3보다 큰 한도도 가능합니다.
uv run jev-fc run --split dev --methods jev_joint --max-repairs 7 \
  --replicates 1 --output results/exploratory_r7 --resume

# 임계값의 선택은 별도 분석입니다. 테스트 데이터는 사용하지 않습니다.
uv run jev-fc run --split dev --methods jev_joint --max-repairs 0 \
  --replicates 1 --output results/calibration --resume
uv run python scripts/calibrate_threshold.py results/calibration
```

기본 수정 한도 실험은 Jev 임계값을 `0.8`로 고정합니다. 임계값·모델·프롬프트를 바꾸면 새로운 개발 기록으로 수정 한도를 다시 선택해야 합니다. 이전 기록의 앞부분만으로 다른 임계값이나 프롬프트의 결과를 재현할 수는 없습니다. 자세한 절차는 `docs/repair-budget.md`에 있습니다.

실행 폴더에는 다음 파일이 생깁니다.

| 파일 | 내용 |
|---|---|
| `manifest.json` | 모델, 설정, 데이터/코드 지문, 대상 ID, 단가, 검증된 수정 한도 선택 자료 |
| `initial.jsonl` | 여섯 방법이 사례·반복별로 공유하는 최초 LLM 제안 하나 |
| `records.jsonl` | 모든 후보, Jev 판단, 실제 피드백, 채점, 비용/지연 기록 |
| `errors.jsonl` | 키와 응답 내용을 제외한 오류 유형, 시각, HTTP 상태, 안전한 요청 ID와 시도 횟수 |
| `summary.json` | 카테고리별·반복별 지표, paired bootstrap, 수락·거절 진단 |
| `summary.csv` | 방법별 비교 요약 |

코드, 모델, API 형식·경로, 작업자 수, 재시도 설정, 데이터 또는 실행 조건이 바뀌면 기존 결과에 이어 쓰지 않고 새 출력 경로를 요구합니다. 키 값은 지문에 포함하지 않으므로 같은 프로젝트 키를 갱신해도 재개할 수 있습니다. 잘못된 응답을 정답으로 간주하거나 API 오류를 no-call 예측으로 바꾸지 않습니다. 완료된 기록의 중복과 설정 불일치도 보고 단계에서 검사합니다.

개발 탐색 비용은 개발 실행 기록과 선택 표에서 별도로 확인하며, 주 테스트의 요청당 비용에는 포함하지 않습니다. 후보별 앞부분 비용을 모두 합하면 같은 호출을 중복 계산하게 됩니다. 비용은 응답에 포함된 사용량과 설정된 단가로 추정합니다. Responses의 캐시 입력과 Anthropic의 캐시 읽기·5분/1시간 쓰기 토큰을 구분합니다. 필요한 단가나 사용량을 알 수 없으면 비용은 미상으로 남깁니다. 기본 단가는 공개 공급자 기준값이며 APIM 프로젝트의 가산 요금이나 실제 청구액은 아닙니다. 누락된 사용량은 0원으로 처리하지 않으며, 공급자 청구서·재시도 중 응답을 받지 못한 요청 비용까지 보장하지 않습니다.

여섯 방법의 비교 비용·지연에는 공유한 최초 제안을 각각 한 번씩 포함합니다. 이후 수정과 검사의 요청도 모두 포함하며, 누적 피드백으로 길어진 입력 토큰도 사용량에 반영됩니다. 실제 전체 실행 시간과 방법별 비교를 위한 누적 지연은 다릅니다.

## 논문

- `paper/main.tex`: 독립 실행형 영문 LaTeX, 본문 4쪽 + 참고문헌 1쪽
- `paper/main.pdf`: 확인된 PDF
- `paper/references.bib`: 별도 참고문헌 자료. 실제 원고는 내장 bibliography로 단일 파일 컴파일이 가능합니다.

Abstract, Introduction, Related Work, Method, Experiments, Conclusion, Acknowledgements가 본문에 포함됩니다. 표 3개에는 카테고리별 테스트 결과, 선택한 한도에서의 방법별 정확도/비용, 개발 데이터의 수정 한도 선택 실험을 준비했습니다. 선택값과 실험 수치는 아직 빈칸입니다. 저자는 익명으로 두었고, 특정 학회의 공식 스타일로 지정하지 않은 일반 2단 숏페이퍼입니다.

```bash
make report  # 개발 선택 자료와 연결된 기본 APIM/Responses 전체 테스트 결과 반영
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
