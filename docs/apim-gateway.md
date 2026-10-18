# APIM 게이트웨이 점검과 실행

사용자가 제공한 `apim-llm-gateway-curl 퀵 가이드`를 기준으로 2026-10-15에 연결 코드를 점검했습니다. 원본 HTML과 인증 값은 저장소에 포함하지 않습니다.

## 확인한 차이와 수정

| 항목 | 기존 코드 | 반영한 방식 |
|---|---|---|
| 인증 | 공급자별 Bearer 키 | APIM 모드에서 공통 `api-key` 헤더와 프로젝트 구독 키 |
| OpenAI | `/chat/completions`, `messages`, 중첩 함수 스키마 | `/llm/openai/v1/responses`, `input`, 평탄한 함수 스키마 |
| Anthropic | 미지원 | `/llm/anthropic/v1/messages`, `input_schema`, `tool_use` |
| Jev | 공급자 직접 `/v1/systemone` | APIM `/llm/typesafe/v1/systemone`; Noul 검증 방식 유지 |
| 사용량 | Chat Completions 토큰 | 공급자 원본 사용량 보존 + 공통 입력/출력/캐시 집계 |
| 재개 | 모델·기본 URL 중심 | 공급자·API 형식·실제 경로·작업자·재시도 설정까지 지문 기록 |

APIM에서는 `Authorization`, `x-api-key`, `anthropic-version`을 보내지 않습니다. 가이드에 따르면 Anthropic에 필요한 공급자 헤더는 게이트웨이가 주입합니다. `Content-Type: application/json`과 프로젝트 `api-key`만 인증에 사용합니다.

Responses 요청에는 `strict: false`를 명시합니다. 이를 생략하면 API가 스키마를 엄격 모드로 정규화하면서 BFCL의 선택적 인자 조건이 달라질 수 있습니다. 함수 이름·설명·필수 인자·허용 속성을 그대로 유지하며, 원래 코드와 동일하게 한 후보당 함수 호출은 최대 하나입니다. `store: false`를 명시하고 이전의 미실행 후보와 검사 피드백은 매 수정 요청에 직접 전달합니다.

최초 요청에는 피드백을 추가하지 않습니다. 수정 요청에서는 모든 이전 후보와 실제 피드백을 시간순 JSON 목록으로 묶어 사용자 문맥 메시지 하나를 원본 대화 앞에 붙입니다. 원래 사용자 요청과 기존 메시지의 내용·순서를 유지하며, 이전 후보를 실행하지 않았다고 명시합니다. 이 의미를 Responses, Chat Completions, Anthropic Messages에 동일하게 적용하되 공급자별 메시지·시스템 입력 형식은 각 어댑터가 변환합니다. 수정 중에는 도구 실행 결과를 만들어 넣지 않습니다.

## 기본 실행

현재 `.env`에는 APIM 연결 설정이 반영되어 있습니다. `APIM_SUBSCRIPTION_KEY`에 프로젝트 키를 넣으세요. `.env`는 Git과 배포 묶음에서 제외됩니다.

```bash
make check-config
make smoke
make table1 WORKERS=8   # Table 1: Direct와 JBFC
make table2 WORKERS=8   # Table 1·2: 일곱 방법
```

`check-config`는 네트워크 요청 없이 실제 경로, API 형식, 모델, 키 설정 여부와 단가 설정만 확인합니다. 키 값은 출력하지 않습니다. 키가 없으면 경로를 보여준 뒤 종료 코드 1을 반환합니다. `smoke`부터 실제 요청과 비용이 발생합니다. 가이드는 사용 가능한 모델 목록이나 프로젝트 권한을 보장하지 않으므로 실제 계정·모델 접근은 이 단계에서 확인해야 합니다.

코드 기본값은 `LLM_API_STYLE=responses`, `LLM_MODEL=gpt-5.4-mini-2026-03-17`, `TYPESAFE_MODEL=jev-1.13.0`입니다. 논문 표를 만드는 `make table1`·`make table2`는 Azure 배포 경로(`LLM_API_STYLE=azure_chat_completions`, `gpt-5-nano`, reasoning effort `none`)를 직접 지정합니다. 모델을 바꾸면 새 실행 경로를 사용하세요.

Anthropic을 사용하려면 다음을 설정합니다. 모델 ID는 프로젝트에서 사용할 수 있는 실제 값으로 지정합니다.

```dotenv
API_GATEWAY=apim
LLM_API_STYLE=anthropic_messages
LLM_MODEL=<허용된 Anthropic 모델 ID>
LLM_REASONING_EFFORT=
LLM_INPUT_PRICE=<입력 단가>
LLM_CACHED_INPUT_PRICE=<캐시 읽기 단가>
LLM_OUTPUT_PRICE=<출력 단가>
```

캐시 쓰기가 보고되면 `LLM_CACHE_WRITE_5M_PRICE`와 `LLM_CACHE_WRITE_1H_PRICE`도 필요합니다. API 응답이 쓰기 수명이나 필요한 사용량을 제공하지 않거나 필요한 단가가 없으면 비용을 미상으로 표시합니다. 새 템플릿의 빈 단가는 기본 OpenAI 모델에만 공개 기준 단가로 해석됩니다. 다른 모델에는 GPT 단가를 자동 적용하지 않습니다. 기존 `.env`에 단가 숫자가 이미 있으면 명시적 설정으로 취급하므로 모델을 바꿀 때 함께 수정하세요.

`make table1`·`make table2`는 다른 배포(`MODEL=...`)로도 실험과 요약까지 완료합니다. 원고는 `gpt-5-nano`·Azure Chat Completions·`r=2`·전체 예제·1회 실행을 기술하므로 다른 설정의 결과는 논문 표에 자동 삽입하지 않습니다. 별도로 논문의 실험 조건을 바꾼 뒤 반영해야 합니다.

## Jev만 직접 호출하기 (Table 3, 2026-10-18)

- 게이트웨이 뒤의 TypeSafe 계정이 HTTP 402(Payment Required)를 반환해, Table 3의 `r=10` 확장에서는 Jev만 TypeSafe API로 직접 호출했습니다. 생성기는 계속 게이트웨이를 씁니다.
- `JEV_GATEWAY=direct`(또는 `scripts/extend_run.py --jev-gateway direct`)이면 Jev 요청은 `TYPESAFE_API_KEY`와 `TYPESAFE_BASE_URL`(비우면 `https://api.typesafe.ai`)을 씁니다. 이 키는 TypeSafe로만 가고, 게이트웨이 키는 생성기에만 쓰입니다. `make table3`은 기본값으로 `direct`를 씁니다. 비우면 `API_GATEWAY`를 따릅니다.
- 확장 실행은 원본 실행과 Jev 경로만 다를 수 있고, 경로 변경을 요청했을 때만 허용합니다. Jev 모델·임계값·질문은 같아야 합니다. 실행 매니페스트의 `jev_routes`에 시도 번호별 경로가 남습니다(Table 3: 시도 0–7 게이트웨이, 8–10 직접). Table 3 캡션에도 자동으로 적힙니다.
- `make table1`·`make table2`(`jev-fc run`)는 지금도 `API_GATEWAY` 하나로 생성기와 Jev를 함께 연결합니다. 이 경로를 바꾸려면 `src/jev_fc`를 고쳐야 하며, 그러면 기존 실행을 이어 돌릴 수 없으므로 Table 3 이후로 미뤘습니다.

## 기록과 오류

응답의 실제 모델명과 요청 모델명을 분리해 저장합니다. 실제 모델명이 응답에 없으면 관측된 모델명을 만들어내지 않습니다. 요청/응답 형식이 바뀌면 다른 실험 설정으로 취급하며 이전 실행에 섞지 않습니다. 키 값은 지문에 포함하지 않으므로 동일 프로젝트 키의 교체는 재개를 막지 않습니다.

인증 오류·제한·전송 오류·불완전 응답·명시적 거절은 정상적인 no-call 예측으로 변환하지 않습니다. 정상적으로 종료된 텍스트 응답은 no-call로 채점하며, 함수 인자의 잘못된 JSON은 잘못된 후보로 보존해 수정 기회를 줍니다. 오류 기록은 시각, 오류 유형, HTTP 상태, 시도 횟수와 검증된 요청 ID만 포함합니다. 응답 본문과 헤더 전체는 저장하지 않습니다.

지연 시간에는 APIM 경유와 재시도 시간이 포함됩니다. 토큰 비용은 설정된 기준 단가에 따른 추정이며, APIM 가산 요금·공급자 청구서·응답을 받지 못한 재시도 비용을 재현하지 않습니다.

## 검증 범위와 공식 근거

모의 HTTP 응답으로 모든 경로의 인증, 함수 호출 변환, no-call/오류 구분, 수정 반복, 재개, 정답 유출 방지와 비용 집계를 검사했습니다. 실제 APIM 호출과 성능 실험은 수행하지 않았습니다.

- [OpenAI 함수 호출과 strict 모드](https://developers.openai.com/api/docs/guides/function-calling)
- [OpenAI Responses 생성 API](https://developers.openai.com/api/reference/resources/responses/methods/create)
- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages/create)
- [TypeSafe API](https://docs.typesafe.ai/api)
