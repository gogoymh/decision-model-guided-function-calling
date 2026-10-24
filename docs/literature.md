# 선행연구와 실험 설계

문헌은 2026년 10월 15일에 조사했고, Jev의 추가 기능과 ReAct는 10월 16일에 확인했습니다. 참고문헌은 `paper/references.bib`에 있습니다. 이 문서는 당시 선행연구 조사와 초기 실험 계획을 보관합니다. 아래 개발 303개·테스트 2,748개 계획과 미실행 상태 설명은 현재 원고의 프로토콜이 아닙니다. 현재는 BFCL 및 탐색적 API-Bank에서 세 생성기와 한 Jev 검증 모델을 평가했으며, [최신 결과](../README.md#results), [실험 재현 안내](../README.md#reproduce-the-paper), [추가 평가 프로토콜](overleaf-extra-evaluation-protocol.md)을 확인하세요.

## Jev와 공식 자료

Jev는 TypeSafe AI의 System One 모델입니다. 상태와 정해진 유형의 질문을 입력받아 구조화된 결정을 반환합니다. Noul은 ‘예’의 확률을 반환하고, Choice는 선택한 항목, 전체 확률 분포와 confidence를 반환합니다. 직접 API는 `POST https://api.typesafe.ai/v1/systemone`입니다.

- [모델과 버전](https://docs.typesafe.ai/models), `typesafe2026models`: 확인 시점의 안정 버전은 `jev-1.13.0`이며 `jev-latest`가 가리킵니다. 10월 14일 공개된 `jev-1.14.0`은 `jev-preview`입니다. 실험에서는 별칭 대신 버전을 고정하고 API가 실제 반환한 버전도 기록합니다.
- [HTTP API](https://docs.typesafe.ai/api): 요청·응답과 오류 형식의 근거입니다.
- [Noul](https://docs.typesafe.ai/primitives/noul): ‘예’의 확률입니다. 올바름을 묻는 질문과 위험을 묻는 질문의 확률 방향을 혼동하면 안 됩니다.
- [Choice](https://docs.typesafe.ai/primitives/choice): 주어진 유한 선택지 중 하나를 고릅니다. no-call은 별도 선택지로 넣습니다. 255개 제한이 있으므로 함수 254개와 no-call 한 개까지 가능합니다.
- [Confidence](https://docs.typesafe.ai/confidence): Choice의 confidence는 선택 확률과 다릅니다. 높은 확률만으로 정답을 보장할 수 없으며, 해당 과제에서 오수락과 확률의 신뢰성을 측정해야 합니다.
- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev): Diogo Almeida가 2026년 9월 15일 쓴 TypeSafe 발표입니다. RLCD라는 학습 설명은 공급자의 설명이며 동료심사를 거친 Jev 기술 논문은 아닙니다. 이 설명을 프로젝트의 검증된 성과로 제시하지 않습니다.

이번 실험에서는 **LLM이 함수와 인자를 제안하고 Jev가 두 Noul 질문으로 검사합니다**. 공급자의 다른 기능을 방법의 근거로 제시하지 않습니다. 공식 기능과 버전별 조사 기록은 [별도 문서](jev-research.md)에 보존하며, 함수 호출 및 SDE Cascade 예제는 논문 참고문헌에서 제외했습니다.

## 직접 관련된 선행연구

| 문헌 | 확인한 출처 | 이번 실험과의 관계 |
|---|---|---|
| Mekala et al., 2024, **ToolVerifier: Generalization to New Tools via Self-Verification**, `mekala2024toolverifier` | [arXiv](https://arxiv.org/abs/2402.14158), [본문](https://arxiv.org/html/2402.14158v2). 저자는 Dheeraj Mekala, Jason Weston, Jack Lanchantin, Roberta Raileanu, Maria Lomeli, Jingbo Shang, Jane Dwivedi-Yu입니다. | 도구 선택과 인자 생성을 나누고 둘을 검증합니다. 합성 데이터로 선택 모델을 학습하고 상위 두 후보를 대비하는 질문과 인자 대안 비교를 사용합니다. 두 단계로 나누거나 검증하는 발상 자체를 새롭다고 주장할 수 없습니다. |
| Hamad et al., 2025, **ToolCritic: Detecting and Correcting Tool-Use Errors in Dialogue Systems**, `hamad2025toolcritic` | [arXiv](https://arxiv.org/abs/2510.17052). 저자는 Hassan Hamad, Yingru Xu, Liang Zhao, Wenbo Yan, Narendra Gyanchandani입니다. 확인한 자료는 프리프린트입니다. | 비평 모델이 8종류의 도구 사용 오류를 찾아 주 LLM에 수정 피드백을 제공합니다. 외부 검사를 통한 함수 호출 수정의 직접적인 선행연구입니다. |
| Gou et al., 2024, **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing**, `gou2024critic` | [arXiv](https://arxiv.org/abs/2305.11738), ICLR 2024로 확인했습니다. | 외부 도구 피드백으로 질문 답변, 프로그램, 유해 출력을 반복 수정합니다. 외부 피드백 사용 자체를 새 기여로 보지 않습니다. |
| Madaan et al., 2023, **Self-Refine: Iterative Refinement with Self-Feedback**, `madaan2023selfrefine` | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html), 36권, 46534–46594쪽. | 같은 모델의 피드백과 수정입니다. Jev의 효과를 단순한 추가 생성 기회와 구분하는 비교 기준입니다. |
| Shinn et al., 2023, **Reflexion: Language Agents with Verbal Reinforcement Learning**, `shinn2023reflexion` | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html), 36권, 8634–8652쪽. | 가중치 변경 없이 언어 피드백과 에피소드별 성찰을 활용합니다. 이번 구현에는 에피소드 기억이나 환경에서의 시행착오가 없습니다. |
| Yao et al., 2023, **ReAct: Synergizing Reasoning and Acting in Language Models** | [arXiv](https://arxiv.org/abs/2210.03629). 저자는 Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao입니다. 최초 프리프린트는 2022년이며 2023년 v3에 ICLR camera-ready임을 명시합니다. | 추론 기록과 행동을 번갈아 생성하며 외부 지식·환경과의 상호작용에서 얻은 정보를 다음 단계에 사용합니다. **DMGFC**의 누적 피드백과 개념상 관련되지만, 이 실험은 실행 전 Jev 판정을 사용하고 수정 중 함수를 실행하지 않습니다. ReAct 전체 절차의 재현은 아닙니다. |
| Agarwal et al., 2024, **Many-Shot In-Context Learning**, `agarwal2024manyshot` | [NeurIPS](https://papers.neurips.cc/paper_files/paper/2024/hash/8cb564df771e9eacbfe9d72bd46a24a9-Abstract-Conference.html), [arXiv](https://arxiv.org/abs/2404.11018). NeurIPS 2024, 37:76930–76966. 저자는 Rishabh Agarwal 외 14명(Google DeepMind)입니다. 2026-10-18 확인. | 문맥 예제를 수백·수천 개로 늘리면 여러 과제에서 성능이 크게 오르고(일부 과제는 일정 개수 이후 떨어짐), 최종 답의 정답 여부로 거른 모델 생성 풀이(Reinforced ICL)가 사람이 쓴 풀이를 대신할 수 있음을 보였습니다. **DMGFC**의 누적 기록은 같은 요청에 대한 모델 생성 후보와 *Jev* 판정의 쌍이므로, 정답 대신 *Jev*가 라벨을 단 문맥 예제로 볼 수 있습니다(판정 1건 평균 0.30초, 1,000건 약 $0.06). 다만 다른 요청의 시연 예제가 아니라 같은 요청의 실패한 시도이므로, *Jev* 라벨 예제가 다른 요청에도 도움이 되는지는 검증하지 않았습니다. |
| Huang et al., 2024, **Large Language Models Cannot Self-Correct Reasoning Yet**, `huang2024selfcorrect` | [arXiv](https://arxiv.org/abs/2310.01798), ICLR 2024로 확인했습니다. | 수정이 도움이 된다고 가정하지 말고 측정해야 한다는 근거입니다. 특정 추론 자기 수정의 부정적 결과를 모든 최신 함수 호출 모델에 일반화하거나 Jev가 효과적이라는 증거로 쓰지 않습니다. |
| Li et al., 2026, **JEV-as-a-Judge: Accept When Confident, Escalate When Unsure**, `li2026jevjudge` | [arXiv v5](https://arxiv.org/abs/2609.26550v5). 저자는 Yubo Li, Yidi Miao, Ramayya Krishnan, Rema Padman입니다. | 완성된 답변을 Jev가 판단하고 불확실할 때 독립된 강한 판단 모델로 넘깁니다. 함수 선택과 호출 수정은 평가하지 않습니다. Choice의 선택 확률과 두 Noul의 긍정 확률을 혼동하지 않아야 합니다. 전체 독해는 [별도 기록](jev-as-a-judge.md)에 있습니다. |

이번 연구는 **DMGFC**의 두 의미 검사와 누적 피드백을 이용한 한도 내 수정을 재현 가능한 코드와 평가 절차로 검증합니다. 모든 이전 후보와 실제 Jev 판정을 원래 사용자 요청 앞에 넣어 같은 LLM이 다시 제안하도록 하고, 정확도·비용·오류 변화를 측정합니다. 최신 피드백 하나만 보내는 **DMGFC-L**과도 비교합니다. 초기 조사에서는 방법과 평가 계획만 있었고, 현재 완료한 결과의 효과는 생성기와 벤치마크에 따라 다릅니다. “최초”, “보장”, “환각 제거”, “최고 성능” 같은 주장은 근거가 없으면 사용하지 않습니다.

## 벤치마크와 도구 학습 배경

| 문헌 | 확인한 출판 정보 | 활용 범위 |
|---|---|---|
| Patil et al., 2025, **The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models**, `patil2025bfcl` | [PMLR](https://proceedings.mlr.press/v267/patil25a.html), ICML 2025, 267권, 48371–48392쪽. 저자 순서는 Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, Joseph E. Gonzalez입니다. | 주 평가 자료입니다. 데이터와 채점기를 고정하고 포함 카테고리를 명시합니다. 단일 턴 하위 집합을 전체 리더보드로 부르지 않습니다. |
| Li et al., 2023, **API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs**, `li2023apibank` | [ACL Anthology](https://aclanthology.org/2023.emnlp-main.187/), EMNLP 2023, 3102–3116쪽. | 실행 가능한 API·대화 과제로 향후 외부 검증에 활용할 수 있습니다. 조사 당시에는 어댑터가 없었습니다. 2026-10-20에는 첫 호출 exact-match 변형을 구현하고 탐색적 평가를 완료했으며, 전체 대화 실행 벤치마크를 재현한 것으로 설명하지 않습니다. |
| Guo et al., 2024, **StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models**, `guo2024stabletoolbench` | [ACL Anthology](https://aclanthology.org/2024.findings-acl.664/), Findings ACL 2024, 11143–11156쪽. | 실제 API 변화에 따른 재현성 문제를 설명합니다. 캐시·시뮬레이션 서버의 평가는 단순 로컬 일치 검사와 다릅니다. |
| Qin et al., 2024, **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs**, `qin2024toolllm` | [arXiv](https://arxiv.org/abs/2307.16789), [공식 저장소](https://github.com/OpenBMB/ToolBench)에서 ICLR 2024를 확인했습니다. | ToolBench와 도구 학습의 배경입니다. ToolVerifier에서 같은 이름으로 부르는 다른 데이터와 혼동하지 않습니다. |
| Patil et al., 2024, **Gorilla: Large Language Model Connected with Massive APIs**, `patil2024gorilla` | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e4c61f578ff07830f5c37378dd3ecb0d-Abstract-Conference.html), 37권, 126544–126565쪽. | 함수 호출 학습과 API 환각의 배경입니다. 프리프린트는 2023년이지만 인용한 학회판은 2024년입니다. |

짧은 논문에서는 직접 관련된 검증 연구를 우선합니다. 참고문헌의 URL과 출판 정보는 `.bib`에 보존합니다.

## 보관한 초기 실험 계획: 현재 원고에 적용하지 않음

### **DMGFC**와 누적 수정 문맥

- **제안**은 사용자 요청 하나에 대한 함수 이름과 인자입니다. no-call은 빈 호출 목록입니다. 현재 데이터는 단일 호출 또는 no-call이며 병렬·순차 실행을 평가하지 않습니다.
- **DMGFC**의 함수 질문은 전체 함수 목록과 요청에 비추어 선택한 함수, 호출 수, no-call이 맞는지 묻습니다. 이름이 존재하는지만 확인하지 않습니다. 인자 질문은 값의 근거, 역할, 필수 값, 함수 정의와의 일치를 묻습니다.
- **DMGFC**는 호출이 있으면 같은 후보에서 `schema_valid AND selection_pass AND arguments_pass`여야 통과합니다. no-call은 `schema_valid AND selection_pass`로 판단합니다. 파싱 가능한 후보는 스키마 오류가 있어도 의미 검사를 수행하지만, 스키마 오류는 최종 통과를 막습니다. 수정 후 필요한 검사를 다시 하며 이전 통과를 재사용하지 않습니다.
- 최초 요청은 원본 메시지를 유지하고 피드백을 추가하지 않습니다. **DMGFC**의 수정 때는 모든 이전 후보의 `round`, `calls`, 실제 `feedback`를 시간순으로 담은 사용자 메시지 하나를 원본 대화 앞에 붙입니다. **DMGFC-L**은 최신 실패 후보·피드백 하나만 담되 원래 라운드 번호를 유지합니다. 두 방법 모두 원래 사용자 요청과 메시지 내용·순서를 보존하고, 이전 후보를 실행하지 않았음을 명시합니다. 저장한 추적에는 모든 라운드가 남습니다.
- 누적 기록에는 Jev가 반환한 확률·판정, 임계값, 로컬 스키마 오류를 사용합니다. Jev가 생성하지 않은 설명이나 정답에서 만든 힌트를 추가하지 않습니다. 함수와 인자는 다음 후보에서 모두 수정할 수 있습니다.
- no-call은 LLM의 빈 호출 제안입니다. 함수 질문은 그 빈 제안이 적절한지 묻고, 인자 질문은 보내지 않습니다. 인자 검사는 생략으로 표시하며 판정·확률은 `null`로 기록합니다. 함수는 선택했지만 필수 인자가 빠진 후보는 인자 검사 대상입니다.
- 파싱 실패와 스키마 실패를 구분합니다. Schema-only 비교는 의미 검증의 효과를 단순한 형식 수정과 분리합니다. API 오류를 검사 통과나 no-call로 바꾸지 않습니다.
- 수정 한도 `r`은 0 이상의 정수입니다. 함수 호출이 필요한 사례에서 최초 제안 이후 최대 `r`회 수정하므로 제안은 최대 `r+1`개입니다. 기본 후보 `0,1,2,3,4,5`는 탐색 범위이며 구현의 고정 상한이 아닙니다.
- 기본 `execute_last`는 끝까지 거부된 마지막 후보도 채점합니다. `abstain`은 거부된 후보를 실행 대상으로 넘기지 않습니다. 보류된 오답을 수정 성공으로 세지 않습니다.
- 실제 BFCL 함수는 실행하지 않습니다. 공식 AST 채점은 호출 정확도의 검사이며 실제 API 성공이나 안전한 후속 행동을 입증하지 않습니다.

### 데이터 분할과 정보 누출 방지

1. 데이터·채점기 버전, ID 순서, 카테고리, 전처리, 모델, 프롬프트, 임계값, 샘플링과 반복 설정을 테스트 전에 고정합니다.
2. 개발·테스트는 카테고리별 ID 해시로 나누고 분할 규칙과 수를 기록합니다. 공개 데이터의 숨겨진 정답 집합이라는 주장은 하지 않습니다. 함수 템플릿 공유나 학습 데이터 포함 가능성은 한계입니다.
3. 개발 데이터로 수정 한도를 선택합니다. 기본 Noul 임계값은 조정하지 않은 `0.8`입니다. 별도 임계값 분석도 개발 데이터에서만 합니다. 테스트 정답은 모델에 전달하지 않습니다.
4. 모델이 보는 것은 원본 대화, 공개 함수 정의, 제안된 호출, 실제 검사 피드백뿐입니다. 누적 수정 문맥도 이 정보만 사용합니다. 정답, 허용 인자 목록, 채점 결과·오류, 데이터 ID·카테고리, 개발 한도 선택 파일은 제외합니다.
5. 지원하지 않는 카테고리나 필요한 정답이 없는 문제의 처리 기준을 사전에 정합니다. `live_relevance`는 호출 유무 지표로 따로 보존합니다. 어려운 실패를 채점 단계에서 조용히 제외하지 않습니다.
6. 공급자 오류와 방법 실패를 구분합니다. 일시 오류는 별도 한도의 재시도로 처리하고 재개 가능한 기록을 남깁니다. 반환 모델, 사용량, 시간, 설정 해시를 저장하되 키와 원본 오류 본문은 저장하지 않습니다.
7. 공급자 API 형식과 함수 호출 방식은 실험 조건입니다. 다른 인터페이스 결과를 같은 조건으로 합치지 않습니다.

### 일곱 비교 방법

| 방법 | 실행 ID | 비교 목적 |
|---|---|---|
| Direct | `direct` | 최초 제안만 사용 |
| Schema-only | `schema_only` | 구조·타입 오류만 수정 |
| Self-refine | `self_refine` | 같은 LLM의 일반적인 재검토와 수정. 추가 생성 기회의 효과 확인 |
| **DMGFC-F** | `jev_function` | 함수 선택 검사만 통과 판단과 수정에 사용 |
| **DMGFC-A** | `jev_arguments` | 인자 검사만 통과 판단과 수정에 사용 |
| **DMGFC** | `jev_joint` | 두 의미 검사를 모두 사용 |
| **DMGFC-L** | `jev_joint_latest` | 같은 검사를 사용하며 최신 실패 후보·피드백 하나만 제공해 누적 문맥의 기여 비교 |

**DMGFC-F**와 **DMGFC-A**는 호출이 있으면 두 Noul을 같은 요청에서 구하지만 지정한 검사와 공통 스키마 검사만 판단·피드백에 사용합니다. no-call에서는 인자 질문을 생략합니다. Self-refine은 전체 수정 한도를 사용하며 다른 수정 방법은 통과하면 일찍 종료할 수 있습니다.

**DMGFC-L**은 **DMGFC**와 모델, 필요한 검사, 임계값, 공유 최초 제안, 개발에서 선택한 공통 `r`, 최종 정책이 같습니다. 차이는 생성기에 제공하는 이전 기록의 범위뿐이며 두 번째 수정부터 입력이 달라집니다. 따라서 `r≤1`이면 누적 문맥의 기여를 평가할 수 없습니다. 별도의 실제 LLM 궤적이 필요하므로 누적 **DMGFC** 기록으로 최신 피드백 결과를 합성하지 않습니다.

수정 한도는 **DMGFC**의 전체 개발 303개를 3회 실행해 고릅니다. 가장 큰 후보 한도로 실행하고 같은 기록의 앞부분으로 작은 한도를 평가합니다. 기록은 909개이며, `live_relevance` 1개를 반복마다 제외한 **906개**의 최종 정확도가 선택 기준입니다. 정확도가 가장 높은 `r`을 고르고 동점이면 작은 값을 택합니다. 선택 파일과 원본을 검증한 후에만 전체 테스트를 시작합니다.

일곱 방법은 선택된 공통 `r`로 **2,748개 × 3회 × 7방법 = 57,708개** 기록을 만듭니다. 당시 `make experiment` 계획에 일곱 방법을 포함했으며 Direct는 수정하지 않습니다. 현재 Makefile에는 이 대상이 없고 탐색 코드는 `scripts/run_suite.py`에 보존되어 있습니다. 테스트 결과나 작은 한도로 재평가한 테스트 기록을 이용해 `r`을 다시 선택하지 않습니다.

앞부분 재평가는 한도가 프롬프트에 없고 중간 함수 실행도 없기 때문에 가능합니다. 더 작은 한도에서 이전 제안과 피드백이 달라지지 않습니다. API 오류 뒤의 누락된 시도를 만들어 채우거나, 같은 기록의 재평가를 독립된 반복 실험으로 세지 않습니다. 개발 탐색 비용은 테스트 비용과 따로 보고하며 후보별 중복 구간의 비용을 모두 합하지 않습니다.

일곱 방법은 사례·반복별로 같은 최초 LLM 제안 하나를 공유하고 방법별 비교 비용에는 그 제안을 각각 한 번 포함합니다. 이후 수정 입력에 실제로 포함한 피드백의 토큰과 검사 비용도 포함합니다. 같은 한도는 같은 제안 수·토큰·시간·비용을 뜻하지 않습니다. 더 강한 비용 통제가 필요하면 관측된 비용이나 지연에 맞춘 비교를 추가해야 합니다. 정답을 보고 여러 샘플 중 최선을 고르는 방식은 배포 가능한 비교 기준이 아닙니다.

### 지표와 분모

완전한 정답이 있는 평가 사례 수를 `N`, 최종 정답 여부를 `C_i`, 방법의 수락 여부를 `A_i`라고 합니다. 모든 지표는 분모와 함께 보고합니다.

| 지표 | 계산과 해석 |
|---|---|
| 최종 정확도 | `sum(C_i)/N`. 거부된 최종 후보도 공식 기준으로 채점합니다. |
| 함수 선택 정확도 | 정답 함수 이름·호출 수와 일치하는 비율. 올바른 빈 호출도 포함합니다. |
| 조건부 인자 정확도 | 함수 선택이 맞고 인자 정답이 있는 사례 중 인자가 맞는 비율. 전체 정확도와 함께 보고합니다. |
| 수락 비율 | `sum(A_i)/N`. 실제 실행으로 넘긴 비율이나 의미 검사 통과 수와 같다고 가정하지 않습니다. |
| 수락한 후보의 정확도 | `sum(A_i*C_i)/sum(A_i)`. 수락 사례가 없으면 `NA`입니다. |
| 수락한 후보 중 오답 비율 | `sum(A_i*(1-C_i))/sum(A_i)`. 수락 정확도의 보수입니다. |
| 오답 중 오수락 비율 | `sum(A_i*(1-C_i))/sum(1-C_i)`. 위 지표와 분모가 다릅니다. |
| 정답 중 오거절 비율 | `sum((1-A_i)*C_i)/sum(C_i)`. |
| 수정과 퇴행 | 처음 오답이 정답으로 바뀐 비율과 처음 정답이 오답으로 바뀐 비율. 실제 최초·최종 제안을 비교합니다. |
| 과정 진단 | 제안 수, 통과 시점, 한도 소진, 파싱·스키마 오류, API 오류, no-call 정확도. |
| 효율 | 실제 생성·검사 요청 수, 입력·출력 토큰, 날짜가 명시된 단가의 추정 비용, 평균·중앙·p95 지연. 재시도와 네트워크 시간을 포함하며 실행 동시성을 기록합니다. |
| 확률 진단 | 선택적 분석으로 Brier 점수나 구간별 정답률을 측정합니다. 두 질문의 오류 독립성을 가정하거나 확률을 곱해 결합 정확도로 해석하지 않습니다. |

Direct와 각 방법의 차이는 같은 사례끼리 계산합니다. 주 비교에는 고정 seed·반복 수의 paired bootstrap으로 95% 구간을 구하며 반복별 결과도 남깁니다. 서로 독립적으로 만든 신뢰구간의 겹침만으로 유의성을 주장하지 않습니다. 누락된 사용량의 비용은 미상으로 두고 0원으로 세지 않습니다. 누적 관측 지연과 전체 작업의 실제 경과 시간, 추정 단가와 공급자 청구액을 구분합니다.

`summary.json`의 `history_ablation`은 **DMGFC − DMGFC-L**의 최종 호출 정확도 차이입니다. 정확히 같은 `(사례 ID, 반복)`을 짝지어 차이를 구하고, ID별 반복 평균을 낸 뒤 seed 1729로 ID를 2,000회 재표집해 95% 구간을 계산합니다. 공유 최초 제안·점수, 수정 한도와 최종 정책의 일치를 확인합니다. `live_relevance`를 제외하고 no-call을 포함한 전체 분모는 **2,733개 ID, 8,199개 짝지은 기록**입니다.

### 네 쪽 원고의 결과표

원고에는 카테고리별 테스트 결과, 공통 한도에서 일곱 방법의 정확도·비용 비교, 개발 한도 선택 표를 둡니다. 더 자세한 조건부 인자 정확도, 오수락·오거절, 수정·퇴행 등은 결과 파일에 보존합니다.

측정하지 않은 셀은 `--`로 두고 API 실험 전이라는 점을 명시합니다. 실제 준비 과정에서 센 데이터 수는 기입할 수 있지만 모델 결과와 구분합니다. 실험 후에는 개선이 없거나 악화되더라도 그 결과에 맞게 abstract·논의·한계·결론을 수정해야 합니다.

## 서지 정보 주의점

- BFCL 저자 순서는 웹사이트의 축약 표기가 아닌 PMLR 학회판을 따릅니다.
- Reflexion은 현재 arXiv와 NeurIPS 학회판의 저자 수가 다릅니다. 여기서는 학회판의 다섯 저자를 일관되게 사용합니다.
- ToolVerifier와 ToolCritic은 확인한 원본에서 학회가 확정되지 않아 프리프린트로 인용합니다.
- ToolVerifier의 ToolBench는 weather·booking·home·cat 네 API 영역을 다룹니다. 이름이 같아도 OpenBMB의 16,000개 API ToolBench와 동일시하지 않습니다.
- Jev 버전 별칭과 공식 예제는 변할 수 있습니다. 조회 날짜와 고정 모델, 프롬프트 및 실행 설정을 보존합니다.
