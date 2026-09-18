# Variant Interpretation Agent (변이 해석 에이전트)

간단한 tool-use(함수 호출) 기반 LLM 에이전트로, 유전체 변이를 해석한다.
먼저 기존에 라벨링된 변이 데이터셋을 조회하고, 처음 보는 변이라면 병원성
예측 모델로 넘어가며, 모델 확신도가 낮으면 추측 대신 "전문가 검토 필요"로
표시한다.

기존에 진행 중이던 조직 특이적 DNA 변이 병원성 예측 프로젝트(DNABERT-2 +
후성유전학 신호)를 확장한 형태이다. 처음부터 새 에이전트 프로젝트를
시작하는 대신, 이미 동작하는 모델 위에 LLM 오케스트레이션 레이어를 얹었다.

## 구조

## 구조

```
             
              ┌─────────────────────────┐
사용자질문－＞ │     LLM (Claude API)    │  
              └────────────┬────────────┘

                           │
              ┌────────────▼────────────┐
              │  1. lookup_clinvar      │  찾음 -> 바로 라벨 보고, 종료
              └────────────┬────────────┘
                           │ 못 찾음
              ┌────────────▼─────────────┐   ┌───────────────────────────┐
              │ 2. get_reference_sequence│   │ 3. get_epigenomic_signal  │
              │   ALT 치환된 DNA 서열     │   │   H3K27ac/DNase, ±512bp   │
              └────────────┬─────────────┘   └──────────────┬────────────┘
                           └───────────────┬────────────────┘
                              ┌────────────▼─────────────┐
                              │ 4. predict_pathogenicity │  실제 Late Fusion
                              │  (model.py, DNABERT-2 +  │  모델
                              └────────────┬─────────────┘
                                           ▼
              확신도 0.7 이상 -> 근거(모델 확신도, 신호값 등)를
                                 명시하며 병원성/양성 판정
              확신도 0.7 미만 -> 판정 대신 "전문가 검토 필요"로 표시
```

## 상태

| 도구 | 상태 |
|---|---|
| 1. `lookup_clinvar` | real (실데이터 연결 시) |
| 2. `get_reference_sequence` | real — `/workspace/hg38.fa` 기준 ALT 치환 위치까지 검증 |
| 3. `get_epigenomic_signal` | real — train-only z-score 정규화 적용 확인 |
| 4. `predict_pathogenicity` | real — 자체 학습한 Late Fusion 체크포인트로 검증 (아래 "모델" 참고) |

`model.py`는 팀의 `baseline2_latefusion.py`(`LateFusionModel` — 서열 + epi_signal
+ tissue_id)를 기반으로 만들었다. 같은 저장소의 `predict.py`는 입력 형태가 다른
별도 모델(stage1, DNA-only)을 불러오는 스크립트라 사용하지 않았다.

## 파일 구성

- `agent.py` — tool-use 루프와 4개 도구용 system prompt
- `tools.py` — 4개 도구 구현체. `extract_signals_max.py`와 hg38 FASTA를
  같은 폴더에 두고 경로만 지정하면 real 모드로 동작
- `model.py` — `LateFusionModel` 구조 + 체크포인트 로드/추론 헬퍼
- `train.py` — 자체 학습 스크립트 (아래 "모델" 참고)
- `build_eval_set.py` — held-out test split에서 라벨 있는 평가셋 샘플링
- `normalize.py` — train-only z-score 계산/적용 (leakage 버그 수정 로직)
- `evaluate.py` — 라벨 있는 테스트셋을 에이전트에 돌려 결과를 `correct`,
  `flagged_for_review`, `wrong_coordinate_parsing`, `tissue_confusion`,
  `tool_call_omission`, `unsupported_claim`, `incorrect`, `no_final_answer`로 분류
- `data/sample_variants.csv` — 합성 예시 데이터
- `data/real_eval_variants.csv` — held-out test split에서 뽑은 실제 라벨 변이 90개

## 설치

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 실행

```bash
python agent.py                              # 단일 예시 질의
python build_eval_set.py                     # 실제 평가셋 생성
python evaluate.py data/real_eval_variants.csv   # 전체 평가 실행
```

## 모델

팀의 공식 baseline2(Late Fusion) 체크포인트는 GPU 서버·팀 GitHub·팀 공유 Drive
링크 어디에서도 지금 코드와 맞는 버전을 찾지 못해, 동일한 아키텍처를 직접
학습시켰다. Train-only로 재정규화한 실제 데이터(leakage 수정 완료본)로 단일
seed, 축소된 epoch으로 학습한 결과 macro AUPRC 0.28을 기록했다 — 팀이 3-seed,
30-epoch 튜닝으로 낸 0.88과는 차이가 있으며, 파이프라인이 실제 체크포인트로
end-to-end 작동함을 검증하는 목적에 한정된다.

## 평가

`build_eval_set.py`로 학습에 사용되지 않은 held-out test split에서 조직당
병원성 15개, 양성 15개씩 총 90개 변이를 뽑아 `data/real_eval_variants.csv`로
준비해두었다. 도구 4개는 각각 실제 데이터로 개별 검증했으며, `evaluate.py`를
통한 전체 자동 평가는 Anthropic API 비용 문제로 아직 실행하지 않았다 — 실행
준비는 끝난 상태다.

## 알려진 한계

- 자체 학습한 체크포인트의 성능(macro AUPRC 0.28)은 팀의 튜닝된 결과(0.88)보다
  낮다. 파이프라인 검증 목적으로는 충분하나, 성능 자체를 대표하지 않는다.
- `evaluate.py`의 전체 자동 실행(실패 유형 분석 포함)은 API 비용으로 인해 아직
  진행하지 않았다.
- "전문가 검토 필요" 기준값(0.7)은 첫 추정치이며, 실제 precision/recall
  트레이드오프에 맞춰 튜닝되지 않았다.
- RAG(검색 증강) 요소가 없다 — ACMG 변이 분류 가이드라인을 근거로 함께
  제시하는 것이 자연스러운 다음 확장 지점이다.










