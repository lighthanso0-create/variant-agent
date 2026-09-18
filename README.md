# Variant Interpretation Agent (변이 해석 에이전트)

간단한 tool-use(함수 호출) 기반 LLM 에이전트로, 유전체 변이를 해석합니다.
먼저 기존에 라벨링된 변이 데이터셋을 조회하고, 처음 보는 변이라면 병원성
예측 모델로 넘어가며, 모델 확신도가 낮으면 추측 대신 "전문가 검토 필요"로
표시합니다.

기존에 진행 중이던 조직 특이적 DNA 변이 병원성 예측 프로젝트(DNABERT-2 +
후성유전학 신호)를 확장한 형태입니다 — 처음부터 새 에이전트 프로젝트를
시작하는 대신, 이미 동작하는 모델 위에 LLM 오케스트레이션 레이어를
얹었습니다.

## 이 프로젝트를 만든 이유

Inocras 인턴십 지원서의 "LLM API 활용 수준" 문항이 다루는 두 역량(LLM
tool-use, 멀티스텝 에이전트 설계·평가)을 실제로 손에 익히기 위해 2주 집중
프로젝트로 진행했습니다.

## 구조

```
                 ┌─────────────────────┐
   사용자 질문 -> │  LLM (Claude API)     │
                 └─────────┬────────────┘
                           │
              ┌────────────▼─────────────┐
              │  1. lookup_clinvar        │  찾음? -> 바로 라벨 보고, 종료
              └────────────┬─────────────┘
                           │ 못 찾음
              ┌────────────▼─────────────┐   ┌───────────────────────────┐
              │ 2. get_reference_sequence │   │ 3. get_epigenomic_signal  │
              │   ALT 치환된 DNA 서열     │   │   H3K27ac/DNase, ±512bp   │
              └────────────┬─────────────┘   └─────────────┬─────────────┘
                           └───────────────┬────────────────┘
                              ┌────────────▼─────────────┐
                              │ 4. predict_pathogenicity  │  실제 Late Fusion
                              │  (model.py, DNABERT-2 +   │  모델 — predict.py의
                              │   epi signal + tissue)    │  모델이 아님
                              └────────────┬─────────────┘
                                           ▼
              근거를 밝힌 판정, 또는 모델 확신도가
              0.7 미만이면 "전문가 검토 필요"
```

도구 2~4는 각각 REAL 경로와 MOCK 대체 경로(값이 정해진 형태로 명확히
표시됨)를 갖고 있어서, 실제 파일이 없어도 파이프라인 전체가 일단 돌아가게
되어 있습니다. **GPU 서버에서 직접 테스트한 현재 상태:**

| 도구 | 상태 | 확인 방법 |
|---|---|---|
| 1. `lookup_clinvar` | 실데이터 연결 시 real | 단순 CSV 조회라 mock/real 구분 자체가 없음 |
| 2. `get_reference_sequence` | ✅ **real, 검증 완료** | `source: real_reference_fasta`; `/workspace/hg38.fa` 기준으로 ALT 염기가 정확한 중앙 위치에 들어간 것까지 확인 |
| 3. `get_epigenomic_signal` | ✅ **real, 검증 완료** | `source: real_bigwig_extraction`; 실제 train-only z-score 통계로 정규화된 값 확인 |
| 4. `predict_pathogenicity` | ⚠️ **mock — 체크포인트 없음** | 아래 "알려진 한계" 참고 |

**중요:** 여기 있는 `model.py`는 팀의 `baseline2_latefusion.py`(`LateFusionModel`
— 서열 + epi_signal + tissue_id, Macro AUPRC 0.880)를 기반으로 만들었습니다.
`predict.py`(다른, stage1 DNA-only 모델을 불러오는 스크립트)를 그대로
쓴 게 아닙니다 — 입력 형태 자체가 다르므로 둘을 혼동해서 바꿔치기하면
안 됩니다.

## 파일 구성

- `agent.py` — tool-use 루프와 4개 도구용 system prompt.
- `tools.py` — 4개 도구 구현체. `extract_signals_max.py`와 hg38 FASTA
  (`REFERENCE_FASTA_PATH`)를 같은 폴더에 두면 도구 2~3이 mock에서 real로
  자동 전환됩니다 — 다른 코드 수정은 필요 없습니다.
- `model.py` — `baseline2_latefusion.py`에서 그대로 가져온 실제
  `LateFusionModel` 구조 + 체크포인트를 불러와 변이 하나를 추론하는
  `predict()` 헬퍼. `MODEL_CHECKPOINT_PATH`를 실제 `best_model.pt`로
  지정하면 real 모드로 동작합니다.
- `normalize.py` — train-only z-score 계산/적용(leakage 버그 수정
  로직), `get_epigenomic_signal`의 3단계에서 재사용됩니다.
- `evaluate.py` — 라벨이 있는 테스트셋을 에이전트에 돌려서, 결과를
  `correct`, `flagged_for_review`, `wrong_coordinate_parsing`,
  `tissue_confusion`, `tool_call_omission`, `unsupported_claim`,
  `incorrect`, `no_final_answer` 중 하나로 분류합니다.
- `data/sample_variants.csv` — **합성 예시 데이터**이며 실제 ClinVar
  기록이 아닙니다. 결론을 내리기 전에 실제 라벨링된 split으로 교체해야
  합니다.

## 설치

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 실행

```bash
python agent.py                        # 단일 예시 질의
python evaluate.py data/sample_variants.csv   # 전체 평가 실행
```

## 실제 모델·데이터 연결 현황

1. ✅ `KNOWN_VARIANTS_PATH` — 실제 캡스톤 split 파일이 준비되면 그걸
   가리키면 됩니다(지금은 합성 데이터인 `data/sample_variants.csv` 사용 중).
2. ✅ **완료.** `extract_signals_max.py`를 `tools.py` 옆에 복사했고,
   `ZSCORE_STATS_PATH`를
   `/workspace/signals_seed42_zscore_trainonly_0904/zscore_stats_trainonly.npy`로
   지정, real 출력 확인함.
3. ✅ **완료.** `REFERENCE_FASTA_PATH`를 `/workspace/hg38.fa`(UCSC에서
   다운로드, `pysam.faidx`로 인덱싱)로 지정, real 출력 확인함 — 정확한
   위치에 정확한 ALT 염기가 들어감.
4. ❌ **막힘.** `MODEL_CHECKPOINT_PATH` — 동작하는 baseline2(Late Fusion)
   체크포인트를 아직 찾지 못했습니다. "알려진 한계" 참고.
5. 4번이 해결되면 `evaluate.py`를 다시 돌려서 아래 "평가 결과"를 실제
   수치로 교체합니다.

## 평가 결과

_아직 실행 안 함 — 모델 체크포인트 문제로 막혀 있습니다(알려진 한계 참고).
도구 1~3은 real이고, 지금 돌리면 4단계(모델 추론)만 mock으로 남습니다._

| 지표 | 값 |
|---|---|
| 정확도 | — |
| 전문가 검토로 넘어간 비율 | — |
| 관찰된 실패 유형(건수 포함) | — |

## 알려진 한계

- **`predict_pathogenicity`가 mock 상태로 동작합니다 — baseline2(Late
  Fusion) 체크포인트를 찾지 못했습니다.** 확인한 순서:
  1. GPU 서버 (`find /workspace -iname "*.pt"` / `*checkpoint*"` —
     아무것도 안 나옴). `baseline2_latefusion.py`가 저장하는
     `outputs/` 구조 자체가 이 서버엔 없음 — 즉 최종 학습 결과가 이
     서버에 저장/실행되지 않았다는 뜻.
  2. 팀 GitHub 저장소(`gLM-with-ephigenomic`) — `baseline2_latefusion.py`
     (학습·모델 구조 코드)는 있지만 체크포인트 파일은 없음. 체크포인트는
     용량이 커서 일반 git 저장소 대신 Drive로 공유됨.
  3. 팀 카톡에 공유된 Google Drive 링크 3개(5/15, 5/18, 6/3) — 각각
     최종 baseline2 결과가 아닌 다른 것으로 명시되어 있음(수정 전
     seed42 버전, 학습률 실험, baseline3). 6/3 팀 진행상황 공유에서
     그 시점까지도 baseline2 학습이 GPU 자리 부족으로 진행 중이었다고
     확인됨 — 그 이후의, 지금 코드와 맞는 체크포인트는 아직 확인되지
     않음.
  이건 개인 역량이 아니라 팀/GPU 스케줄링에 달린 외부 의존성입니다 —
  도구 1~3은 real로 검증됐고, `predict_pathogenicity`의 real 경로
  (`model.py`, 실제 `LateFusionModel` 구조 기반)도 이미 구현되어 있어서
  맞는 체크포인트만 확보되면 바로 돌아갑니다.
- "전문가 검토 필요" 기준값(0.7)은 첫 추정치이며, 실제 precision/recall
  트레이드오프에 맞춰 튜닝된 값은 아닙니다.
- 아직 RAG(검색 증강) 요소가 없습니다(예: ACMG 변이 분류 가이드라인을
  근거로 함께 제시하는 것) — 자연스러운 다음 확장 지점입니다.
