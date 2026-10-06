# 네 policy의 공통 Stock Reward 과제 평가

**모든 row의 evaluation reward는 Stock Reward다.** Baseline의 matching-budget training을 추가하고, 기존 세 policy의 common-stock 결과는 재실행/수정하지 않은 채 통합했다. [상위 실험 README](../../README.md)가 전체 실험의 기준 문서다. [이전 3-policy 기록](../README.md)은 작성 당시의 결과와 Baseline 부재 상태를 보존한 역사적 snapshot이다.

## Primary comparison

Training은 네 조건 모두 4096 env × 32 steps × 1000 PPO updates = 131,072,000 transitions, seed 42 / terrain seed 42, fresh initialization이다. Evaluation은 **100 env, seed 24 / terrain seed 42**, deterministic mean action, 환경별 첫 episode, 최대 16 s / 960 steps / 60 Hz다. Terminal reward는 포함하고 post-reset reward는 제외한다. 각 policy는 자기 trained observation interface를 유지한다.

| 학습 조건 | Input | Stock Return | Displacement (m) | Duration (s) | Speed (m/s) | Fall | Timeout | ≥5 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline + Stock | 59-D | 51.2382 ± 26.5191 | 47.6936 ± 24.7009 | 12.4010 ± 5.8321 | 3.4099 ± 1.2702 | 36/100 | 64/100 | 85/100 |
| HeightScan + Stock | 122-D | 61.3354 ± 31.2305 | 60.5140 ± 29.1880 | 12.1947 ± 5.5930 | 4.4506 ± 1.5331 | 41/100 | 59/100 | 87/100 |
| HeightScan + Contact + Stock | 126-D | 63.5682 ± 28.5777 | 61.8529 ± 27.8017 | 12.9088 ± 5.3955 | 4.3692 ± 1.3281 | 29/100 | 71/100 | 89/100 |
| HeightScan + Contact + Modified-trained | 126-D | 60.9104 ± 30.3957 | 56.5627 ± 28.3639 | 12.1187 ± 5.6815 | 4.0718 ± 1.5654 | 43/100 | 57/100 | 86/100 |

연속형 값은 평균 ± population std (`ddof=0`)다. Fall/timeout/≥5 m의 분모는 100이므로 count와 백분율의 숫자가 같다. 원시 [comparison.csv](comparison.csv)와 [comparison.json](comparison.json)에 반올림하지 않은 값이 있다.

## 단계별 변화

다음은 오른쪽 policy − 왼쪽 policy의 평균 또는 count 차이다.

| 비교 | Δ Stock Return | Δ Displacement (m) | Δ Duration (s) | Δ Speed (m/s) | Δ Fall | Δ Timeout | Δ ≥5 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline → HeightScan | +10.0972 | +12.8205 | −0.2063 | +1.0407 | +5 | −5 | +2 |
| HeightScan → Contact + Stock | +2.2328 | +1.3389 | +0.7142 | −0.0814 | −12 | +12 | +2 |
| Contact + Stock → Modified-trained | −2.6578 | −5.2902 | −0.7902 | −0.2974 | +14 | −14 | −3 |

이 single-seed 결과에서 HeightScan은 더 높은 전진 성과와 함께 fall 증가를 보였다. Contact는 fall을 줄이고 duration을 늘렸지만 속도는 소폭 낮았다. Modified Reward 학습은 공통 Stock Reward objective에서 Stock-trained Contact policy를 능가하지 못했다. 네 단계를 단조로운 개선으로 표현하지 않는다.

## Control과 pairing 제한

[4-task config verification](../../baseline_4096x32x1000/evaluation_config_verification/reward_config_verification.json)에서 정확한 Stock reward 함수/parameter/weight 및 환경 control을 확인했다. 기존 task config에서 reward만 Stock `RewardsCfg()`로 지정하며 physics, action, termination, reset event, terrain, curriculum은 유지한다. 두 Contact config는 reward 교체 후 동일하다. HeightScan-only에서는 Contact sensor/4-D term만 없고, Baseline은 HeightScan sensor/63-D term도 없다. Actor/critic은 59/122/126/126-D이며 normalization과 observation feature 순서/scale을 바꾸지 않는다.

네 조건의 terrain mesh, row/column, root state, joint velocity, origin, material property는 일치한다. **Baseline의 초기 joint position sample은 다르므로 Stage 1을 full paired evaluation이라고 주장하지 않는다.** RayCaster의 zero-drift sampling도 RNG를 소비하는 구조이며, Baseline은 이 sensor가 없는 native reset을 사용한다. 평가용 sensor 추가, sample replay, RNG 보정이나 reset normalization은 하지 않았다. 같은 seed가 동일한 random sample을 보장하지 않는 이 한계를 명시한다.

나머지 세 policy는 full initial state가 같아 Stage 2·3 pairing을 유지한다. JSON의 Stage 1 delta는 단순 mean 차이이며 paired std/CI를 계산하지 않았다. Stage 2·3에는 기존 convention의 descriptive paired difference mean/std만 보존한다. Independent training seed 간 significance나 새로운 bootstrap method를 추가하지 않는다.

## 동일 Stock Reward decomposition

실제 RewardManager contribution인 `raw × weight × control_dt`를 같은 이름/정의로 집계한 평균 ± population std다.

| Component | Baseline | HeightScan | Contact + Stock | Contact + Modified-trained |
|---|---:|---:|---:|---:|
| progress | 47.6726 ± 24.7041 | 60.4711 ± 29.1633 | 61.8339 ± 27.8095 | 56.5487 ± 28.3737 |
| alive | 6.1975 ± 2.9193 | 6.0939 ± 2.7998 | 6.4520 ± 2.7012 | 6.0558 ± 2.8440 |
| upright | 1.1480 ± 0.5783 | 1.0924 ± 0.5508 | 1.1876 ± 0.5511 | 1.1440 ± 0.5749 |
| move_to_target | 5.8318 ± 3.2112 | 5.3633 ± 2.6815 | 5.8745 ± 2.6763 | 5.4832 ± 2.6863 |
| action_l2 | -0.1941 ± 0.3196 | -1.2979 ± 8.7617 | -0.5343 ± 1.3955 | -0.1891 ± 0.3820 |
| energy | -5.9736 ± 3.0826 | -7.2101 ± 3.5063 | -7.6240 ± 3.3849 | -6.4250 ± 3.2256 |
| joint_pos_limits | -3.4440 ± 1.6919 | -3.1773 ± 1.6496 | -3.6215 ± 1.6988 | -1.7071 ± 1.0048 |

Modified-trained policy는 일부 제어 penalty가 덜 음수였지만 progress도 낮아 total Stock Return은 낮았다. 기존 Modified objective return은 상위 실험 문서의 학습 objective diagnostic에만 보고하며 위 primary table과 직접 비교하지 않는다.

## Artifact와 실행

- [Baseline 실험 전체](../../baseline_4096x32x1000/README.md): training/config/selection/checkpoint/hash/provenance 및 후처리 오류 복구 기록.
- 네 evaluation 원본: [Baseline](../baseline/), [HeightScan](../heightscan_stock/), [Contact + Stock](../heightscan_contact_stock/), [Contact + Modified-trained](../heightscan_contact_modified_trained/). 각 디렉터리에 `summary.json`, `episodes.csv`, `manifest.json`, `initial_conditions.json`, `config.json`, `agent.json`이 있다.
- [Comparison](comparison.json), [CSV](comparison.csv), [검증](validation.json): 100/100 episode, 입력 차원, summary/decomposition 재계산, pairing 차이, budget/selection 검증.
- [Baseline 명령](../../baseline_4096x32x1000/commands.log)과 [기존 세 조건 명령](../commands.log)은 실제 실행 기록이다.

보존된 결과를 덮어쓰지 않고 재검증한다.

```bash
python scripts/finalize_baseline_ablation.py --check-only
python scripts/analyze_common_stock_reward.py --check-only
```

새 Baseline rollout 예시는 다음과 같다. [Root runtime 설정](../../../../README.md)을 먼저 적용하고 output은 존재하지 않는 경로로 지정한다. 선택 checkpoint는 evaluator 내부의 고정된 `best_model.pt`다.

```bash
python scripts/evaluate_common_stock_reward.py evaluate --headless \
  --policy baseline \
  --verification-record experiments/observation_ablation/baseline_4096x32x1000/evaluation_config_verification/reward_config_verification.json \
  --output experiments/observation_ablation/validation/baseline_common_stock_reproduction \
  --kit_args '--portable-root=/tmp/ant_baseline_kit --/app/settings/persistent=false'
```

Baseline evaluation의 selected checkpoint SHA256은 `673ce3492e979ad99c12f0becd7bf223bbdd8075cbbc54a33139a8f2faf65a1a`다. 나머지 세 checkpoint hash와 이전 결과 파일은 그대로다. 모든 policy에서 100개 첫 episode가 완료됐고 NaN/Inf, duplicate/missing env ID가 없었다. 최대 episode component residual은 2.5216e-6이다.

## 한계와 역사적 기록

Single training seed 42, evaluation seed 24, 하나의 terrain realization에 한정된다. Boundary termination이 없어 generated terrain X 범위 `[-102, 102]` m 밖 이동도 점수에 포함된다. Out-of-terrain-X는 Baseline부터 순서대로 20/100, 28/100, 27/100, 26/100이다. Gait 안정성이나 terrain generalization을 aggregate return만으로 주장하지 않는다.

Baseline training의 사후 metadata 오류와 복구, exact wall-time 부재는 [Baseline 기록](../../baseline_4096x32x1000/training_finalization_recovery.json)에 남겼다. 기존 runtime shutdown caveat도 유지한다. 기존 3-policy README/CSV/JSON/hash manifest는 당시 snapshot이며 수정하지 않았다. 확장한 evaluator의 이전 source도 Baseline `source_snapshots/`에 보존해 역사적 source hash와 구분한다. 새 최종 보호 검증은 [Baseline final integrity](../../baseline_4096x32x1000/final_integrity.json)를 기준으로 확인한다.
