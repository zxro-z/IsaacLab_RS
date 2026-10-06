# 공통 Stock Reward 기준 과제 평가

세 policy의 **학습 objective**와 **과제 평가 objective**를 구분하기 위한 별도 평가다. 기존 canonical summary/CSV는 각 학습 reward로 계산한 진단으로 보존하며, 아래 평가는 모두 동일한 Stock Reward를 사용한다. 재학습과 checkpoint 재선택은 없다.

## 비교 조건과 결과

| 학습 조건 | Observation | 평가 Reward | Stock Return | Displacement (m) | Duration (s) | Mean vx (m/s) | Fall | Timeout | ≥5 m |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| HeightScan + Stock | 122-D | Stock | 61.3354 ± 31.2305 | 60.5140 ± 29.1880 | 12.1947 ± 5.5930 | 4.4506 ± 1.5331 | 41/100 | 59/100 | 87/100 |
| HeightScan + Contact + Stock | 126-D | Stock | 63.5682 ± 28.5777 | 61.8529 ± 27.8017 | 12.9088 ± 5.3955 | 4.3692 ± 1.3281 | 29/100 | 71/100 | 89/100 |
| HeightScan + Contact + Modified | 126-D | Stock | 60.9104 ± 30.3957 | 56.5627 ± 28.3639 | 12.1187 ± 5.6815 | 4.0718 ± 1.5654 | 43/100 | 57/100 | 86/100 |

연속형 지표는 100개 환경의 첫 episode에 대한 평균 ± population std (`ddof=0`)다. Count는 분모가 100이므로 fall/timeout/≥5 m 비율은 각각 표의 count와 같은 백분율이다. 추가 ≥2/≥10 m 및 맵 경계 진단은 각 `summary.json`에 있다.

Contact + Stock − HeightScan + Stock의 평균 return 차이는 +2.2328이다. Fall은 12개 적고 duration과 displacement는 높지만 평균 속도는 낮다. Modified-trained − Stock-trained Contact policy의 평균 return 차이는 −2.6578이다. Modified-trained policy는 공통 Stock Reward objective에서 Stock-trained Contact policy를 능가하지 못했으며, fall은 14개 많고 displacement·duration·속도는 낮다. 이는 하나의 training seed에서 관찰한 결과다.

## 정확한 공통 reward와 환경 control

[전용 evaluator](../../../scripts/evaluate_common_stock_reward.py)는 각 **기존 task config**를 생성하고 `.rewards`만 `isaaclab_tasks.manager_based.classic.ant.ant_env_cfg.RewardsCfg()`로 교체한다. Stock config 전체를 다른 framework 환경으로 바꾸지 않는다. Reward 교체 전후의 모든 non-reward 설정이 동일한지 assert하며, 실제 RewardManager의 함수 identity·parameter·weight도 해당 Stock 클래스와 대조한다.

| Term | Weight | Parameter |
|---|---:|---|
| progress | 1.0 | `target_pos=(1000.0, 0.0, 0.0)` |
| alive | 0.5 | 없음 |
| upright | 0.1 | `threshold=0.93` |
| move_to_target | 0.5 | `threshold=0.8`, 동일 target |
| action_l2 | -0.005 | 없음 |
| energy | -0.05 | `gear_ratio={".*": 15.0}` |
| joint_pos_limits | -0.1 | `threshold=0.99`, 동일 gear ratio |

함수 경로와 전체 parameter는 [config verification](reward_config_verification.json) 및 각 `manifest.json`의 `runtime_reward_terms`에 보존한다. `alive`는 실제 terminated episode의 terminal step에서 제외되며, timeout은 fall과 구분된다. 모든 term은 실제 manager 연산을 사용한다.

| 평가 설정 | 공통 값 |
|---|---|
| Evaluation seed / terrain seed | 24 / 42 |
| 환경 수 / 최대 길이 | 100 / 16 s, 960 control steps |
| Physics / control rate | 120 Hz / 60 Hz |
| 추론 | Deterministic mean action |
| Episode 집계 | 환경별 첫 episode만, terminal reward 포함, post-reset reward 제외 |
| Termination | `body_z_down(pi/2)` 또는 timeout; height/boundary termination 없음 |
| Reset / terrain / curriculum | Native reset 유지, 초기 terrain assignment 유지, curriculum 없음 |

세 config는 동일한 termination, reset event, terrain, action, robot dynamics, sim 설정을 사용한다. 두 Contact config는 reward 교체 후 완전히 동일하다. HeightScan-only에서는 추가 observation용 `feet_contacts` sensor와 마지막 4-D binary term만 없다. HeightScan ray pattern, clipping, scale, observation 순서 및 normalization은 기존 그대로다. Seed와 환경 수는 canonical evaluation 값으로 지정했고, 그 밖의 환경 설정을 정규화하지 않았다.

Reward 정의 자체는 termination이나 physical reset을 바꾸지 않는다. Modified `TotalReward`와 Stock `progress_reward`는 자기 potential을 reset하지만 환경의 pose·joint reset을 변경하지 않는다. Canonical evaluator 세 개는 각각 native reward를 사용한다. 일반 `play.py`의 reward override는 이번 평가에 사용하지 않는다.

## Episode 집계와 pairing

기존 evaluator와 동일하게 reset 직전 world-X와 world-X velocity를 읽는 `_reset_idx` hook을 사용하며, original reset을 그대로 호출한다. `~done` mask는 해당 step 이전의 상태를 기준으로 하므로 terminal reward를 포함하고 이미 첫 episode를 끝낸 환경의 후속 reward는 누적하지 않는다. Component는 `RewardManager._step_reward × step_dt`로 누적한다. Step별 합과 return residual은 최대 3.8709e-8, episode residual은 최대 2.3498e-6이다.

환경 ID 0–99의 초기 full root state, joint position/velocity, origin, terrain row/column, material property와 terrain mesh hash가 세 조건에서 모두 일치했다. Mesh hash는 각 기존 canonical evaluation의 `results/manifest.json`과도 일치한다. [initial_conditions.json](heightscan_stock/initial_conditions.json)과 [comparison.json](comparison.json)에 pairing 근거가 있다. 두 Stock policy의 return과 세 policy의 episode별 보행 지표는 기존 canonical CSV와 동일하다. 따라서 Modified policy의 reward 교체가 이번 rollout의 보행 궤적을 바꾸지 않았음을 결과에서도 확인했다.

[comparison.json](comparison.json)의 paired difference는 기술 통계다. 100개 평가 환경 간 변동을 independent training seed 간 변동으로 해석하지 않으며, significance나 새로운 bootstrap CI는 주장하지 않는다.

## Stock Reward decomposition

각 값은 동일한 Stock Reward의 실제 weighted contribution 평균 ± population std다.

| Component | HeightScan + Stock | Contact + Stock | Contact + Modified-trained |
|---|---:|---:|---:|
| progress | 60.4711 ± 29.1633 | 61.8339 ± 27.8095 | 56.5487 ± 28.3737 |
| alive | 6.0939 ± 2.7998 | 6.4520 ± 2.7012 | 6.0558 ± 2.8440 |
| upright | 1.0924 ± 0.5508 | 1.1876 ± 0.5511 | 1.1440 ± 0.5749 |
| move_to_target | 5.3633 ± 2.6815 | 5.8745 ± 2.6763 | 5.4832 ± 2.6863 |
| action_l2 | -1.2979 ± 8.7617 | -0.5343 ± 1.3955 | -0.1891 ± 0.3820 |
| energy | -7.2101 ± 3.5063 | -7.6240 ± 3.3849 | -6.4250 ± 3.2256 |
| joint_pos_limits | -3.1773 ± 1.6496 | -3.6215 ± 1.6988 | -1.7071 ± 1.0048 |

Modified-trained policy는 energy·action·joint-limit penalty가 덜 음수지만 progress contribution이 낮다. 제어 penalty 감소가 낮은 전진 성과를 상쇄하지 못했다. 기존 Modified return 134.8413 ± 68.0187은 다른 objective의 값이므로 위 Stock Return과 직접 비교하지 않는다. Aggregate 지표만으로 hopping 감소나 더 안정적인 gait를 주장하지 않는다.

## Checkpoint와 artifact

Selected checkpoint는 [상위 실험 기록](../README.md)의 `best_model.pt` 선택을 그대로 따른다.

| 학습 조건 | Checkpoint | SHA256 |
|---|---|---|
| HeightScan + Stock | [best_model.pt](../heightscan_4096x32x1000/checkpoints/best_model.pt) | `49a8a007f152ed9c63df616fbee12a866873be0ee0337ad83597d653ec9b3c06` |
| Contact + Stock | [best_model.pt](../heightscan_contact_stock_4096x32x1000/checkpoints/best_model.pt) | `82c7d0f781c37e0620230ff4b401835d54ef0a18102cdbc8c22c28bb576f57b9` |
| Contact + Modified | [best_model.pt](../heightscan_contact_modified_4096x32x1000/checkpoints/best_model.pt) | `0471caaadd646feac85cb7bbed0d2c3668279d2d65db9507030a649895ea11b4` |

- [heightscan_stock/](heightscan_stock/), [heightscan_contact_stock/](heightscan_contact_stock/), [heightscan_contact_modified_trained/](heightscan_contact_modified_trained/): `summary.json`, `episodes.csv`, `config.json`, `agent.json`, `manifest.json`, `initial_conditions.json`.
- [comparison.csv](comparison.csv), [comparison.json](comparison.json): 비교 표와 환경 ID별 paired difference 요약.
- [smoke/](smoke/): 세 checkpoint 각각 4-env, 최대 32-step 실제 actor smoke test; 실험 결과로 사용하지 않는다.
- [commands.log](commands.log): 환경 변수와 실행한 실제 CLI 명령. 별도 config/smoke/evaluate log도 보존한다.
- [integrity_before.json](integrity_before.json), [integrity_validation.json](integrity_validation.json), [final_integrity.json](final_integrity.json): 평가 전 snapshot, 결과 검증, 문서 추가 후 보존 상태.

## 실행 및 재검증

Runtime은 Python 3.11.16, Isaac Lab 2.3.0, Isaac Sim 5.1.0.0, PyTorch 2.7.0+cu128, CUDA 12.8이다. GPU는 RTX 5070 Ti를 사용했다. 각 manifest에 UTC timestamp, 당시 HEAD, checkpoint/config/source hash가 있다. 당시 HEAD는 `09ccc66d44b1304147a37c18fa07223210964abe`이며, 신규 evaluator는 아직 commit하지 않은 source로 실행했으므로 source hash를 함께 기준으로 삼는다.

실행 순서는 config verification → 세 loaded-policy smoke test → 세 100-env evaluation → CSV 재계산과 integrity 검사다. `run_common_stock_reward_evaluation.py`는 세 조건을 순차 실행하며 결과 디렉터리나 log가 이미 있으면 중단한다. Runtime cache는 `/tmp`로 지정했다. Config 검사는 sandbox에서 통과했지만 GPU 접근이 제한되어 smoke와 full rollout은 GPU 접근이 가능한 실행으로 수행했다.

기존 결과를 쓰지 않고 확인만 하려면 repository root에서 실행한다.

```bash
python scripts/analyze_common_stock_reward.py --check-only
```

새 rollout은 [root README의 runtime 설정](../../../README.md)을 적용하고, 보존된 verification/smoke gate를 사용하는 evaluator에 **새 output 경로**를 지정한다. 예를 들어 Modified-trained checkpoint를 다시 공통 Stock Reward로 평가하려면 아래 명령을 사용한다. Checkpoint는 evaluator 내부에서 고정되며 CLI로 바꾸지 않는다.

```bash
python -u scripts/evaluate_common_stock_reward.py evaluate --headless \
  --policy heightscan_contact_modified_trained \
  --output experiments/observation_ablation/validation/common_stock_reproduction_modified \
  --kit_args '--portable-root=/tmp/ant_common_stock_kit --/app/settings/persistent=false'
```

나머지 조건은 `--policy heightscan_stock` 또는 `--policy heightscan_contact_stock`을 사용하고 output을 각각 새 경로로 지정한다. 모든 실행은 고정된 세 checkpoint, reward/config verification 및 세 smoke PASS 기록을 요구한다. 새 결과를 기존 artifact 위에 저장하지 않는다.

## 검증과 한계

각 조건에서 환경 ID 중복/누락 없이 첫 episode 100개를 완료했다. Reward, action, processed observation, CSV metric에 NaN/Inf가 없고, per-episode component sum과 return이 일치한다. Existing artifact 521개의 hash는 문서 추가 전에 모두 일치했다. 문서 추가 이후에는 실험 README의 의도적인 section 추가만 별도로 기록하고 나머지 520개는 그대로 유지한다. Historical integrity manifest는 변경하지 않았다.

Single training seed 42와 evaluation seed 24, 하나의 terrain realization에 대한 결과다. Matching-budget baseline은 없다. Map-boundary termination이 없어 X 범위 `[-102, 102]` m 밖의 이동도 progress와 displacement에 포함된다. Out-of-terrain-X는 조건 순서대로 28/100, 27/100, 26/100이며, 거리 전체를 rough terrain 위의 보행 성과로 해석하지 않는다. Training objective가 다른 policy의 우열은 여기서 사용한 공통 Stock Reward와 동반 behavioral metric의 범위에서만 해석한다.
