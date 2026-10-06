# Matching-budget Baseline + Stock

Stage 1의 프로젝트 native-observation Baseline이다. Task는 `Ant-rl-Ablation-Baseline-v0`, config는 `ant.ablation_env_cfg:AblationBaseCfg`, runner는 `AblationBaselinePPORunnerCfg`다. Framework `Isaac-Ant-v0` 60-D task를 사용하지 않는다.

## 학습 조건과 선택

| 항목 | 값 |
|---|---|
| Observation / action | 59-D / 8-D |
| Training reward | 기존 Stock `RewardsCfg`의 7개 term |
| Training / terrain seed | 42 / 42 |
| Budget | 4096 env × 32 steps × 1000 updates = 131,072,000 transitions |
| Actor/critic hidden layers | [400, 200, 100], ELU |
| Empirical normalization / resume | 없음 / fresh initialization |
| Save interval | 50 updates; 마지막 iteration 999도 보존 |
| Terrain assignment / curriculum | 초기 배정 유지 / 없음 |
| Physics / control / horizon | 120 Hz / 60 Hz / 16 s |
| Selected iteration / training mean return | 875 / 42.83538151267916 |
| Runner reported elapsed | 471 s; rounded display |

기존 세 실험의 저장 env/agent config와 현재 config를 비교했다. Terrain, robot, simulation, action, reset event, termination, episode horizon, PPO와 budget은 같으며, 추가 observation sensor/term과 Modified training reward만 다르다. Native 24-D incoming foot wrench는 유지하고 HeightScan과 binary Contact observation은 없다. First-layer input 크기가 59이므로 다른 입력 차원의 policy와 parameter 수까지 같지는 않다.

선택 기준은 기존 run과 같은 **최근 100개 completed episode의 최고 logged training mean return**이다. [선택 근거](checkpoint_selection.json)의 iteration/info는 1000-row training curve의 최고값과 일치한다. Evaluation 성과는 선택에 사용하지 않았다.

| Checkpoint | SHA256 |
|---|---|
| [best_model.pt](checkpoints/best_model.pt) | `673ce3492e979ad99c12f0becd7bf223bbdd8075cbbc54a33139a8f2faf65a1a` |
| [final_model.pt](checkpoints/final_model.pt), iteration 999 | `c232bea24cd7cf335f8e3bb76848a46cc8fc14b7747a7cbe9e74f941e47ce616` |

## 공통 Stock Reward 평가

Evaluation seed 24, terrain seed 42, 100 env, deterministic mean action, 첫 episode, 16 s / 960 control steps를 사용했다. Native termination과 reset 분포를 유지하고 terminal reward를 포함하며 post-reset reward는 제외한다. Policy input은 59-D 그대로다.

| 지표 | 결과 |
|---|---:|
| Stock Return | 51.2382 ± 26.5191 |
| Displacement | 47.6936 ± 24.7009 m |
| Duration | 12.4010 ± 5.8321 s |
| Mean vx | 3.4099 ± 1.2702 m/s |
| Fall / timeout | 36/100 (36%) / 64/100 (64%) |
| ≥5 m | 85/100 (85%) |
| Out-of-terrain-X | 20/100 (20%) |

±는 평가 환경 100개에 대한 population std다. [Summary](evaluation_summary.json)와 [100-episode CSV](episode_metrics.csv)는 [공통 평가 원본](../common_stock_reward_evaluation/baseline/)의 동일한 파일을 byte-identical하게 복사한 것이다. Rollout은 한 번만 수행했다. Component residual 최대값은 step 2.0664e-8, episode 2.5216e-6이다.

Baseline과 기존 세 policy는 terrain mesh/배정, root state, joint velocity, friction이 같다. **초기 joint position은 다르므로 Baseline 포함 네 조건의 full pairing을 주장하지 않는다.** RayCaster reset의 zero-drift sampling도 RNG를 소비하지만 Baseline에는 해당 sensor가 없다. Reset 분포를 바꾸거나 sensor를 추가해 sample을 맞추지 않았다. Stage 2·3의 기존 pairing은 유지한다.

HeightScan − Baseline의 평균 차이는 return +10.0972, displacement +12.8205 m, duration −0.2063 s, 속도 +1.0407 m/s다. Fall +5, timeout −5, ≥5 m +2이다. Single-seed에서 전진 성과는 높지만 fall도 증가하므로 일관된 robustness 개선으로 해석하지 않는다. [전체 네 policy 비교](../common_stock_reward_evaluation/four_policy_comparison/README.md)를 함께 확인한다.

## 실행과 provenance

실제 실행한 명령과 환경 변수는 [commands.log](commands.log)에 있다. 기본 단계는 다음과 같으며, 이미 존재하는 결과나 run을 덮어쓰지 않는다.

```bash
python scripts/run_baseline_ablation.py verify
python scripts/run_baseline_ablation.py smoke
python scripts/run_baseline_ablation.py train
python scripts/run_baseline_ablation.py verify_eval
python scripts/run_baseline_ablation.py smoke_eval
python scripts/run_baseline_ablation.py evaluate
python scripts/finalize_baseline_ablation.py
```

Runtime은 Python 3.11.16, Isaac Lab 2.3.0, Isaac Sim 5.1.0.0, PyTorch 2.7.0+cu128, CUDA 12.8, RTX 5070 Ti다. Git HEAD는 `09ccc66d44b1304147a37c18fa07223210964abe`였으며, **uncommitted common-stock work가 있는 dirty tree**에서 실행했다. [Manifest](manifest.json), [학습 직전 상태](working_tree_before.txt), [기존 diff](working_tree_before.diff), [source hash](runtime_source_mapping.json), [실제 학습 source snapshot](source_snapshots/baseline_trainer_used_for_training.py)을 보존한다. Source mapping은 기록 시점의 hash이며, 후처리 경로 수정과 최신 CLI는 별도 최종 검증에서 구분한다.

다른 새 artifact/run에서 동일 예산 학습을 재현하려면 [root runtime 설정](../../../README.md)을 적용하고 전용 trainer의 `--artifact-root`와 `--output`을 새 경로로 지정한다. 아래 값은 기존 결과를 덮어쓰지 않는 재현 예시다.

```bash
BASELINE_REPLAY="experiments/observation_ablation/validation/baseline_reproduction"
python scripts/baseline_observation_ablation_budget.py verify --headless \
  --artifact-root "$BASELINE_REPLAY" --output "$BASELINE_REPLAY"
python scripts/baseline_observation_ablation_budget.py smoke --headless \
  --artifact-root "$BASELINE_REPLAY" --output "$BASELINE_REPLAY/smoke"
python scripts/baseline_observation_ablation_budget.py train --headless \
  --artifact-root "$BASELINE_REPLAY" \
  --output logs/rsl_rl/observation_ablation/baseline_reproduction_s42_e4096_n32_i1000
```

## 후처리 오류와 복구

Full training은 1000 updates와 131,072,000 transitions를 끝냈고 iteration 999 및 selected checkpoint도 저장했다. 이후 `Path.relative_to(absolute ROOT)`에 상대 output 경로를 전달해 metadata 저장이 실패하면서 process exit code는 1이었다. [원본 stdout](train_stdout.log)을 보존한다.

[finalize_baseline_training.py](../../../scripts/finalize_baseline_training.py)는 saved checkpoint, complete training CSV, 마지막 logged transition 수를 검증하고 누락된 selection/summary만 작성했다. **재학습·checkpoint 수정·evaluation 기반 선택은 없다.** [복구 기록](training_finalization_recovery.json)에 원본 및 경로 수정 source hash를 구분한다. Exact perf-counter wall time은 복구하지 않았고 runner의 rounded elapsed만 보고한다. TensorBoard `Train/mean_reward`는 iteration 0–999의 **1000개 point**가 보존됐음을 [검증](tensorboard_verification.json)했다. 이 metadata 오류와 기존 runtime shutdown caveat는 성능 결과와 별도로 남긴다.

## Artifact 목록

- [Pretraining verification](pretraining_verification.json), [initialization smoke](smoke/smoke.json), [4-task 평가 config 검증](evaluation_config_verification/reward_config_verification.json).
- [Training summary](training_summary.json), [selection](checkpoint_selection.json), [config](config.json), [agent](agent.json), [protocol](protocol.json), [manifest](manifest.json).
- [전체 training run](../../../logs/rsl_rl/observation_ablation/ablation_baseline_stock_s42_e4096_n32_i1000/): CSV, TensorBoard event, resolved configs, periodic checkpoints.
- [Evaluation summary](evaluation_summary.json), [episodes](episode_metrics.csv), [공통 평가 manifest/초기 상태](../common_stock_reward_evaluation/baseline/).
- [평가 전 보호 hash](integrity_before_baseline.json), [최종 integrity](final_integrity.json): 기존 artifact 보존과 의도적인 문서/script/protocol 변경을 구분한다. Historical manifest는 수정하지 않는다.
