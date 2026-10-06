# HeightScan 기반 지형 인식 Ant 보행

로봇 시뮬레이션 과제 1 제출 자료. 본 프로젝트는 불규칙 지형에서 Ant의 보행 성능을 개선하기 위해 **HeightScan을 통한 지형 인식**, **접촉 상태 관측**, **reward shaping**을 단계적으로 적용하고 그 영향을 분석한다. 학습에는 Isaac Lab과 PPO를 사용한다.

**Baseline → HeightScan → HeightScan + Contact → HeightScan + Contact + Modified Reward**

각 요소의 영향을 staged ablation으로 비교한다. 기록된 single-seed 평가에서는 Contact 추가 후 fall 수가 감소했지만, Modified Reward는 보고된 보행 안정성 지표를 개선하지 못했다. 동일 학습 예산의 정량적 baseline 결과는 확보되어 있지 않다.

## 프로젝트 개요

Ant는 여덟 개의 joint-effort action으로 계단, 역계단, 박스, 경사면, 역경사면을 전진한다. 각 지형의 생성 비율은 20%이며, 맵은 10 × 10 m 크기의 패치 20 × 10개로 구성된다. 물리 시뮬레이션은 120 Hz, 제어는 60 Hz로 동작하며 episode 길이는 최대 16초다.

HeightScan은 기존 로봇 observation에 주변 지형의 높이 정보를 추가한다. 이를 바탕으로 지형 형상 정보와 발의 접촉 상태 정보가 proprioception만 사용하는 구성에 비해 어떤 영향을 주는지 살펴본다. 단계적 개선은 개발 목표이며, 모든 단계에서 실제 성능이 향상되었다는 의미는 아니다.

## 연구 질문

주변 지형의 높이 정보, 명시적인 발 접촉 상태, reward 설계는 불규칙 지형에서 전진 보행과 안정성에 어떤 영향을 주는가?

평가는 displacement, episode duration, 전진 속도, fall, timeout, 목표 거리 도달 비율, reward decomposition을 함께 사용한다. 특히 reward objective가 달라지는 비교에서는 return만으로 성능을 판단하지 않는다.

## 실험 설계

| 구성 | HeightScan | Contact observation | Reward |
|---|---|---|---|
| Baseline | 없음 | 없음 | Stock |
| HeightScan | 사용 | 없음 | Stock |
| HeightScan + Contact | 사용 | 사용 | Stock |
| 최종 구성: HeightScan + Contact + Modified Reward | 사용 | 사용 | Modified |

Baseline은 기존 24-D incoming foot wrench를 포함한 프로젝트의 **59-D native observation**과 Stock Reward를 사용하는 시작 구성이다. Contact observation은 이 wrench를 유지하면서 별도의 4-D binary 접촉 신호를 추가한다. 다만 **동일 학습 예산의 정량적 baseline run과 checkpoint는 없다**. Framework stock Ant의 60-D interface도 동일 조건의 baseline으로 대체하지 않는다.

보존된 세 policy는 **4096개 환경 × 32 rollout steps × 1000 PPO updates = 131,072,000 transitions**, training seed 42, terrain seed 42를 사용하며, 이전 checkpoint를 이어서 학습하지 않고 새로 초기화한다. 학습 중 terrain patch는 최초 배정을 유지하고 curriculum은 사용하지 않는다. Stage 2와 Stage 3은 환경·PPO·학습 예산의 동등성을 기록으로 확인했으며, actor/critic은 [400, 200, 100] ELU network를 사용한다. 비교 결과는 single-seed 실험 범위에 한정된다.

## Ablation 단계

### Stage 1 — 지형 인식

**명시적인 local terrain height 정보가 불규칙 지형 보행을 개선하는가?** Stock observation과 Stock Reward를 사용하는 baseline에 HeightScan을 추가하는 비교를 설계했다. 동일 조건의 baseline 결과가 없으므로, 기록된 HeightScan 성능을 baseline 대비 개선량으로 해석할 수 없다.

[HeightScan 설정](source/ant/ablation_env_cfg.py)은 torso에 부착된 yaw-aligned RayCaster를 사용한다. **9 × 7개의 하향 ray**를 0.2 m 간격으로 배치하며 offset은 `(0.8, 0, 20)`이다. 63개 높이 값은 `sensor_z - hit_z - 0.5`로 계산하고, scale 1과 `[-1, 1]` clipping을 적용한다. 이를 기존 59개 feature 뒤에 concatenate하여 actor/critic에 **122-D** observation을 입력한다. Empirical observation normalization은 사용하지 않는다.

### Stage 2 — 접촉 상태 관측

**접촉 상태 정보가 지형 형상 정보만 사용할 때보다 보행 안정성을 높이는가?** HeightScan과 Stock Reward를 유지하고 네 발의 binary Contact observation만 추가한다. Input dimension은 **122-D에서 126-D**로 바뀐다.

[Contact 명세](experiments/observation_ablation/shared/contact_observation.json)는 현재 sample의 **world-frame net-force norm > 1 N** 여부를 사용한다. History length는 0, encoding은 float32 binary, scale은 1이며 clipping과 normalization은 없다. 발의 순서는 고정된다. 이 단계에서는 Contact를 observation에만 추가하고 reward-side contact term은 활성화하지 않는다.

### Stage 3 — Reward 설계

**지형과 접촉 정보가 주어진 상태에서 reward shaping이 안정적인 보행을 추가로 개선하는가?** **126-D HeightScan + Contact observation**과 학습 조건을 유지하고, Stock Reward를 [Modified Reward](source/ant/rewards.py)로 교체한다.

설계 목적은 반복적인 jumping/hopping을 완화하고, 지면 접촉, foot slip, 관절 운동, 제어 부담를 함께 고려하는 것이다. 기존 weight를 변경하고 contact, joint-velocity, slip term을 추가한다. 개별 term의 효과를 분리하는 실험은 아니며 reward objective 전체의 변경을 비교한다. Reward-side contact는 observation과 다른 정의를 사용한다. 세 프레임의 최대 vertical force > 5 N으로 접촉을 판정하고, 최소 두 발이 접촉할 때 bonus를 부여한다. 집계 결과만으로 hopping 감소나 더 안정적인 gait를 확인할 수는 없다.

## 공통 Stock Reward 기준 과제 평가

**세 policy 모두 동일한 Stock Reward로 평가했다.** HeightScan + Stock은 학습 당시의 **122-D**, 두 Contact policy는 **126-D** observation을 그대로 받는다. Modified-trained policy도 학습에는 Modified Reward를 사용했지만, 여기서는 Stock Reward로 점수를 계산한다. 아래 표가 과제의 공통 objective에 따른 직접적인 policy 점수 비교다.

Evaluation seed 24, terrain seed 42, **100개 환경**, deterministic mean action, 각 환경의 첫 episode, 최대 16 s / 960 control steps를 사용했다. Termination과 reset은 기존 설정을 유지하며 terminal reward를 포함하고 post-reset reward를 제외한다. Terrain mesh와 환경별 초기 root/joint 상태, friction, terrain 배정이 일치하는 paired evaluation이다. 각 조건은 100/100 episode를 완료했다. ±는 100개 평가 환경에 대한 population standard deviation이며 training seed 간 변동이 아니다.

| 학습 조건 | 평가 Reward | Stock Return | Displacement (m) | Duration (s) | 평균 전진 속도 (m/s) | Fall | Timeout | ≥5 m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HeightScan + Stock | Stock | 61.3354 ± 31.2305 | 60.5140 ± 29.1880 | 12.1947 ± 5.5930 | 4.4506 ± 1.5331 | 41/100 | 59/100 | 87/100 |
| HeightScan + Contact + Stock | Stock | 63.5682 ± 28.5777 | 61.8529 ± 27.8017 | 12.9088 ± 5.3955 | 4.3692 ± 1.3281 | 29/100 | 71/100 | 89/100 |
| HeightScan + Contact + Modified | Stock | 60.9104 ± 30.3957 | 56.5627 ± 28.3639 | 12.1187 ± 5.6815 | 4.0718 ± 1.5654 | 43/100 | 57/100 | 86/100 |

Contact 추가 후 평균 Stock Return은 **2.2328** 높고 fall은 41→29로 감소했다. 동일한 126-D interface를 사용하는 Modified-trained policy는 Contact + Stock-trained policy보다 평균 Stock Return이 **2.6578 낮으며**, displacement·duration·속도가 낮고 fall은 29→43으로 증가했다. 따라서 이 single-seed 평가에서 Modified Reward 학습은 공통 Stock Reward 과제 objective의 성능을 개선하지 못했다. 두 Stock-trained policy의 return과 세 policy의 episode별 보행 지표는 기존 canonical 기록과 동일하다.

이 평가는 새 checkpoint 선택이나 재학습 없이 수행한 별도 평가다. 기존 Modified objective의 **134.8413** return과 새 Stock Return을 직접 비교하지 않는다. Baseline 결과 부재, single-seed 범위, 맵 경계의 영향은 그대로 적용된다. [평가 기록과 재현 방법](experiments/observation_ablation/common_stock_reward_evaluation/README.md), [비교 CSV](experiments/observation_ablation/common_stock_reward_evaluation/comparison.csv), [검증 기록](experiments/observation_ablation/common_stock_reward_evaluation/integrity_validation.json)을 함께 제공한다.

## 주요 결과 — 학습 objective 기준 진단

아래 표는 보존된 세 policy의 canonical evaluation 결과다. **Seed 24, 100개 환경, deterministic mean action, 각 환경의 첫 episode만 사용**했으며, 조건별 100개 episode가 모두 완료되었다. 평가 길이는 최대 16 s / 960 control steps이고, terminal reward는 포함하되 reset 이후 reward는 제외한다. 각 policy는 학습 당시의 reward로 평가한다. 연속형 지표는 평균이며, ±는 population standard deviation을 나타낸다.

| 지표 | HeightScan + Stock | HeightScan + Contact + Stock | HeightScan + Contact + Modified |
|---|---:|---:|---:|
| Return — 각 학습 objective 기준 | 61.3354 ± 31.2305 | 63.5682 ± 28.5777 | 134.8413 ± 68.0187 |
| 전진 displacement (m) | 60.5140 ± 29.1880 | 61.8529 ± 27.8017 | 56.5627 ± 28.3639 |
| Episode duration (s) | 12.1947 ± 5.5930 | 12.9088 ± 5.3955 | 12.1187 ± 5.6815 |
| 평균 전진 속도 (m/s) | 4.4506 | 4.3692 | 4.0718 |
| Fall | 41/100 | 29/100 | 43/100 |
| Timeout | 59/100 | 71/100 | 57/100 |
| ≥5 m 도달 episode | 87/100 | 89/100 | 86/100 |

**Stock Reward와 Modified Reward는 정의와 scale이 다르므로, return 차이를 성능 개선량으로 해석하지 않는다.** Displacement에는 지형 맵 밖으로 이동한 거리도 포함될 수 있다. 전체 지표, reward component별 표, 경계 진단 결과는 [상세 실험 기록](experiments/observation_ablation/README.md)에 정리되어 있다.

## 결과 해석

Stock Reward를 유지하고 Contact를 추가한 조건에서는 fall이 감소하고(41 → 29), timeout이 증가했으며(59 → 71), 평균 displacement는 소폭 증가했다. 평균 전진 속도는 다소 낮아졌다. 이 결과는 접촉 상태 정보의 효과를 추가로 검토할 근거가 되지만, 통계적 유의성이나 여러 seed에서의 일관된 개선을 입증하지는 않는다.

Observation interface를 126-D로 유지한 Modified Reward 조건에서는 Contact + Stock보다 fall이 증가했고(29 → 43), displacement, episode duration, 전진 속도가 모두 낮았다. 더 큰 return은 서로 다른 objective에 따른 값이다. 최종 구성은 설계 순서상 마지막 단계이며 최고 성능의 policy를 뜻하지 않는다. HeightScan의 baseline 대비 추가 효과를 입증하는 정량적 비교는 없다.

## Task 및 Observation 구성

Task ID는 [source/ant/__init__.py](source/ant/__init__.py)에 등록되어 있다.

| Task | 설정 | Actor/critic input |
|---|---|---:|
| `Ant-rl-Ablation-Baseline-v0` | [AblationBaseCfg](source/ant/ablation_env_cfg.py) | 59-D; 기준 설정, 동일 조건 결과 없음 |
| `Ant-rl-Ablation-HeightScan-v0` | [AblationHeightScanCfg](source/ant/ablation_env_cfg.py) | 59 + 63 = **122-D** |
| `Ant-rl-Ablation-HeightScan-Contact-Stock-v0` | [HeightScanContactStockCfg](source/ant/contact_stock_env_cfg.py) | 59 + 63 + 4 = **126-D** |
| `Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0` | [Stage2HeightScanContactCfg](source/ant/stage2_env_cfg.py) | 59 + 63 + 4 = **126-D** |

두 Contact policy는 동일한 126-D observation 구성을 사용한다. 따라서 Stage 3에서는 policy input dimension이 아니라 reward 설계를 변경한다. `stage2_*` 구현 이름은 이전 명명 방식을 유지한 것으로, 이 문서에서는 Modified Reward 결과를 Stage 3에 배치한다. 기존 custom-reward task인 `Ant-rl-v0`도 남아 있다. Baseline의 공통 환경 설정은 기본 2048개 환경이므로 canonical 학습 예산에 맞추려면 `--num_envs 4096`을 지정해야 한다.

## Repository 구조

```text
.
├── source/ant/                         # Project configs, observations, rewards, PPO, task registration
├── source/isaaclab/                    # Isaac Lab framework core
├── source/isaaclab_tasks/              # Stock tasks and shared Ant runtime helpers
├── experiments/observation_ablation/   # Protocols, results, selected/final checkpoints, validation
├── scripts/                           # Dedicated experiment drivers and training/play entry points
├── logs/rsl_rl/observation_ablation/    # Canonical training logs and saved configurations
├── validation_compare/                # Supplementary terrain-transfer evaluation and video evidence
└── docs/submission_provenance/         # Source provenance, integrity records, verification reports
```

구현은 [프로젝트 소스](source/ant/), 결과 자료는 [실험 artifact](experiments/observation_ablation/), 학습 과정은 [training log](logs/rsl_rl/observation_ablation/)에서 확인할 수 있다. [지형 전이 평가](validation_compare/environment_provenance.md)는 별도의 host와 protocol에서 얻은 보충 자료이며, 해당 수치를 위 canonical 결과 표에 섞지 않는다.

## 재현 및 평가 방법

실행 전에 보존된 결과를 먼저 확인한다. 각 실험 디렉터리에는 evaluation summary, episode CSV, source mapping과 `checkpoints/best_model.pt`, `final_model.pt`가 있다.

- [HeightScan + Stock 자료](experiments/observation_ablation/heightscan_4096x32x1000/)
- [HeightScan + Contact + Stock 자료](experiments/observation_ablation/heightscan_contact_stock_4096x32x1000/)
- [HeightScan + Contact + Modified 자료](experiments/observation_ablation/heightscan_contact_modified_4096x32x1000/)

평가 재현에 사용한 runtime은 Conda 환경 `lerobot-arena`의 **Isaac Lab 2.3.0, Isaac Sim 5.1.0, Python 3.11.16, PyTorch 2.7.0+cu128**다. Isaac Sim과 해당 환경은 외부 실행 요건이다. Repository 루트에서 검증 당시와 같은 source 경로를 지정한다.

```bash
conda activate lerobot-arena
unset LD_PRELOAD ISAAC_SIM_SITE_PACKAGES ISAAC_PATH CARB_APP_PATH EXP_PATH
export ISAACLAB_RS="$PWD"
export ISAACLAB_ROOT="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/source:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"
```

위 설정은 repository에 포함된 source를 선택한다. Framework package가 겹쳐 있는 루트 package의 editable installation은 검증하지 않았다. Task를 runtime에서 import하려면 AppLauncher/Kit 초기화가 필요하다. Framework 관련 정보는 [source provenance](docs/submission_provenance/README.md)와 [원본 framework 문서](docs/submission_provenance/official_README.md)를 참고한다.

평가는 아래 전용 evaluator를 사용하고 **새 output 디렉터리**를 지정한다. 이미 존재하는 디렉터리는 사용할 수 없다. 각 evaluator는 기록된 protocol에 따라 evaluation seed, 환경 수, deterministic action, 첫 episode 집계 방식, 평가 길이를 고정한다. 일반 `play.py`는 평가 reward를 변경하므로 아래 evaluator를 대신하는 용도로 사용하지 않는다.

```bash
./isaaclab.sh -p scripts/observation_ablation_budget.py evaluate --headless \
  --checkpoint experiments/observation_ablation/heightscan_4096x32x1000/checkpoints/best_model.pt \
  --output experiments/observation_ablation/validation/reproduction_heightscan_stock

./isaaclab.sh -p scripts/contact_stock_observation_ablation.py evaluate --headless \
  --checkpoint experiments/observation_ablation/heightscan_contact_stock_4096x32x1000/checkpoints/best_model.pt \
  --output experiments/observation_ablation/validation/reproduction_contact_stock

./isaaclab.sh -p scripts/stage2_observation_ablation.py evaluate --headless \
  --checkpoint experiments/observation_ablation/heightscan_contact_modified_4096x32x1000/checkpoints/best_model.pt \
  --output experiments/observation_ablation/validation/reproduction_contact_modified
```

학습 명령과 설정은 각 실험의 command log와 manifest에 보존되어 있다. 실험 조건은 [Stock protocol](experiments/observation_ablation/protocol.json)과 [Modified Reward protocol](experiments/observation_ablation/shared/stage2_contact_modified_protocol.json)에 정리되어 있다. Modified evaluator는 [logs/rsl_rl/ant/modified/params/](logs/rsl_rl/ant/modified/params/)의 reward provenance를 읽으며, [config mapping](docs/submission_provenance/modified_config_mapping.json)에 source hash가 기록되어 있다.

## 검증 및 재현성

기록된 submission validation에서 세 조건의 canonical evaluation을 100개 환경에서 재현했으며, summary와 episode CSV가 기존 결과와 일치했다. [최종 평가 검증 자료](experiments/observation_ablation/validation/final_submission_check/final_100env_evaluation/)는 원본 실험 결과와 별도로 보존되어 있다.

[최종 검증 기록](docs/submission_provenance/heightscan_cleanup/final_verification/README.md)에는 프로젝트의 다섯 Ant task에 대한 import/config 검사, Python compile, checkpoint/config input의 일치 여부, **473개 보호 artifact의 byte 단위 동일성 유지**가 기록되어 있다. Rendering과 video 기능도 유지되어 있다. 이 cleanup 검증은 기존 기록이며, 새 공통 Stock Reward 평가의 검증은 [별도 기록](experiments/observation_ablation/common_stock_reward_evaluation/integrity_validation.json)으로 보존한다. 기존 canonical 결과와 checkpoint는 변경하지 않았다.

Checkpoint는 학습 전에 정한 기준인 완료 episode의 최고 logged training mean return으로 선택하며, evaluation 결과를 이용해 선택하지 않는다. Selected iteration은 결과 표의 조건 순서대로 804, 972, 788이다. 각 `final_model.pt`에는 iteration 999가 보존되어 있다. Hash, source mapping, checkpoint selection 기록은 상세 실험 README에서 확인할 수 있다. Historical manifest는 작성 당시 snapshot과 원래 경로를 기록한 자료이므로, 현재 파일 목록이나 다른 환경에서 그대로 실행할 수 있는 명령 목록으로 해석하지 않는다.

## 한계

- **Baseline 결과 부재:** 동일 학습 예산의 baseline 측정값과 checkpoint가 없다. Stage 1은 연구 질문과 비교 설계를 제시하지만 baseline 대비 개선을 입증하지 않는다.
- **Single-seed 평가:** Canonical training은 seed 42, evaluation은 seed 24를 사용하며, 기록된 하나의 terrain realization에 대한 결과다. Terrain mesh hash, 초기 world-X 위치, row/column 배정의 일치는 조건 비교를 뒷받침하지만 일반적인 통계적 유의성이나 임의의 지형으로의 전이를 입증하지는 않는다.
- **학습 조건과 exploratory run:** 보존된 세 run은 1000 updates, 학습 예산, seed, 초기 terrain 배정 유지, curriculum 미사용, native reset 설정을 공유한다. 이전 exploratory HeightScan run은 2048 × 32 × 10000 = 655,360,000 transitions를 사용했다. 동일 조건의 baseline이 아니며, 환경 수나 observation의 효과를 분리하는 비교도 아니다.
- **맵 경계:** Map-boundary 및 torso-height termination이 없다. 생성 지형의 X 범위 `[-102, 102]` m를 벗어난 이동도 displacement와 progress에 포함될 수 있으므로, 보고된 거리 전체를 불규칙 지형 위의 보행 거리로 해석하지 않는다.
- **Reward 해석:** Stock Reward와 Modified Reward의 return scale은 다르다. 이 seed에서는 Modified Reward의 fall/displacement 결과가 더 나빴다. 집계 지표만으로 hopping 감소, 더 안정적인 gait, 개별 reward term의 효과를 확인할 수 없다.
- **평가 범위:** Canonical 표는 native termination 규칙에서 deterministic 첫 episode를 평가한 결과다. 보충적인 terrain-transfer/video 평가는 별도 맥락을 가지며, 추가적인 통제 학습 실험으로 간주하지 않는다.
- **Runtime 재현성:** Evaluation process는 종료 코드 0으로 끝났지만, 최종 `SimulationApp.close()`의 반환 marker는 확인되지 않았다. Shutdown 진단은 [final_submission_check/](experiments/observation_ablation/validation/final_submission_check/)에 보존되어 있으며, application-close 완료를 주장하지 않는다. Historical 경로와 finalizer를 재사용하기 전에는 별도 검토가 필요하다.

## 상세 실험 문서

[experiments/observation_ablation/README.md](experiments/observation_ablation/README.md)는 실험 설계, task 이름, observation dimension, 학습 조건, evaluation protocol, 지표, artifact 목록, 결과 해석과 caveat를 정리한 **기준 문서(source of truth)**다. 루트 README는 해당 기록을 요약한 제출용 안내 문서다. 전체 terrain parameter, PPO 설정, reward 정의, decomposition 표, checkpoint hash는 상세 실험 문서를 기준으로 확인한다.
