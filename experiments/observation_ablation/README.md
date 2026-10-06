# Terrain Observation and Locomotion Ablation

## 프로젝트 개요

본 프로젝트는 Isaac-Ant 환경을 기반으로, 복잡한 terrain에서 주변 지형 정보와 foot-contact feedback을 활용해 안정적으로 이동하는 PPO policy를 개발하고 비교하는 것을 목표로 한다. Robot 구조와 기본 control framework는 유지하면서 terrain representation, contact observation, reward shaping을 단계적으로 변경하고, 각 요소가 locomotion에 미치는 영향을 ablation 방식으로 분석한다.

Return뿐 아니라 displacement, episode duration, 평균 전진 속도, fall/timeout, reach rate, reward decomposition을 함께 기록한다. 각 Stage에서 가능한 한 하나의 핵심 변수만 바꾸도록 비교를 구성하되, 실제 학습 조건의 차이와 검증 범위도 함께 명시한다.

## 연구 질문

1. **Terrain perception:** HeightScan과 DepthCam 중 어떤 terrain representation이 locomotion에 더 효과적인가?
2. **Contact feedback:** HeightScan policy에 explicit 4-D foot-contact observation을 추가하면 locomotion 특성이 어떻게 변하는가?
3. **Reward shaping:** HeightScan + Contact observation을 고정한 상태에서 Modified Reward가 locomotion 특성에 어떤 영향을 주는가?

## 전체 실험 구성

| 단계 | 비교 | 고정되는 것 | 확인하려는 효과 |
|---|---|---|---|
| **Stage 1 — Terrain perception** | HeightScan + Stock vs DepthCam + Stock | Reward = Stock | Terrain representation 효과 |
| **Stage 2 — Contact feedback** | HeightScan + Stock vs HeightScan + Contact + Stock | Terrain perception = HeightScan, Reward = Stock | 4-D Contact observation 효과 |
| **Stage 3 — Reward shaping** | HeightScan + Contact + Stock vs HeightScan + Contact + Modified | Observation = HeightScan + Contact | Modified Reward 효과 |

Stage 1에서는 Stock Reward를 고정하고 HeightScan과 DepthCam을 비교한다. Stage 2에서는 HeightScan과 Stock Reward를 고정하고 4-D Contact observation 추가 효과를 본다. Stage 3에서는 HeightScan + Contact observation을 고정하고 reward만 Stock → Modified로 변경한다.

> **실제 통제 범위:** Stage 2·3의 HeightScan 비교는 저장된 config로 환경·PPO·budget parity를 확인했다. Stage 1은 총 transitions를 맞췄지만 병렬 환경 수, update 횟수, sensor encoder뿐 아니라 학습 중 terrain 재선택 여부도 다르다. 따라서 현재 Stage 1 결과를 terrain representation만의 인과 효과로 단정하지 않는다.

## 공통 환경

현재 [환경 source](../../source/ant/ant_env_cfg.py)와 HeightScan의 저장 config에서 확인한 terrain 구성 및 기본 dynamics는 다음과 같다. DepthCam 평가도 같은 환경 config와 terrain generator를 사용한다. 학습 중 terrain 재선택 차이는 아래 학습 설정에 별도로 기록한다.

| 항목 | 설정 |
|---|---|
| Terrain 종류 | Stairs, inverted stairs, boxes, slope, inverted slope |
| Terrain proportion | 각 0.2 (20%) |
| Terrain layout | 20 × 10 patches |
| Patch 크기 | 10 × 10 m |
| Terrain seed | 42 |
| Curriculum / difficulty range | 사용 안 함 / [0.0, 1.0] |
| Physics dt / decimation | 1/120 s / 2 |
| Control dt | 1/60 s |
| Episode 길이 | 16 s / 960 control steps |
| Action | 8-D joint effort, scale 7.5 |
| Root reset | `reset_root_state_uniform`, 추가 pose/velocity 범위 없음 |
| Joint reset | Position ±0.2, velocity ±0.1 |
| Startup friction | Static friction 0.3–1.0, dynamic friction = 0.8 × static friction |
| Termination | Timeout 또는 `body_z_down(pi/2)` |
| Map-boundary / torso-height termination | 없음 |

| Terrain | Config key | Proportion | Parameter 범위 / 크기 | Platform width |
|---|---|---:|---|---:|
| Stairs | `pyramid_stairs` | 0.2 (20%) | Step height: 0.03–0.07 m, step width: 0.3 m | 1.0 m |
| Inverted stairs | `pyramid_stairs_inv` | 0.2 (20%) | Step height: 0.03–0.07 m, step width: 0.3 m | 1.0 m |
| Boxes | `boxes` | 0.2 (20%) | Grid width: 0.45 m, `grid_height_range`: 0.02–0.10 m | 1.0 m |
| Slope | `hf_pyramid_slope` | 0.2 (20%) | `slope_range`: 0.0–0.20, `inverted=False` | 1.0 m |
| Inverted slope | `hf_pyramid_slope_inv` | 0.2 (20%) | `slope_range`: 0.0–0.20, `inverted=True` | 1.0 m |

`slope_range`는 각도가 아닌 높이 변화량과 수평 거리의 비율이다. Inverted slope에는 반대 방향의 기울기를 적용한다. `random_rough`는 proportion 0.0이므로 생성 대상에서 제외된다.

## 학습 설정

### Training budget

DepthCam은 image/CNN 처리에 따른 GPU memory 사용량을 고려해 병렬 환경 수를 2048로 줄이고 iterations를 2000으로 늘렸다. Rollout length는 32 steps로 유지했으며, training summary에 기록된 총 transitions는 두 representation 모두 **131,072,000**이다.

| 항목 | HeightScan 계열 3개 실험 | DepthCam 계열 2개 실험 |
|---|---:|---:|
| 병렬 환경 수 | 4096 | 2048 |
| 환경별 iteration당 rollout steps | 32 | 32 |
| Iterations | 1000 | 2000 |
| 총 transitions | 4096 × 32 × 1000 = 131,072,000 | 2048 × 32 × 2000 = 131,072,000 |
| Training seed | 42 | 42 |
| Terrain seed (공유 source) | 42 | 42 |
| 학습 중 terrain patch | 초기 할당 유지 | Reset마다 재선택 (training summary 기록) |
| Resume | False, load_run/load_checkpoint 없음 | 해당 run의 training summary에 미기록 |

**총 transitions가 같아도 모든 학습 조건이 같은 것은 아니다.** DepthCam의 rollout batch는 HeightScan의 절반이고 PPO update 횟수는 두 배다. 또한 DepthCam summary의 `terrain_reselected_on_reset=true`는 HeightScan의 reset 설정과 다르다. 현재 source에는 이 재선택 event가 없고 해당 DepthCam run의 저장 env config도 포함되어 있지 않아, 학습 당시 구현의 세부사항까지는 직접 검증할 수 없다.

### 공통 PPO 및 network 설정

아래 값은 현재 [PPO source](../../source/ant/agents/rsl_rl_ppo_cfg.py)와 HeightScan 3개 실험의 저장 agent config에서 대조했다. HeightScan은 이 공통 PPO를 상속하며 CNN 없이 feature를 concatenate한다. DepthCam의 run별 저장 agent config는 이 repository에 없어 실행 당시 전체 hyperparameter의 독립적인 재검증 범위에는 제한이 있다.

| 항목 | 설정 |
|---|---|
| Actor/Critic post-feature MLP | [400, 200, 100], ELU |
| Action distribution / 초기 noise std | Gaussian, scalar std / 1.0 |
| Actor/Critic empirical observation normalization | 사용 안 함 |
| Learning rate / schedule | 0.0005 / adaptive |
| Gamma / lambda | 0.99 / 0.95 |
| Clip parameter / entropy coefficient | 0.2 / 0.0 |
| Value loss coefficient / clipped value loss | 1.0 / 사용 |
| Learning epochs / minibatches | 5 / 4 |
| Desired KL / max grad norm | 0.01 / 1.0 |
| Minibatch별 advantage normalization | 사용 안 함 |

### Checkpoint 선택

선택 기준은 **highest logged training mean return**이며, 각 실험의 training reward 기준으로 `best_model.pt`를 사용한다. HeightScan의 selection 기록은 학습 전에 규칙을 고정했음을 명시한다. DepthCam은 [training runner](../../scripts/reinforcement_learning/rsl_rl/train.py)의 best-model 저장 규칙과 training/evaluation summary를 대조했다. 서로 다른 reward로 계산된 training mean return끼리는 직접 비교하지 않는다.

| 실험 | Selected iteration | Training mean return | Checkpoint |
|---|---:|---:|---|
| HeightScan + Stock | 804 | 53.0052 | `best_model.pt` |
| DepthCam + Stock | 905 | 61.5568 | `best_model.pt` |
| HeightScan + Contact + Stock | 972 | 53.8264 | `best_model.pt` |
| HeightScan + Contact + Modified | 788 | 107.4448 | `best_model.pt` |
| DepthCam + Modified | 1974 | 129.2691 | `best_model.pt` |

Iteration은 저장된 checkpoint의 번호를 그대로 사용한다. HeightScan 3개 실험은 마지막 iteration 999의 `final_model.pt`도 별도로 보존한다.

## Observation 설계

### Proprioception

공통 proprio interface는 **59-D**이다. Base velocity/orientation, target direction, joint position/velocity, 이전 action과 **24-D incoming foot wrench**를 포함한다. 아래의 explicit 4-D Contact는 이 wrench와 별개의 추가 feature다.

### HeightScan

| 항목 | 설정 |
|---|---|
| 출력 / Stage 1 actor·critic 입력 | 63-D / 59 + 63 = 122-D |
| Sensor 부착 / 정렬 | Torso / yaw-aligned |
| Ray pattern | 9 × 7 downward rays, spacing 0.2 m |
| Offset | `(0.8, 0, 20)` |
| Height 계산 | `sensor_z - hit_z - 0.5` |
| Scale / clipping | 1 / `[-1, 1]` |
| CNN / empirical normalization | 없음 / 없음 |

HeightScan 3개 실험은 동일한 preprocessing을 사용한다. HeightScan과 explicit Contact 통합은 기존 IsaacLab_RS HeightScan+Contact 구현을 참고했다. [HeightScan source](../../source/ant/ablation_env_cfg.py)를 따른다.

### DepthCam

[Camera config](../../source/ant/ant_env_cfg.py), [preprocessing](../../source/ant/depth_obs.py), [CNN source](../../source/ant/depth_actor_critic.py)를 기준으로 정리한 구성이다. 평가 script는 두 DepthCam checkpoint 모두 proprio `(100, 59)`, depth `(100, 48, 64, 1)`을 확인한다.

| 항목 | 설정 |
|---|---|
| Depth image | 48 × 64, 1 channel, 15 Hz |
| Distance range / preprocessing | 0.1–5.0 m clipping 후 [0, 1]로 정규화 |
| CNN | Conv(1→16, k5/s2/p2) → Conv(16→32, k3/s2/p1) → Conv(32→32, k3/s2/p1) |
| Feature projection | AdaptiveAvgPool(2×2) → Flatten → Linear(128→64), 각 conv·linear 뒤 ELU |
| Depth feature | 64-D, actor/critic이 encoder 공유 |
| Actor/Critic post-feature 입력 | 59-D proprio + 64-D feature = 123-D |
| 별도 explicit Contact | 두 DepthCam 모델 모두 없음 |

Image의 NaN/+Inf는 far distance, -Inf는 near distance로 치환한 뒤 clipping한다. Sensor마다 입력 형태가 다르므로 적합한 preprocessing과 feature encoder를 사용하며, 차원을 맞추는 dummy feature는 추가하지 않는다.

### Contact observation

[공통 Contact spec](shared/contact_observation.json)에 따른 현재 sample의 **world-frame net-force norm > 1 N** 여부를 float32 binary로 표현한다. History length는 **0**, scale은 **1**, clipping과 normalization은 없다. Feature 순서는 다음과 같다.

1. `front_left_foot`
2. `front_right_foot`
3. `left_back_foot`
4. `right_back_foot`

HeightScan + Contact의 actor/critic 입력은 **59 + 63 + 4 = 126-D**이며 Contact는 마지막에 concatenate한다. Stock 조건에서는 Contact를 observation에만 추가하고 reward-side contact term은 활성화하지 않는다.

## Reward 설계

### Stock Reward

Canonical Isaac-Ant의 7개 reward 함수와 weight를 사용하며 추가 shaping이나 weight tuning은 하지 않았다. [Stock HeightScan config](../../source/ant/ablation_env_cfg.py)와 [Contact + Stock config](../../source/ant/contact_stock_env_cfg.py)는 같은 reward를 사용한다.

| Component | Weight |
|---|---:|
| progress | 1.0 |
| alive | 0.5 |
| upright | 0.1 |
| move_to_target | 0.5 |
| action_l2 | -0.005 |
| energy | -0.05 |
| joint_pos_limits | -0.1 |

### Modified Reward

전진 중심 reward에서 반복적인 jumping/hopping 형태의 locomotion이 나타나는 경향을 완화하고, ground contact, foot slip, joint motion, control effort를 함께 고려하도록 reward objective를 확장하기 위해 사용했다. 이는 설계 목적이며, Modified Reward가 안정적인 gait나 성능 향상을 보장한다는 의미는 아니다.

실제 구현은 v3_depth 학습에 사용한 [ant.rewards.TotalReward](../../source/ant/rewards.py)를 재사용한다. RewardManager에는 `total_reward` 하나가 weight 1.0으로 등록되고, 내부에 다음 weighted component가 있다. 학습과 평가 모두 해당 reward를 사용하며 play용 evaluation reward로 교체하지 않는다.

| Component | Weight |
|---|---:|
| progress | 2.5 |
| alive | 0.5 |
| upright | 0.05 |
| move_to_target | 1.5 |
| foot_contact | 1.0 |
| action_l2 | -0.005 |
| energy | -0.15 |
| joint_velocity | -0.001 |
| joint_pos_limits | -0.5 |
| foot_slip | -0.07 |

Stock 대비 progress와 target-direction weight를 높이고 energy·joint-limit penalty를 강화했다. Foot-contact reward, joint-velocity penalty, foot-slip penalty를 추가하고 upright weight를 낮췄다. 이 비교는 reward objective 전체의 변경이며 개별 term의 효과를 분리하지 않는다.

> **Contact 정의 구분:** Reward-side Contact는 3-frame 최대 vertical force > 5 N을 사용한다. 최소 두 발이 접촉하면 contact bonus를 부여하고, foot-slip penalty는 접촉 중인 발의 XY 속도를 합산한다. Observation-side Contact의 현재 net-force norm > 1 N, history 0 정의와는 다르다.

## Stage 1 — Terrain perception

### 설정

**HeightScan + Stock vs DepthCam + Stock.** Stock Reward를 고정하고 terrain representation을 비교한다. HeightScan은 63-D terrain heights를 MLP에 입력하고, DepthCam은 depth image를 CNN의 64-D feature로 변환한 뒤 MLP에 입력한다.

공통 PPO source, seed, terrain seed, 총 training transitions 및 평가 절차를 맞췄다. 다만 병렬 환경 수·update 횟수·encoder와 학습 terrain 재선택 여부가 달라, 현재 비교는 완전히 통제된 단일 변수 실험이 아니다.

### 결과

| 지표 | HeightScan + Stock | DepthCam + Stock |
|---|---:|---:|
| Return 평균 ± 표준편차 | 61.3354 ± 31.2305 | 57.0139 ± 26.9578 |
| Displacement 평균 ± 표준편차 | 60.5140 ± 29.1880 m | 53.1530 ± 25.3308 m |
| Episode duration 평균 ± 표준편차 | 12.1947 ± 5.5930 s | 12.4178 ± 5.5163 s |
| 평균 전진 속도 | 4.4506 m/s | 3.8172 m/s |
| Fall | 41/100 | 41/100 |
| Timeout | 59/100 | 59/100 |
| Other | 0/100 | 0/100 |
| ≥2 m | 87/100 | 89/100 |
| ≥5 m | 87/100 | 89/100 |
| ≥10 m | 86/100 | 88/100 |
| Out-of-terrain-X | 28/100 | 23/100 |

### Reward decomposition

같은 Stock Reward contribution의 평균 ± population std이다.

| Component | HeightScan + Stock | DepthCam + Stock |
|---|---:|---:|
| progress | 60.4711 ± 29.1633 | 53.1379 ± 25.2545 |
| alive | 6.0939 ± 2.7998 | 6.2055 ± 2.7614 |
| upright | 1.0924 ± 0.5508 | 1.1881 ± 0.5556 |
| move_to_target | 5.3633 ± 2.6815 | 6.0237 ± 2.7660 |
| action_l2 | -1.2979 ± 8.7617 | -0.1404 ± 0.0631 |
| energy | -7.2101 ± 3.5063 | -6.4442 ± 3.0680 |
| joint_pos_limits | -3.1773 ± 1.6496 | -2.9568 ± 1.3043 |
| total | 61.3354 ± 31.2305 | 57.0139 ± 26.9578 |

### 해석

이 single-seed 평가에서는 HeightScan의 mean return, displacement, 평균 전진 속도가 더 높았고 fall/timeout 수는 동일했다. DepthCam은 ≥5 m 도달 episode가 더 많았다. 학습 terrain-reset과 update 조건 차이가 있으므로 이 결과만으로 HeightScan이 일반적으로 더 효과적이라고 결론내리지 않는다.

## Stage 2 — Contact feedback

### 설정

**HeightScan + Stock vs HeightScan + Contact + Stock.** Environment, HeightScan preprocessing, Stock Reward, PPO, actor/critic MLP, seed 42, terrain seed 42, 4096 × 32 × 1000 budget 및 평가 절차를 고정했다. 변경되는 핵심 변수는 **explicit 4-D Contact observation 추가**이며, actor/critic 입력은 122-D → 126-D로 바뀐다.

### 결과

| 지표 | HeightScan + Stock | HeightScan + Contact + Stock |
|---|---:|---:|
| Return 평균 ± 표준편차 | 61.3354 ± 31.2305 | 63.5682 ± 28.5777 |
| Displacement 평균 ± 표준편차 | 60.5140 ± 29.1880 m | 61.8529 ± 27.8017 m |
| Episode duration 평균 ± 표준편차 | 12.1947 ± 5.5930 s | 12.9088 ± 5.3955 s |
| 평균 전진 속도 | 4.4506 m/s | 4.3692 m/s |
| Fall | 41/100 | 29/100 |
| Timeout | 59/100 | 71/100 |
| Other | 0/100 | 0/100 |
| ≥2 m | 87/100 | 90/100 |
| ≥5 m | 87/100 | 89/100 |
| ≥10 m | 86/100 | 88/100 |
| Out-of-terrain-X | 28/100 | 27/100 |

### Reward decomposition

두 policy 모두 같은 Stock Reward를 사용하므로 component별 weighted contribution을 직접 비교할 수 있다.

| Component | HeightScan + Stock | HeightScan + Contact + Stock |
|---|---:|---:|
| progress | 60.4711 ± 29.1633 | 61.8339 ± 27.8095 |
| alive | 6.0939 ± 2.7998 | 6.4520 ± 2.7012 |
| upright | 1.0924 ± 0.5508 | 1.1876 ± 0.5511 |
| move_to_target | 5.3633 ± 2.6815 | 5.8745 ± 2.6763 |
| action_l2 | -1.2979 ± 8.7617 | -0.5343 ± 1.3955 |
| energy | -7.2101 ± 3.5063 | -7.6240 ± 3.3849 |
| joint_pos_limits | -3.1773 ± 1.6496 | -3.6215 ± 1.6988 |
| total | 61.3354 ± 31.2305 | 63.5682 ± 28.5777 |

### 해석

이 single-seed evaluation에서 Contact observation을 추가한 policy는 fall count가 41→29, timeout count가 59→71로 변했고, mean displacement는 60.5140→61.8529 m로 소폭 증가했다. 다만 single training seed 결과이므로 일반적인 통계적 유의성이나 Contact가 항상 성능을 향상시킨다고 주장하지 않는다.

## Stage 3 — Reward shaping

### 설정

**HeightScan + Contact + Stock vs HeightScan + Contact + Modified.** 59-D proprio + 63-D HeightScan + 4-D Contact의 **126-D observation**과 environment, PPO, MLP, seed, terrain seed, 4096 × 32 × 1000 budget 및 평가 절차를 고정했다. 변경되는 핵심 변수는 **reward definition**이다. 각 policy는 학습에 사용한 reward로 평가한다.

> **Return 해석:** Stock Reward와 Modified Reward는 reward term과 scale이 다르므로 두 조건의 total return 절대값은 직접적인 성능 향상량으로 해석하지 않는다. 아래 return은 각 objective 기준의 기록이며, 성능을 판단할 때는 displacement, duration, fall/timeout, reach rate와 control-related behavior를 중심으로 본다.

### 결과

| 지표 | HeightScan + Contact + Stock | HeightScan + Contact + Modified |
|---|---:|---:|
| Return (각 조건의 Reward 기준) | 63.5682 ± 28.5777 | 134.8413 ± 68.0187 |
| Displacement 평균 ± 표준편차 | 61.8529 ± 27.8017 m | 56.5627 ± 28.3639 m |
| Episode duration 평균 ± 표준편차 | 12.9088 ± 5.3955 s | 12.1187 ± 5.6815 s |
| 평균 전진 속도 | 4.3692 m/s | 4.0718 m/s |
| Fall | 29/100 | 43/100 |
| Timeout | 71/100 | 57/100 |
| Other | 0/100 | 0/100 |
| ≥2 m | 90/100 | 86/100 |
| ≥5 m | 89/100 | 86/100 |
| ≥10 m | 88/100 | 84/100 |
| Out-of-terrain-X | 27/100 | 26/100 |

### Reward decomposition

각 조건의 실제 weighted contribution이다. 공통 component 이름도 weight와 구현이 같다고 가정하지 않으며, 아래 값을 동일 scale의 증감량으로 해석하지 않는다.

| Component | HeightScan + Contact + Stock | HeightScan + Contact + Modified |
|---|---:|---:|
| progress | 61.8339 ± 27.8095 | 141.3719 ± 70.9343 |
| alive | 6.4520 ± 2.7012 | 6.0558 ± 2.8440 |
| upright | 1.1876 ± 0.5511 | 0.5720 ± 0.2875 |
| move_to_target | 5.8745 ± 2.6763 | 16.4496 ± 8.0588 |
| foot_contact | — (미사용) | 2.5842 ± 1.3074 |
| action_l2 | -0.5343 ± 1.3955 | -0.1891 ± 0.3820 |
| energy | -7.6240 ± 3.3849 | -19.2751 ± 9.6769 |
| joint_velocity | — (미사용) | -2.6589 ± 1.3529 |
| joint_pos_limits | -3.6215 ± 1.6988 | -8.5355 ± 5.0240 |
| foot_slip | — (미사용) | -1.5334 ± 0.7835 |
| total | 63.5682 ± 28.5777 | 134.8413 ± 68.0187 |

### 해석

이 seed에서는 Modified 조건의 mean displacement와 duration이 낮고 fall count가 29→43으로 증가했다. Total return이 높다는 사실을 성능 향상으로 해석하지 않는다. Jumping/hopping 완화 여부나 더 안정적인 gait를 이 집계 지표만으로 확정할 수 없으며, 개별 reward term의 효과도 분리할 수 없다.

## 참고: DepthCam + Modified Reward

이 실험은 **59-D proprio + depth CNN 64-D feature = 123-D** 입력을 사용하고, 별도의 4-D explicit Contact observation은 없다. `contact_depth`라는 run 이름은 Contact observation 추가를 뜻하지 않는다. Modified Reward에는 reward-side contact/slip term이 포함되지만 observation-side Contact와는 구분한다.

HeightScan + Contact + Modified와 observation 구성이 일치하지 않고 학습 terrain-reset 조건도 달라, 본 3-stage ablation의 controlled terrain-perception 직접 비교에는 포함하지 않는다. 확보된 결과는 참고용으로 기록한다.

| 지표 | DepthCam + Modified |
|---|---:|
| Return (Modified Reward) | 120.4836 ± 63.8864 |
| Displacement | 48.6275 ± 25.8809 m |
| Episode duration | 12.0768 ± 5.8787 s |
| 평균 전진 속도 | 3.4589 m/s |
| Fall | 41/100 |
| Timeout | 59/100 |
| ≥2 m | 86/100 |
| ≥5 m | 85/100 |
| ≥10 m | 82/100 |
| Out-of-terrain-X | 19/100 |

[평가 요약](depthcam_modified_2048x32x2000/results/evaluation_summary.json)과 [reward decomposition](depthcam_modified_2048x32x2000/results/reward_components.csv)에 전체 수치를 보존한다.

## 평가 protocol

| 항목 | 설정 |
|---|---|
| Evaluation seed / 환경 수 | 24 / 100 |
| Policy action | Deterministic mean action |
| Episode | 각 환경의 첫 episode만 사용, 모두 100/100 완료 |
| 평가 reward | 각 실험의 training reward와 동일 |
| 최대 길이 | 16 s / 960 control steps |
| Terminal reward / post-reset reward | 포함 / 제외 |
| Displacement | 종료 시점 world-X − 초기 world-X |
| Mean vx | 각 episode의 step별 world-X velocity 평균을 100 episode에 대해 평균 |
| 표준편차 | Population std (`ddof=0`) |

HeightScan의 평가 manifest와 [DepthCam 평가 script](../../scripts/evaluate_depthcam_ablation.py), evaluation summary 및 episode CSV를 대조했다. 다섯 결과의 terrain mesh SHA256과 환경별 initial world-X는 일치한다. HeightScan 3개 결과의 terrain row/column 배정도 일치한다. DepthCam CSV에는 row/column이 없어 해당 배정의 전체 직접 비교는 할 수 없다.

Reward decomposition은 실제 RewardManager contribution을 누적한다. Stock은 `raw × weight × control_dt`, Modified는 `TotalReward`의 실제 internal weighted component × manager weight 1 × control dt이며 dt를 중복 적용하지 않는다. 각 실험의 component sum과 공식 episode return residual은 평가 요약에 기록되어 있다.

## 한계 및 해석 시 주의점

- **Stage 1 통제의 한계:** 동일 transitions에도 env count, rollout batch, PPO update 횟수와 terrain 재선택 여부가 다르다. Encoder와 feature 차원도 63-D HeightScan과 64-D depth feature로 다르다.
- **Stage 3 reward scale:** Stock과 Modified total return을 차감하거나 배수로 비교해 성능 향상량을 주장하지 않는다. Reward component도 서로 다른 weight·objective 기준이다.
- **Terrain 경계:** Map-boundary termination이 없어 displacement와 progress에 generated terrain X bounds `[-102, 102]` m 밖의 이동이 포함될 수 있다. 보고된 displacement 전체를 rough-terrain locomotion 거리로 해석하지 않는다.
- **단일 seed:** 모든 학습 summary의 training seed는 42다. PPO의 확률적 변동이 있으므로 일반적인 통계적 유의성이나 항상 성립하는 개선을 주장하지 않는다.
- **DepthCam 검증 범위:** 이 checkout에는 두 DepthCam run의 원본 checkpoint와 저장 agent/env config가 없다. Training/evaluation summary의 checkpoint SHA는 일치하지만 tensor·학습 당시 전체 config 수준의 재검증은 수행할 수 없다. 현재 source와 summary에 근거한 구성과, 직접 확인 가능한 HeightScan artifact의 검증 범위를 구분한다.

## Artifact / checkpoint 위치

| 실험 | Training 기록 | Evaluation 기록 | Checkpoint |
|---|---|---|---|
| HeightScan + Stock | [Summary](heightscan_4096x32x1000/training_summary.json) | [Summary](heightscan_4096x32x1000/evaluation_summary.json) | [Best](heightscan_4096x32x1000/checkpoints/best_model.pt), [마지막 iteration](heightscan_4096x32x1000/checkpoints/final_model.pt) |
| HeightScan + Contact + Stock | [Summary](heightscan_contact_stock_4096x32x1000/training_summary.json) | [Summary](heightscan_contact_stock_4096x32x1000/evaluation_summary.json) | [Best](heightscan_contact_stock_4096x32x1000/checkpoints/best_model.pt), [마지막 iteration](heightscan_contact_stock_4096x32x1000/checkpoints/final_model.pt) |
| HeightScan + Contact + Modified | [Summary](heightscan_contact_modified_4096x32x1000/training_summary.json) | [Summary](heightscan_contact_modified_4096x32x1000/evaluation_summary.json) | [Best](heightscan_contact_modified_4096x32x1000/checkpoints/best_model.pt), [마지막 iteration](heightscan_contact_modified_4096x32x1000/checkpoints/final_model.pt) |
| DepthCam + Stock | [Summary](depthcam_2048x32x2000/training_summary.json) | [Summary](depthcam_2048x32x2000/results/evaluation_summary.json) | Summary에 기록된 `logs/rsl_rl/ant/base_depth/best_model.pt` (미포함) |
| DepthCam + Modified | [Summary](depthcam_modified_2048x32x2000/training_summary.json) | [Summary](depthcam_modified_2048x32x2000/results/evaluation_summary.json) | Summary에 기록된 `logs/rsl_rl/ant/contact_depth/best_model.pt` (미포함) |

<details>
<summary>Selected checkpoint SHA256</summary>

| 실험 | SHA256 |
|---|---|
| HeightScan + Stock | `49a8a007f152ed9c63df616fbee12a866873be0ee0337ad83597d653ec9b3c06` |
| DepthCam + Stock | `25ee324c7eeb932f15fbe69515d4593afde06c9c87d78e0ccde02f754e117928` |
| HeightScan + Contact + Stock | `82c7d0f781c37e0620230ff4b401835d54ef0a18102cdbc8c22c28bb576f57b9` |
| HeightScan + Contact + Modified | `0471caaadd646feac85cb7bbed0d2c3668279d2d65db9507030a649895ea11b4` |
| DepthCam + Modified | `2ea3326e502cc1b79d1af3e4bd2abd723f5f3da3abe59504c5f5d5be198e3d45` |

</details>

[기존 Stock protocol](protocol.json), [Contact spec](shared/contact_observation.json), [기존 Contact + Modified protocol](shared/stage2_contact_modified_protocol.json)을 함께 보존한다. 기존 `stage2_*` 파일명과 JSON의 Stage 번호는 이전 비교 구조의 기록이다. 이 README에서는 해당 HeightScan Contact + Modified 결과를 **Stage 3**에 배치했으며, 원본 source·protocol·결과 파일은 변경하지 않았다.

[이전 exploratory HeightScan run](heightscan/manifest.json)은 2048 × 32 × 10000 조건의 참고 자료로 보존한다. 위의 131,072,000-transition 비교에는 포함하지 않는다.

## 현재 상태

| 실험 | 상태 | 본 문서의 배치 |
|---|---|---|
| HeightScan + Stock | 학습·평가 완료 | Stage 1, Stage 2 |
| DepthCam + Stock | 학습 summary·평가 결과 확보 | Stage 1 (학습 조건 차이 명시) |
| HeightScan + Contact + Stock | 학습·평가 완료 | Stage 2, Stage 3 |
| HeightScan + Contact + Modified | 학습·평가 완료 | Stage 3 |
| DepthCam + Modified | 학습 summary·평가 결과 확보 | 참고 실험 (별도 Contact observation 없음) |

## Submission copy 실행 경로

이 문서는 기존 결과·해석·Stage 구조를 유지한다. Submission의 RL entry point는 `scripts/reinforcement_learning/rsl_rl/`에 있으며, canonical 평가는 기존 전용 evaluator를 사용한다. [Submission README](../../README.md)의 환경 설정과 [별도 path audit](validation/final_submission_check/path_audit.md)을 확인한다. `ISAACLAB_RS`와 `ISAACLAB_ROOT`는 submission root를 지정하며, historical command log와 manifest의 원래 경로는 수정하지 않았다. Runtime import 및 checkpoint 호환성 검증이 완료되기 전에는 새 평가를 실행하지 않는다.
