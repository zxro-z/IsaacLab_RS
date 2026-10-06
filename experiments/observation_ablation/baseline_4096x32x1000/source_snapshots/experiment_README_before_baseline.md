# Progressive improvement of HeightScan-based terrain-aware locomotion

## 프로젝트 개요

본 프로젝트는 Isaac-Ant 환경을 기반으로, 복잡한 terrain에서 주변 지형 정보와 foot-contact feedback을 활용해 안정적으로 이동하는 PPO policy를 개발하고 비교하는 것을 목표로 한다. Robot 구조와 기본 control framework는 유지하면서 terrain representation, contact observation, reward shaping을 단계적으로 변경하고, 각 요소가 locomotion에 미치는 영향을 ablation 방식으로 분석한다.

Return뿐 아니라 displacement, episode duration, 평균 전진 속도, fall/timeout, reach rate, reward decomposition을 함께 기록한다. 각 Stage에서 가능한 한 하나의 핵심 변수만 바꾸도록 비교를 구성하되, 실제 학습 조건의 차이와 검증 범위도 함께 명시한다.

## 연구 질문 및 단계

**Baseline → HeightScan → HeightScan + Contact → HeightScan + Contact + Modified Reward**

| Variant | HeightScan | Contact Observation | Modified Reward |
|---|---|---|---|
| Baseline (`Ant-rl-Ablation-Baseline-v0`) | No | No | No |
| HeightScan (`Ant-rl-Ablation-HeightScan-v0`) | Yes | No | No |
| HeightScan + Contact (`Ant-rl-Ablation-HeightScan-Contact-Stock-v0`) | Yes | Yes | No |
| Final (`Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0`) | Yes | Yes | Yes |

1. **Terrain perception:** Does explicit local terrain-height information improve locomotion over uneven terrain? Compare the project stock-observation/stock-reward baseline with HeightScan. A matching-budget baseline result is unavailable; no quantitative Stage 1 conclusion is claimed.
2. **Contact feedback:** Does direct contact-state feedback improve robustness beyond terrain geometry alone? Keep HeightScan and stock reward fixed, then add the explicit 4-D Contact observation.
3. **Reward design:** Does reward shaping further improve robust forward locomotion once terrain and contact information are available? Keep HeightScan + Contact fixed, then replace the stock reward with the modified reward.

Baseline uses the existing `AblationBaseCfg` (set `--num_envs 4096` to match the canonical budget) (59-D native proprioception, including incoming foot wrench); cleanup registers this config without training it. The framework stock Ant's 60-D observation is a different interface. Stage 2·3 use preserved environment/PPO/budget parity evidence. These are single-seed descriptive comparisons, not proof that every stage improves performance.

## 공통 환경

현재 [환경 source](../../source/ant/ant_env_cfg.py)와 HeightScan의 저장 config에서 확인한 terrain 구성 및 기본 dynamics는 다음과 같다.

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

The three canonical HeightScan runs use 4096 environments × 32 steps × 1000 PPO updates = **131,072,000 transitions**, training seed 42 and terrain seed 42, fresh initialization (`resume=false`). Terrain patches retain their initial assignment during training; curriculum is disabled. No matching-budget baseline run, seed, training duration, or evaluation result is available. The earlier exploratory run uses 2048 × 32 × 10000 = 655,360,000 transitions and cannot isolate HeightScan's effect from training-budget differences.

### 공통 PPO 및 network 설정

아래 값은 현재 [PPO source](../../source/ant/agents/rsl_rl_ppo_cfg.py)와 HeightScan 3개 실험의 저장 agent config에서 대조했다. HeightScan은 이 공통 PPO를 상속하며 CNN 없이 feature를 concatenate한다.

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

선택 기준은 **highest logged training mean return**이며, 각 실험의 training reward 기준으로 `best_model.pt`를 사용한다. HeightScan의 selection 기록은 학습 전에 규칙을 고정했음을 명시한다. 서로 다른 reward로 계산된 training mean return끼리는 직접 비교하지 않는다.

| 실험 | Selected iteration | Training mean return | Checkpoint |
|---|---:|---:|---|
| HeightScan + Stock | 804 | 53.0052 | `best_model.pt` |
| HeightScan + Contact + Stock | 972 | 53.8264 | `best_model.pt` |
| HeightScan + Contact + Modified | 788 | 107.4448 | `best_model.pt` |

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

실제 구현은 보존된 modified-reward 구현인 [ant.rewards.TotalReward](../../source/ant/rewards.py)를 재사용한다. RewardManager에는 `total_reward` 하나가 weight 1.0으로 등록되고, 내부에 다음 weighted component가 있다. 학습과 평가 모두 해당 reward를 사용하며 play용 evaluation reward로 교체하지 않는다.

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

Compare **Baseline + Stock vs HeightScan + Stock** with native proprioception, stock reward, dynamics, PPO, seeds, resets, terrain assignment, curriculum and training budget held fixed. The existing baseline configuration supports this design, but no matching-budget baseline measurement or checkpoint is present. HeightScan alone reports return **61.3354 ± 31.2305**, displacement **60.5140 ± 29.1880 m**, fall **41/100**, timeout **59/100**. These values describe HeightScan; they do not measure improvement over baseline. No missing baseline values are inferred.

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

## Stage 3 — Reward design

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

## 공통 Stock Reward 기준 과제 평가

기존 canonical 결과는 각 policy를 **학습 objective로 평가한 진단**으로 보존한다. 이와 별도로, 선택된 `best_model.pt` 세 개를 **모두 동일한 Stock Reward**로 평가했다. Modified-trained policy의 training reward는 Modified이고 evaluation reward는 Stock이다. Policy observation과 checkpoint는 각각 **122-D / 126-D / 126-D**로 유지하며 재학습이나 checkpoint 재선택은 하지 않는다.

Evaluation seed 24, terrain seed 42, 100개 환경, deterministic mean action, 첫 episode만 집계, 최대 16 s / 960 steps / 60 Hz를 사용한다. 환경 설정을 비교한 결과 reward 교체 외에는 기존 dynamics·action·termination·reset·terrain·curriculum을 변경하지 않았다. 두 Contact 평가 config는 reward 교체 후 완전히 동일하며, HeightScan-only는 추가 observation용 `feet_contacts` sensor와 4-D term만 다르다. Runtime reward 함수·parameter·weight의 일치를 확인했고 terminal reward를 포함하며 post-reset reward를 제외한다.

Terrain mesh와 환경별 초기 root state, joint position/velocity, origin, terrain row/column, material property가 동일하므로 환경 ID별 paired evaluation이다. 모든 조건은 100/100 첫 episode를 완료했다. 연속형 지표는 평균 ± population std (`ddof=0`)다.

| 학습 조건 | 평가 Reward | Stock Return | Displacement (m) | Duration (s) | Mean vx (m/s) | Fall | Timeout | ≥5 m |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HeightScan + Stock | Stock | 61.3354 ± 31.2305 | 60.5140 ± 29.1880 | 12.1947 ± 5.5930 | 4.4506 ± 1.5331 | 41/100 | 59/100 | 87/100 |
| HeightScan + Contact + Stock | Stock | 63.5682 ± 28.5777 | 61.8529 ± 27.8017 | 12.9088 ± 5.3955 | 4.3692 ± 1.3281 | 29/100 | 71/100 | 89/100 |
| HeightScan + Contact + Modified | Stock | 60.9104 ± 30.3957 | 56.5627 ± 28.3639 | 12.1187 ± 5.6815 | 4.0718 ± 1.5654 | 43/100 | 57/100 | 86/100 |

Stage 2에서 Contact + Stock − HeightScan + Stock의 평균 return 차이는 **+2.2328**, displacement 차이는 **+1.3389 m**, duration 차이는 **+0.7142 s**이며 fall은 12개 적다. Stage 3에서 Modified-trained − Stock-trained Contact policy의 평균 return 차이는 **−2.6578**, displacement 차이는 **−5.2902 m**, duration 차이는 **−0.7902 s**이며 fall은 14개 많다. Modified Reward 학습은 이 공통 Stock Reward objective에서 Stock-trained Contact policy를 능가하지 못했다. 이는 single training seed의 기술적 비교이며 training seed 간 통계적 유의성을 주장하지 않는다.

같은 Stock Reward decomposition에서는 Modified-trained policy의 energy·action·joint-limit penalty가 덜 음수였지만 progress contribution도 낮았다. 제어 penalty 감소가 낮은 전진 성과를 상쇄하지 못했다. Aggregate 결과로 hopping 감소나 gait의 일반적 우월성을 주장하지 않는다. 기존 Modified objective return **134.8413 ± 68.0187**은 위 Stock Return과 직접 비교하지 않는다. 맵 밖 이동은 여전히 displacement와 progress에 포함될 수 있다.

두 Stock policy의 return과 세 policy의 episode별 보행 지표가 기존 canonical CSV와 동일함을 확인했다. 원본 summary·CSV·checkpoint·historical manifest는 보존한다. [별도 평가 기록](common_stock_reward_evaluation/README.md)에는 재현 명령, 공통 reward 명세, decomposition, pairing 및 integrity 검증과 artifact 목록이 있다.

## Canonical 평가 protocol — 학습 objective 기준 진단

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

HeightScan 3개 결과의 terrain mesh SHA256, 환경별 initial world-X 및 terrain row/column 배정이 일치한다. 원본 evaluation manifest와 episode CSV를 보존한다.

Reward decomposition은 실제 RewardManager contribution을 누적한다. Stock은 `raw × weight × control_dt`, Modified는 `TotalReward`의 실제 internal weighted component × manager weight 1 × control dt이며 dt를 중복 적용하지 않는다. 각 실험의 component sum과 공식 episode return residual은 평가 요약에 기록되어 있다.

## 한계 및 해석 시 주의점

- **Stage 3 reward scale:** Stock과 Modified total return을 차감하거나 배수로 비교해 성능 향상량을 주장하지 않는다. Reward component도 서로 다른 weight·objective 기준이다.
- **Terrain 경계:** Map-boundary termination이 없어 displacement와 progress에 generated terrain X bounds `[-102, 102]` m 밖의 이동이 포함될 수 있다. 보고된 displacement 전체를 rough-terrain locomotion 거리로 해석하지 않는다.
- **단일 seed:** 모든 학습 summary의 training seed는 42다. PPO의 확률적 변동이 있으므로 일반적인 통계적 유의성이나 항상 성립하는 개선을 주장하지 않는다.

## Artifact / checkpoint 위치

| 실험 | Training 기록 | Evaluation 기록 | Checkpoint |
|---|---|---|---|
| HeightScan + Stock | [Summary](heightscan_4096x32x1000/training_summary.json) | [Summary](heightscan_4096x32x1000/evaluation_summary.json) | [Best](heightscan_4096x32x1000/checkpoints/best_model.pt), [마지막 iteration](heightscan_4096x32x1000/checkpoints/final_model.pt) |
| HeightScan + Contact + Stock | [Summary](heightscan_contact_stock_4096x32x1000/training_summary.json) | [Summary](heightscan_contact_stock_4096x32x1000/evaluation_summary.json) | [Best](heightscan_contact_stock_4096x32x1000/checkpoints/best_model.pt), [마지막 iteration](heightscan_contact_stock_4096x32x1000/checkpoints/final_model.pt) |
| HeightScan + Contact + Modified | [Summary](heightscan_contact_modified_4096x32x1000/training_summary.json) | [Summary](heightscan_contact_modified_4096x32x1000/evaluation_summary.json) | [Best](heightscan_contact_modified_4096x32x1000/checkpoints/best_model.pt), [마지막 iteration](heightscan_contact_modified_4096x32x1000/checkpoints/final_model.pt) |

<details>
<summary>Selected checkpoint SHA256</summary>

| 실험 | SHA256 |
|---|---|
| HeightScan + Stock | `49a8a007f152ed9c63df616fbee12a866873be0ee0337ad83597d653ec9b3c06` |
| HeightScan + Contact + Stock | `82c7d0f781c37e0620230ff4b401835d54ef0a18102cdbc8c22c28bb576f57b9` |
| HeightScan + Contact + Modified | `0471caaadd646feac85cb7bbed0d2c3668279d2d65db9507030a649895ea11b4` |

</details>

[기존 Stock protocol](protocol.json), [Contact spec](shared/contact_observation.json), [기존 Contact + Modified protocol](shared/stage2_contact_modified_protocol.json)을 함께 보존한다. 기존 `stage2_*` 파일명은 historical implementation naming이며, source names는 재현성을 위해 유지한다. 이 README에서는 해당 HeightScan Contact + Modified 결과를 **Stage 3**에 배치했으며, Reward, HeightScan behavior 및 결과 파일은 변경하지 않았다. Submission-facing protocol의 비교 metadata만 정리했고, 원본 protocol은 cleanup provenance에 보존했다.

[이전 exploratory HeightScan run](heightscan/manifest.json)은 2048 × 32 × 10000 조건의 참고 자료로 보존한다. 위의 131,072,000-transition 비교에는 포함하지 않는다.

## 현재 상태

| 실험 | 상태 | 본 문서의 배치 |
|---|---|---|
| HeightScan + Stock | 학습·평가 완료 | Stage 1, Stage 2 |
| HeightScan + Contact + Stock | 학습·평가 완료 | Stage 2, Stage 3 |
| HeightScan + Contact + Modified | 학습·평가 완료 | Stage 3 |

## Submission copy 실행 경로

이 문서는 HeightScan 중심의 staged ablation을 설명하며 기존 HeightScan 결과를 보존한다. Submission의 RL entry point는 `scripts/reinforcement_learning/rsl_rl/`에 있으며, canonical 평가는 기존 전용 evaluator를 사용한다. [Submission README](../../README.md)의 환경 설정과 [별도 path audit](validation/final_submission_check/path_audit.md)을 확인한다. `ISAACLAB_RS`와 `ISAACLAB_ROOT`는 submission root를 지정하며, historical command log와 manifest의 원래 경로는 수정하지 않았다. 기존 runtime 및 checkpoint interface 검증 기록을 보존하며 cleanup 검증은 별도로 기록한다.


Historical saved configs, source mappings and integrity records are immutable pre-cleanup evidence. See [cleanup audit](../../docs/submission_provenance/heightscan_cleanup/README.md) for intentional removals and current validation.
