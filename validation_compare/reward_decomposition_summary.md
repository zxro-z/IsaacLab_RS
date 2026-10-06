# RS Contact in teammate1 current native evaluation environment

Seed24, terrain seed42,100 first episodes. Population std (ddof=0). Teammate policy and reward modifications excluded. Current environment identity is source-confirmed; historical final reward execution log was unavailable.

## 100-env metrics

| Metric | Value |
|---|---:|
| episode_return mean ± std | 49.438447 ± 23.164547 |
| episode_steps mean ± std | 768.630000 ± 336.857259 |
| episode_seconds mean ± std | 12.810500 ± 5.614288 |
| forward_displacement mean ± std | 44.837190 ± 21.080600 |
| mean_forward_velocity mean ± std | 3.076907 ± 1.233464 |
| Return min/max | -0.388292 / 79.983515 |
| Fall / timeout | 32/100 / 68/100 |
| Observed world-X ≥5m | 87/100 |

## Stock weighted contributions

Each term is function×weight×control_dt. `energy` is a stock reward proxy, not literal physical energy.

| Component | Teammate1 mean ± std | FinalUnseen mean | Teammate2 mean |
|---|---:|---:|---:|
| progress | 44.838557 ± 21.081335 | 61.970421 | 46.800420 |
| alive | 6.402584 ± 2.810364 | 7.511667 | 7.616750 |
| upright | 1.203050 ± 0.556026 | 1.464467 | 1.369817 |
| move_to_target | 6.076421 ± 2.739446 | 7.396902 | 7.332467 |
| action_l2 | -0.184288 ± 0.406684 | -0.105305 | -0.151553 |
| energy | -5.867395 ± 2.668103 | -5.448845 | -7.082615 |
| joint_pos_limits | -3.030481 ± 1.269163 | -3.179251 | -3.265337 |
| total env.step | 49.438447 ± 23.164547 | 69.610059 | 52.619947 |

## Reward identity

Step max residual=1.7847924e-08; episode max/mean=2.52632071e-06/1.59367748e-06.
Maximum official float32 vs float64 accumulated difference=7.15521164e-05. All finite; no reward function/weight/manager changes.

## Same policy across three environments

| Environment | Return mean ± std | Displacement mean ± std (m) | Fall | Timeout | Other termination |
|---|---:|---:|---:|---:|---|
| My FinalUnseen | 69.610 ± 16.748 | 61.914 ± 14.977 (preterminal) |16%|84%|—|
| Teammate1 current host | 49.438 ± 23.165 | 44.837 ± 21.081 (terminal inclusive) | 32% | 68% | none configured |
| Teammate2 SelfEval | 52.620 ± 9.341 | 46.792 ± 8.744 (terminal inclusive) | 2% | 80% | boundary18% |

Matching preterminal displacement for teammate1: 44.785556m. Terminal-inclusive telemetry avoids auto-reset teleportation.
Final uses clearance-based fall; teammates use >90° orientation. Teammate2 has boundary truncation and a different spawn/terrain-reset rule. Teammate1 preserves startup friction randomization and has no map-boundary termination. Percentages therefore do not define an overall cross-environment robustness ranking.
In this host,32/100 runs ended by orientation fall and shorter episodes reduced alive accumulation. Accumulated progress is below the frozen Final value. Smaller accumulated energy proxy penalty than teammate2 cannot alone establish more efficient control because durations differ. Different terrain realizations and physics/reset conditions preclude a causal attribution.

## Video

[Seed24 env0 first episode](contact_teammate1_seed24_env0.mp4): 15.017000s,1280×720,60fps.
Single-env telemetry: return=58.044839, signed-X displacement=52.105164m, steps=901, fall=True, timeout=False.
Qualitative only, first episode without selection. Native vectorized randomization means n1 env0 is not n100 env0. Camera/material/light edits are visual only and physical mesh/material signatures are checked unchanged.

[Provenance](environment_provenance.md) · [Source mapping](source_mapping.json) · [Commands](commands.log) · [Integrity](integrity.json) · [Video manifest](video_manifest.json)
