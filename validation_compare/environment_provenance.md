# Teammate 1 current evaluation environment

## Evidence and unresolved history
`source/ant/__init__.py:18-26` registers only Ant-rl-v0. `command.txt` PLAY section and `project.md` sections3,9,10 identify play.py + Ant-rl-v0. Current terrain and termination match saved `logs/rsl_rl/ant/v3_depth/params/env.yaml`; v2 saved config uses a different 8×8 random-grid terrain. No final reward-output log/result artifact was found. Exact historical final seed, num_envs and executed checkpoint command are therefore unknown. This measurement uses the confirmed current native evaluation host and does not claim to reproduce an unrecorded historical reward result.

Documented original command template (not executed here):
```bash
python scripts/rsl_rl/play.py --task Ant-rl-v0 --num_envs 4 --checkpoint logs/rsl_rl/ant/RUN_FOLDER/best_model.pt
```

## Geometry, physics and reset
- ROUGH_TERRAINS_CFG copy with native overrides (`source/ant/ant_env_cfg.py:59-96`), generator seed42, no curriculum, difficulty range[0,1].
- 20×10 tiles of10×10m; border2m; horizontal scale0.1m, vertical scale0.005m, slope threshold0.75, cache false.
- Each active subterrain proportion0.2: pyramid_stairs, pyramid_stairs_inv, boxes, hf_pyramid_slope, hf_pyramid_slope_inv. random_rough proportion0.
- Stairs height0.03–0.07m,width0.3m; boxes grid0.45m,height0.02–0.10m; slopes0–0.20; central platforms1m; native holes/border settings retained.
- Ground static/dynamic friction1, restitution0; both combine modes multiply.
- Robot static friction uniform[0.3,1.0] at startup, dynamic0.8×static, restitution0; unchanged across resets. Per-run material arrays saved.
- Stock host noninstanceable Ant asset, joint ordering and effort action scale7.5 retained.
- Root reset extra pose/velocity randomization empty; joint position offset[-0.2,0.2], velocity[-0.1,0.1]. Native importer terrain origins/assignment retained, no teammate2 reselect event.
- Physics dt1/120s, decimation2, control dt1/60s; 16s/960steps timeout.
- Fall only body_z_down, bad_orientation(limit_angle=pi/2); torso-height condition disabled; no boundary exit termination.

## Reward separation
Training RewardManager has one total_reward term whose custom function internally combines10 components. Native play.py overrides it with6 EvalRewardsCfg terms; it omits joint_pos_limits. Neither is used for Contact measurement. The local task explicitly instantiates RS stock RewardsCfg, unchanged weights/functions, with seven terms. Contributions are the existing manager._step_reward × control_dt (recovering its weighted buffer); no reward functions are recomputed. Env.step total is authoritative. Terminal step included, subsequent reset episodes excluded; float64 trace plus official float32 accumulation are recorded.

## Observation separation
The original59-D proprioception/depth group and depth camera are not used. The Contact checkpoint receives native RS60-D stock observation +63 RayCaster heights +4 binary foot contacts, validated127/127. Exactly one scanner and one feet sensor are present. Host asset/terrain/action/events/termination/sim config equality is asserted before launch.

## Limits
- Historical final reward execution provenance is incomplete; this is the verified current native evaluation environment, not a proven replay of an unknown historical final run.
- Teammate depth perception is replaced with checkpoint-required RS HeightScan/Contact. Geometry and physics definitions are retained; observation/camera differences are intentional.
- Host noninstanceable ant.usd is retained; RS checkpoint trained with stock instanceable Ant. Runtime joint order and effort action scale7.5 are saved.
- No terrain-relative height fall check: fall is body_z_down (>90deg). Different from FinalUnseen clearance<0.31m.
- Startup robot friction is random static[0.3,1.0], dynamic0.8*static; kept unchanged. Teammate2 uses different material/reset/map-boundary definitions.
- World-X displacement is terminal-inclusive here; original Final uses preterminal pose. The matching legacy metric is retained.
- No map-boundary termination is configured; travel can leave the initial 10m tile and cross other terrain tiles/border.
- Generator seed alone does not guarantee identical historical meshes across runtime/Torch RNG differences; per-run mesh/origin/material data is saved.
