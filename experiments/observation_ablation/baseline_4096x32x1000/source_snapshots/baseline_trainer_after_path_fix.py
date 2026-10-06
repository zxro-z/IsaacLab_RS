"""Fresh 59-D project Baseline; same canonical PPO/budget/selection semantics.

Verify and smoke must pass before training. No existing output is overwritten.
Common-objective evaluation uses evaluate_common_stock_reward.py separately.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "experiments/observation_ablation/baseline_4096x32x1000"
sys.path.insert(0, str(ROOT / "source"))
for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
    sys.path.insert(0, str(ROOT / "source" / package))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("verify", "smoke", "train"))
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output = args.output.resolve()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import ant
from ant.ablation_env_cfg import AblationBaseCfg, AblationHeightScanCfg
from ant.agents.ablation_budget_ppo_cfg import AblationBaselinePPORunnerCfg, AblationBudgetPPORunnerCfg
from ant.contact_stock_env_cfg import HeightScanContactStockCfg
from ant.stage2_env_cfg import Stage2HeightScanContactCfg
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg

TASK = "Ant-rl-Ablation-Baseline-v0"
SELECTION = "best_model: highest logged completed-episode training mean return (last 100 completed episodes); no evaluation-based selection"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize(value):
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    raise TypeError(type(value))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serialize, allow_nan=False).encode()).hexdigest()


def dump(directory, name, value):
    path = directory / name
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2, default=serialize, allow_nan=False) + "\n")


class BestModelRunner(OnPolicyRunner):
    """Same selection hook as retained Stock/Contact runs, with full CSV evidence."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.best_mean_reward = float("-inf")

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        mean = statistics.mean(locs["rewbuffer"]) if locs["rewbuffer"] else None
        with (Path(self.log_dir) / "training_curve.csv").open("a", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            if stream.tell() == 0:
                writer.writerow(["iteration", "mean_reward", "completed_episode_buffer_count"])
            writer.writerow([locs["it"], mean, len(locs["rewbuffer"])])
        if mean is not None and mean > self.best_mean_reward:
            self.best_mean_reward = mean
            self.save(str(Path(self.log_dir) / "best_model.pt"), infos={"mean_reward": mean})


def main():
    cfg, agent = AblationBaseCfg(), AblationBaselinePPORunnerCfg()
    cfg.seed, cfg.scene.num_envs = 42, 4096
    if args.device:
        cfg.sim.device = agent.device = args.device
    assert gym.spec(TASK).kwargs["env_cfg_entry_point"] == "ant.ablation_env_cfg:AblationBaseCfg"
    assert gym.spec(TASK).kwargs["rsl_rl_cfg_entry_point"].endswith(":AblationBaselinePPORunnerCfg")
    assert cfg.rewards.to_dict() == RewardsCfg().to_dict()
    assert agent.seed == 42 and agent.num_steps_per_env == 32 and agent.max_iterations == 1000
    assert not agent.resume and agent.load_run is None and agent.load_checkpoint is None and agent.save_interval == 50
    assert not agent.policy.actor_obs_normalization and not agent.policy.critic_obs_normalization
    assert cfg.scene.terrain.terrain_generator.seed == 42 and not cfg.scene.terrain.terrain_generator.curriculum
    assert cfg.curriculum is None and cfg.episode_length_s == 16 and cfg.decimation == 2 and cfg.sim.dt == 1/120
    assert getattr(cfg.scene, "height_scanner", None) is None and getattr(cfg.scene, "feet_contacts", None) is None
    policy = cfg.observations.policy.to_dict()
    assert "feet_body_forces" in policy and not any(any(word in k.lower() for word in ("height_scan", "foot_contacts", "camera", "depth")) for k in policy)
    retained = []
    for folder, env_cls, dim in (("heightscan_4096x32x1000", AblationHeightScanCfg, 122),
                                 ("heightscan_contact_stock_4096x32x1000", HeightScanContactStockCfg, 126),
                                 ("heightscan_contact_modified_4096x32x1000", Stage2HeightScanContactCfg, 126)):
        path = ROOT / "experiments/observation_ablation" / folder
        summary = json.loads((path / "training_summary.json").read_text())
        assert summary["iterations"] == 1000 and summary["num_steps_per_env"] == 32 and summary["actual_total_transitions"] == 131072000
        old = env_cls(); old.seed = 42; old.scene.num_envs = 4096
        old.sim.device = cfg.sim.device
        old_cfg = old.to_dict(); baseline_cfg = cfg.to_dict()
        old_cfg["rewards"] = baseline_cfg["rewards"]
        old_cfg["scene"].pop("height_scanner")
        old_cfg["scene"].pop("feet_contacts", None)
        old_cfg["observations"]["policy"].pop("height_scan")
        old_cfg["observations"]["policy"].pop("foot_contacts", None)
        assert old_cfg == baseline_cfg, folder
        cp = path / "checkpoints/best_model.pt"
        weights = torch.load(cp, map_location="cpu", weights_only=False)["model_state_dict"]
        assert weights["actor.0.weight"].shape[1] == dim
        run = ROOT / "logs/rsl_rl/observation_ablation" / ("ablation_" + folder.replace("_4096x32x1000", "") + ("_stock" if folder.startswith("heightscan_4096") else "") + "_s42_e4096_n32_i1000")
        saved_agent = json.loads((run / "agent.json").read_text())
        for key in ("seed", "num_steps_per_env", "max_iterations", "save_interval", "policy", "algorithm", "obs_groups"):
            assert saved_agent[key] == agent.to_dict()[key], (folder, key)
        saved_env = json.loads((run / "config.json").read_text())
        for key in ("seed", "sim", "actions", "events", "terminations", "episode_length_s", "decimation", "curriculum"):
            assert saved_env[key] == json.loads(json.dumps(cfg.to_dict()[key], default=serialize)), (folder, key)
        assert saved_env["scene"]["num_envs"] == 4096 and saved_env["scene"]["terrain"] == json.loads(json.dumps(cfg.scene.terrain.to_dict(), default=serialize))
        retained.append(dict(directory=folder, observation_dimension=dim, checkpoint_sha256=sha(cp), environment_parity_except_observation_sensors_and_training_reward=True, saved_ppo_and_environment_parity=True))
    agent_reference = AblationBudgetPPORunnerCfg().to_dict()
    agent_baseline = agent.to_dict()
    agent_reference.pop("run_name"); agent_baseline.pop("run_name")
    assert agent_reference == agent_baseline
    record = dict(pass_=True, task=TASK, observation_dimension=59, action_dimension=8,
                  num_envs=4096, rollout_steps=32, iterations=1000, total_transitions=131072000,
                  training_seed=42, terrain_seed=42, curriculum=False, fixed_initial_terrain_assignment=True,
                  assignment_basis="No curriculum manager; native reset does not call terrain origin/assignment updates",
                  reward="Stock", reward_config=cfg.rewards.to_dict(), ppo=agent.to_dict(),
                  environment=cfg.to_dict(), environment_sha256=digest(cfg.to_dict()), ppo_sha256=digest(agent.to_dict()),
                  retained_parity=retained, fresh_initialization=True, init_at_random_ep_len=True,
                  checkpoint_selection=SELECTION, source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/'source/ant/ablation_env_cfg.py',ROOT/'source/ant/ant_env_cfg.py',ROOT/'source/ant/agents/ablation_budget_ppo_cfg.py']})
    if args.mode == "verify":
        args.output.mkdir(parents=True, exist_ok=True)
        dump(args.output, "pretraining_verification.json", record)
        print("PRETRAINING CONFIG PASS: 59-D native Baseline, identical environment/PPO/budget", flush=True)
        return
    assert json.loads((ARTIFACT / "pretraining_verification.json").read_text())["source_sha256"] == record["source_sha256"]
    if args.mode == "train":
        assert json.loads((ARTIFACT / "smoke/smoke.json").read_text())["pass"]
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    if args.mode == "smoke":
        cfg.scene.num_envs = 4
    cfg.log_dir = str(args.output)
    dump(args.output, "config.json", cfg.to_dict())
    dump(args.output, "agent.json", agent.to_dict())
    dump(args.output, "manifest.json", dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), base_git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
         working_tree_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True), clean_committed_tree=False,
         working_tree_diff_sha256=hashlib.sha256(subprocess.check_output(["git","diff","--binary"],cwd=ROOT)).hexdigest(),
         python=sys.version, isaaclab=(ROOT/'VERSION').read_text().strip(), isaacsim=importlib.metadata.version('isaacsim'),
         pytorch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), protocol=record))
    env = RslRlVecEnvWrapper(gym.make(TASK,cfg=cfg),clip_actions=agent.clip_actions)
    obs = env.get_observations(); base=env.unwrapped
    assert obs["policy"].shape == (env.num_envs,59) and env.num_actions == 8
    assert "height_scanner" not in base.scene.sensors and "feet_contacts" not in base.scene.sensors
    assert base.reward_manager.active_terms == ["progress","alive","upright","move_to_target","action_l2","energy","joint_pos_limits"]
    if args.mode == "smoke":
        with torch.inference_mode():
            for _ in range(32):
                obs,reward,_,_=env.step(torch.zeros((env.num_envs,8),device=base.device))
                assert obs["policy"].shape==(4,59) and torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
        dump(args.output,"smoke.json",dict(**{"pass":True},task=TASK,observation_dimension=59,action_dimension=8,steps=32,finite=True,stock_reward=True,heightscan=False,binary_contact=False,metrics_are_experiment_results=False))
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        runner=BestModelRunner(env,agent.to_dict(),log_dir=str(args.output),device=agent.device)
        started=time.perf_counter(); timestamp=datetime.now(timezone.utc).isoformat()
        runner.learn(num_learning_iterations=1000,init_at_random_ep_len=True)
        assert runner.current_learning_iteration==999 and runner.tot_timesteps==131072000
        selected=args.output/'best_model.pt'; final=args.output/'model_999.pt'
        saved=torch.load(selected,map_location='cpu',weights_only=False)
        with (args.output/'training_curve.csv').open() as stream: curve=list(csv.DictReader(stream))
        assert len(curve)==1000
        best=max((r for r in curve if r['mean_reward']),key=lambda r:float(r['mean_reward']))
        assert int(best['iteration'])==saved['iter'] and float(best['mean_reward'])==saved['infos']['mean_reward']
        checkpoints=ARTIFACT/'checkpoints';assert not checkpoints.exists();checkpoints.mkdir()
        shutil.copyfile(selected,checkpoints/'best_model.pt');shutil.copyfile(final,checkpoints/'final_model.pt')
        assert sha(selected)==sha(checkpoints/'best_model.pt') and sha(final)==sha(checkpoints/'final_model.pt')
        dump(ARTIFACT,'checkpoint_selection.json',dict(rule=SELECTION,selected_iteration=saved['iter'],selected_training_mean_reward=saved['infos']['mean_reward'],curve=str((args.output/'training_curve.csv').relative_to(ROOT)),evaluation_not_used=True,checkpoint_sha256=sha(selected)))
        dump(ARTIFACT,'training_summary.json',dict(completed=True,iterations=1000,num_steps_per_env=32,total_transitions=131072000,actual_total_transitions=runner.tot_timesteps,
             actual_completed_iterations=1000,resume=False,training_seed=42,terrain_seed=42,start_timestamp=timestamp,end_timestamp=datetime.now(timezone.utc).isoformat(),
             wall_time_seconds=time.perf_counter()-started,run_path=str(args.output.relative_to(ROOT)),selected_iteration=saved['iter'],selected_training_mean_reward=saved['infos']['mean_reward'],
             selected_checkpoint=str((checkpoints/'best_model.pt').relative_to(ROOT)),checkpoint_sha256=sha(selected),final_checkpoint=str((checkpoints/'final_model.pt').relative_to(ROOT)),final_checkpoint_sha256=sha(final)))
    env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:
    app.close()
