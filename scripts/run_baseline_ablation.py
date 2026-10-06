"""Execute one Baseline experiment phase with reproducible environment/logging."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "experiments/observation_ablation/baseline_4096x32x1000"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("verify", "smoke", "train", "verify_eval", "smoke_eval", "evaluate"))
args = parser.parse_args()
env = os.environ.copy()
for key in ("LD_PRELOAD", "ISAAC_SIM_SITE_PACKAGES", "ISAAC_PATH", "CARB_APP_PATH", "EXP_PATH"):
    env.pop(key, None)
env.update(ISAACLAB_RS=str(ROOT), ISAACLAB_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=":".join(str(ROOT/"source"/p) for p in ("", "isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic")),
           MPLCONFIGDIR="/tmp/ant_baseline_matplotlib", XDG_CACHE_HOME="/tmp/ant_baseline_cache")
common = ROOT / "experiments/observation_ablation/common_stock_reward_evaluation"
if args.mode in ("verify", "smoke", "train"):
    output = {"verify": ARTIFACT, "smoke": ARTIFACT/"smoke", "train": ROOT/"logs/rsl_rl/observation_ablation/ablation_baseline_stock_s42_e4096_n32_i1000"}[args.mode]
    cmd = [sys.executable, "-u", "scripts/baseline_observation_ablation_budget.py", args.mode, "--output", str(output.relative_to(ROOT))]
else:
    mode = {"verify_eval": "verify", "smoke_eval": "smoke", "evaluate": "evaluate"}[args.mode]
    output = {"verify_eval": ARTIFACT/"evaluation_config_verification", "smoke_eval": common/"smoke/baseline", "evaluate": common/"baseline"}[args.mode]
    cmd = [sys.executable, "-u", "scripts/evaluate_common_stock_reward.py", mode, "--output", str(output.relative_to(ROOT)),
           "--verification-record", str((ARTIFACT/"evaluation_config_verification/reward_config_verification.json").relative_to(ROOT))]
    if mode != "verify":
        cmd += ["--policy", "baseline"]
cmd += ["--headless", "--kit_args", "--portable-root=/tmp/ant_baseline_kit --/app/settings/persistent=false"]
log = ARTIFACT / (args.mode + "_stdout.log")
assert not log.exists(), log
with (ARTIFACT/"commands.log").open("a") as stream:
    stream.write("unset LD_PRELOAD ISAAC_SIM_SITE_PACKAGES ISAAC_PATH CARB_APP_PATH EXP_PATH\n")
    for key in ("ISAACLAB_RS", "ISAACLAB_ROOT", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        stream.write("export " + key + "=" + shlex.quote(env[key]) + "\n")
    stream.write(shlex.join(cmd) + "\n")
print("Starting", args.mode, flush=True)
with log.open("w") as stream:
    process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
print("Exit", process.returncode, "log:", log.relative_to(ROOT), flush=True)
if process.returncode:
    print(log.read_text()[-6000:], flush=True)
raise SystemExit(process.returncode)
