"""Run loaded-policy smoke gates and common Stock Reward evaluations sequentially.

The verify step must already have produced reward_config_verification.json.
Logs/commands are append-only, result directories must be new. No training.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/observation_ablation/common_stock_reward_evaluation"
POLICIES = ("heightscan_stock", "heightscan_contact_stock", "heightscan_contact_modified_trained")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("smoke", "evaluate"))
args = parser.parse_args()
assert json.loads((OUTPUT / "reward_config_verification.json").read_text())["pass_"]
runtime = os.environ.copy()
for key in ("LD_PRELOAD", "ISAAC_SIM_SITE_PACKAGES", "ISAAC_PATH", "CARB_APP_PATH", "EXP_PATH"):
    runtime.pop(key, None)
runtime.update(ISAACLAB_RS=str(ROOT), ISAACLAB_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=":".join(str(ROOT / "source" / p) for p in ("", "isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic")),
               MPLCONFIGDIR="/tmp/ant_common_stock_matplotlib", XDG_CACHE_HOME="/tmp/ant_common_stock_cache")
with (OUTPUT / "commands.log").open("a") as stream:
    stream.write("unset LD_PRELOAD ISAAC_SIM_SITE_PACKAGES ISAAC_PATH CARB_APP_PATH EXP_PATH\n")
    for key in ("ISAACLAB_RS", "ISAACLAB_ROOT", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        stream.write("export " + key + "=" + shlex.quote(runtime[key]) + "\n")
for policy in POLICIES:
    destination = OUTPUT / "smoke" / policy if args.mode == "smoke" else OUTPUT / policy
    assert not destination.exists(), destination
    log = OUTPUT / (args.mode + "_" + policy + ".log")
    assert not log.exists(), log
    cmd = [sys.executable, "-u", "scripts/evaluate_common_stock_reward.py", args.mode, "--policy", policy,
           "--headless", "--output", str(destination.relative_to(ROOT)), "--kit_args",
           "--portable-root=/tmp/ant_common_stock_kit --/app/settings/persistent=false"]
    with (OUTPUT / "commands.log").open("a") as stream:
        stream.write(shlex.join(cmd) + "\n")
    print("Starting", args.mode, policy, flush=True)
    with log.open("w") as stream:
        process = subprocess.run(cmd, cwd=ROOT, env=runtime, stdout=stream, stderr=subprocess.STDOUT)
    print("Exit", process.returncode, "log:", log.relative_to(ROOT), flush=True)
    if process.returncode:
        print(log.read_text()[-6000:], flush=True)
        raise SystemExit(process.returncode)
