"""Headless import-only target-stack validation. No environment or physics stepping."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
report = {"python": sys.executable, "conda_prefix": os.environ.get("CONDA_PREFIX"), "isaaclab_root": str(ROOT), "framework_VERSION": (ROOT / "VERSION").read_text().strip(), "LD_PRELOAD": os.environ.get("LD_PRELOAD"), "imports": [], "environment_created": False, "physics_steps": 0, "checkpoints_loaded": False, "application_closed_cleanly": False}
app = None

def save_new(name, value):
    with (OUT / name).open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")

def libraries():
    p = Path("/proc/self/maps")
    return sorted({line.split()[-1] for line in p.read_text().splitlines() if "libstdc++.so" in line}) if p.exists() else []

try:
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    launcher = AppLauncher(parser.parse_args())
    app = launcher.app
    report["app_launcher"] = "PASS"
    report["isaac_sim_location"] = os.environ.get("ISAAC_PATH")
    report["kit_location"] = os.environ.get("CARB_APP_PATH")
    report["experience"] = getattr(launcher, "_sim_experience_file", None)
    sim = importlib.import_module("isaacsim")
    version_file = Path(sim.__file__).parent / "VERSION"
    report["isaac_sim_VERSION"] = version_file.read_text().strip() if version_file.exists() else None
    assert report["isaac_sim_VERSION"] and report["isaac_sim_VERSION"].startswith("5.1.0"), report["isaac_sim_VERSION"]
    for name in ["omni.log", "carb", "isaaclab", "isaaclab_tasks", "ant", "isaaclab_tasks.manager_based.classic.ant.ant_contact_observations", "isaaclab_tasks.manager_based.classic.ant.ant_terrain_heightscan_env_cfg"]:
        try:
            mod = importlib.import_module(name)
            report["imports"].append({"module": name, "status": "PASS", "file": getattr(mod, "__file__", None)})
            if name in ["isaaclab", "isaaclab_tasks", "ant"]:
                assert Path(mod.__file__).is_relative_to(ROOT), (name, mod.__file__)
        except BaseException:
            report["imports"].append({"module": name, "status": "FAIL", "traceback": traceback.format_exc()})
            raise
    report["task_registration_module_import"] = "PASS: ant/__init__.py executed; registry lookup deferred until clean shutdown passes"
    report["imports_status"] = "PASS"
except BaseException:
    report["status"] = "FAIL"
    report["traceback"] = traceback.format_exc()
    print(report["traceback"], flush=True)
finally:
    report["loaded_libstdc++"] = libraries()
    save_new("runtime_lerobot_arena_before_close.json", report)
    if app is not None:
        try:
            app.close()
            report["application_closed_cleanly"] = True
        except BaseException:
            report["close_traceback"] = traceback.format_exc()
            report["status"] = "FAIL"
    report["status"] = "PASS" if report.get("imports_status") == "PASS" and report["application_closed_cleanly"] else "FAIL"
    save_new("runtime_lerobot_arena_validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
if report["status"] != "PASS":
    sys.exit(1)
