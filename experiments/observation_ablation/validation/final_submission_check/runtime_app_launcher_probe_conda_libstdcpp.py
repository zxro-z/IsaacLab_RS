"""Non-stepping submission import/config/checkpoint probe; never creates an environment."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
runtime = {"python": sys.executable, "conda_prefix": os.environ.get("CONDA_PREFIX"), "isaac_lab_root": str(ROOT), "framework_version": (ROOT / "VERSION").read_text().strip(), "sim_site_packages": os.environ.get("ISAAC_SIM_SITE_PACKAGES"), "imports": [], "environment_created": False, "physics_steps": 0, "rollout": False, "application_closed": False}
app = None

def write_new(name, data):
    with (OUT / name).open("x") as file:
        json.dump(data, file, indent=2)
        file.write("\n")

def checked_import(name):
    try:
        module = importlib.import_module(name)
        runtime["imports"].append({"module": name, "status": "PASS", "file": getattr(module, "__file__", None)})
        return module
    except BaseException:
        runtime["imports"].append({"module": name, "status": "FAIL", "traceback": traceback.format_exc()})
        raise

try:
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    runtime["app_launcher"] = "PASS"
    runtime["experience"] = getattr(launcher, "_sim_experience_file", None)
    runtime["isaac_sim_location"] = os.environ.get("ISAAC_PATH")
    runtime["kit_location"] = os.environ.get("CARB_APP_PATH")
    sim_root = Path(os.environ.get("ISAAC_PATH", ""))
    if (sim_root / "VERSION").is_file():
        runtime["isaac_sim_version"] = (sim_root / "VERSION").read_text().strip()
    for name in ["omni.log", "isaaclab", "isaaclab_tasks", "ant", "isaaclab_tasks.manager_based.classic.ant.ant_contact_observations", "isaaclab_tasks.manager_based.classic.ant.ant_terrain_heightscan_env_cfg"]:
        checked_import(name)
    # ant/__init__.py is the project task registration module and has already run.
    runtime["task_registration_module"] = "ant.__init__"
    runtime["canonical_evaluator_import"] = {"status": "NOT RUN", "reason": "Each evaluator unconditionally launches AppLauncher and calls main() at module scope; importing it would create an environment, forbidden by this probe."}
    import gymnasium as gym
    ids = ["Ant-rl-v0", "Ant-rl-Ablation-HeightScan-v0", "Ant-rl-Ablation-HeightScan-Contact-Stock-v0", "Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0"]
    registrations = []
    for task in ids:
        spec = gym.spec(task)
        registrations.append({"task": task, "entry_point": spec.entry_point, "kwargs": spec.kwargs})
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    write_new("task_registration_validation_conda_libstdcpp.json", {"status": "PASS", "tasks": registrations, "environment_created": False})
    config_types = [
        ("heightscan_4096x32x1000", "ant.ablation_env_cfg", "AblationHeightScanCfg", 122),
        ("heightscan_contact_stock_4096x32x1000", "ant.contact_stock_env_cfg", "HeightScanContactStockCfg", 126),
        ("heightscan_contact_modified_4096x32x1000", "ant.stage2_env_cfg", "Stage2HeightScanContactCfg", 126),
    ]
    configs = []
    for experiment, module_name, class_name, expected in config_types:
        cls = getattr(checked_import(module_name), class_name)
        cfg = cls()
        terms = [name for name, term in vars(cfg.observations.policy).items() if hasattr(term, "func")]
        assert "height_scan" in terms
        assert ("foot_contacts" in terms) == (expected == 126)
        saved = json.loads((ROOT / "experiments/observation_ablation" / experiment / "results/config.json").read_text())
        saved_terms = [name for name, term in saved["observations"]["policy"].items() if isinstance(term, dict) and "func" in term]
        assert terms == saved_terms, (terms, saved_terms)
        configs.append({"experiment": experiment, "config_class": module_name + ":" + class_name, "status": "PASS", "policy_terms": terms, "matches_preserved_term_order": True, "expected_interface": expected, "runtime_observation_tensor_dimension": None, "dimension_note": "Exact environment-produced observation tensor not measured because no environment is instantiated.", "scene_num_envs_config_default": cfg.scene.num_envs})
    write_new("runtime_config_load_validation_conda_libstdcpp.json", {"status": "PASS", "configs": configs, "environment_created": False})
    import torch
    checkpoints = []
    for experiment, _, _, expected in config_types:
        directory = ROOT / "experiments/observation_ablation" / experiment
        summary = json.loads((directory / "training_summary.json").read_text())
        path = directory / "checkpoints/best_model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"]
        actor = state["actor.0.weight"]
        critic = state["critic.0.weight"]
        assert tuple(actor.shape) == (400, expected), tuple(actor.shape)
        assert tuple(critic.shape) == (400, expected), tuple(critic.shape)
        nonfinite = [name for name, tensor in state.items() if isinstance(tensor, torch.Tensor) and (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all().item()]
        assert not nonfinite, nonfinite
        checkpoints.append({"experiment": experiment, "checkpoint": str(path.relative_to(ROOT)), "serialization": "PASS", "weights_only": True, "actor_first_layer": list(actor.shape), "critic_first_layer": list(critic.shape), "actual_actor_input_dim": int(actor.shape[1]), "matches_config_snapshot_and_expected_interface": True, "checkpoint_iteration_field": checkpoint.get("iter"), "selected_iteration": summary["selected_iteration"], "model_tensors_finite": True})
    write_new("checkpoint_interface_validation_conda_libstdcpp.json", {"status": "PASS", "checkpoints": checkpoints, "inference_performed": False})
    runtime["status"] = "PASS imports/configs/checkpoints; canonical evaluator passive import intentionally excluded"
except BaseException:
    runtime["status"] = "FAIL; stopped without source changes"
    runtime["traceback"] = traceback.format_exc()
    runtime.setdefault("app_launcher", "FAIL")
    print(runtime["traceback"], flush=True)
finally:
    if app is not None:
        try:
            app.close()
            runtime["application_closed"] = True
        except BaseException:
            runtime["application_close_traceback"] = traceback.format_exc()
    write_new("runtime_app_launcher_validation_conda_libstdcpp.json", runtime)
    print(json.dumps(runtime, indent=2), flush=True)
if runtime.get("status", "").startswith("FAIL"):
    sys.exit(1)
