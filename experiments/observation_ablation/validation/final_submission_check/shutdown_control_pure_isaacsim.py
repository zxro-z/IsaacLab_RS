"""Independent shutdown-isolation probe. No environment creation or explicit stepping."""
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
STEM = 'shutdown_control_pure_isaacsim'
IMPORTS = []
MECHANISM = 'simulationapp'
app = None

def event(kind, **fields):
    row = {"event": kind, "timestamp": datetime.now(timezone.utc).isoformat(), "monotonic": time.monotonic(), **fields}
    with (OUT / (STEM + "_events.jsonl")).open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row), flush=True)

event("process_started", python=sys.executable, cwd=os.getcwd(), conda_prefix=os.environ.get("CONDA_PREFIX"), LD_PRELOAD=os.environ.get("LD_PRELOAD"))
failed = False
try:
    if MECHANISM == "applauncher":
        from isaaclab.app import AppLauncher
        launcher = AppLauncher(headless=True)
        app = launcher.app
        experience = getattr(launcher, "_sim_experience_file", None)
        import isaaclab.app
        launcher_source = isaaclab.app.__file__
    else:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True})
        experience = None
        launcher_source = None
    event("initialization_completed", experience=experience, launcher_source=launcher_source, isaac_sim_location=os.environ.get("ISAAC_PATH"), kit_location=os.environ.get("CARB_APP_PATH"))
    for name in IMPORTS:
        module = importlib.import_module(name)
        event("import_pass", module=name, file=getattr(module, "__file__", None))
except BaseException:
    failed = True
    event("exception", traceback=traceback.format_exc())
finally:
    if app is not None:
        event("close_invoked")
        try:
            app.close()
            event("close_returned")
        except BaseException:
            failed = True
            event("close_exception", traceback=traceback.format_exc())
    event("probe_finished", status="FAIL" if failed else "PASS")
if failed:
    sys.exit(1)
