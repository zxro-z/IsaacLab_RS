# Shutdown-isolation diagnosis

## Runtime and scope

- Conda: `lerobot-arena`; Python: `/home/zxro/miniforge3/envs/lerobot-arena/bin/python`.
- Isaac Sim installed package: 5.1.0.0; submission Isaac Lab: 2.3.0.
- No LD_PRELOAD, source changes, checkpoints loaded/modified, RL environments created, explicit physics steps, training, evaluation, Git initialization, commits or remote operations.
- Seven independent serial processes; startup/import timeout 120 seconds; external cutoff 30 seconds after close invocation.
- AppLauncher probes use `./isaaclab.sh -p <probe.py>` from the relevant framework root with source paths selected explicitly. Pure control uses the target Python directly with no PYTHONPATH override. Exact commands/environment paths are preserved in each JSON.

## Results

| Probe | Init/imports | close returned | Exit code | Shutdown result |
|---|---|---|---:|---|
| A — bare AppLauncher (submission) | PASS | No | 139 | SIGSEGV (shell status 128+11) |
| B — + isaaclab | PASS | No | -9 | 30 s timeout; external SIGKILL |
| C — + isaaclab_tasks | PASS | No | 139 | SIGSEGV (shell status 128+11) |
| D — + ant | PASS | No | 134 | SIGABRT (shell status 128+6) |
| E — + Contact / HeightScan helpers | PASS | No | -9 | 30 s timeout; external SIGKILL |
| Control — bare AppLauncher (original framework) | PASS | No | 134 | SIGABRT (shell status 128+6) |
| Control — pure SimulationApp | PASS | No | -11 | SIGSEGV |

`-9` / `-11` are Python subprocess signal return codes. Shell-wrapped exits `139` and `134` correspond to SIGSEGV and SIGABRT respectively; console traces confirm the signals. Forced termination is never a clean shutdown. Every close timestamp, wall time, console tail and command is in the individual record.

## Classification

**INCONCLUSIVE for the exact hang mechanism.** Bare A failed first with SIGSEGV; B first timed out. Original-framework bare AppLauncher aborted; pure Isaac Sim segfaulted. Thus the probes confirm a **base runtime shutdown failure**, but they did not reproduce identical bare hangs in both roots. Outcomes vary between hangs and crashes, so the requested BASE RUNTIME HANG label would be too specific.

The shutdown failure is not confined to the submission overlay, isaaclab_tasks, ant or custom helpers: it occurs without those project imports and with pure SimulationApp. This does not identify the underlying native-library/Kit cause or exclude additional issues in project code. All requested explicit imports completed before close. AppLauncher necessarily imports isaaclab.app; B adds an explicit isaaclab import rather than isolating an entirely new package lifecycle.

AppLauncher uses the root-specific `apps/isaaclab.python.headless.kit`; pure SimulationApp uses its default `isaacsim.exp.base.python.kit`. No extension settings or Kit configuration were changed. The pure control is a baseline failure control, not an identical experience comparison. Crash traces locate the calling thread in `simulation_app.py:838` inside close; this alone is not a root-cause diagnosis.

## Proposed next validation (not executed)

A separately approved, bounded one-env observation-shape diagnostic is reasonable because baseline shutdown already fails without project imports. Create one environment at a time, flush/fsync config, import origins and observed tensor dimensions before normal close, then let an external watchdog terminate the process only after the evidence is recorded if close stalls. Record crashes and forced termination as failures of the runtime lifecycle, never as a clean validation pass. Do not alter source, skip close, call os._exit, retrain or replace canonical evaluation outputs.

The user’s exact conditional gate (identical bare hangs in both roots) was not met. Therefore this is a proposal requiring the next approval, not authorization inferred from the diagnostic. No one-env test was run. A full 100-env evaluation is not yet validated or approved.

## Preservation

All three originals retain identical HEAD, branch, status and remotes against the pre-probe snapshot. The original project inventory (873 entries) and framework inventory (2975 entries) retain recorded file hashes and mtimes. The old excluded repository was not used and its Git state is unchanged; no full content inventory of that old repository is claimed. See `shutdown_preservation_validation.json`.

Existing diagnostics and canonical artifacts were not overwritten. Newly created files are limited to these shutdown probe scripts, event records, console logs and reports in this validation directory.
