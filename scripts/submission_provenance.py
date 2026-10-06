"""Manifest-only provenance for Git checkouts and preserved submission snapshots."""
import json
import os
from pathlib import Path
import re
import subprocess


def manifest_git_provenance(root, *, allow_snapshot=False):
    """Keep Git HEAD when available; use recorded provenance for evaluation snapshots."""
    root = Path(root)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode == 0:
        return dict(git_commit=result.stdout.strip(),
                    git_commit_source="git_rev_parse_HEAD", repository_snapshot=False)
    unavailable = "not a git repository" in result.stderr and not (root / ".git").exists()
    if not allow_snapshot or not unavailable:
        raise subprocess.CalledProcessError(result.returncode, result.args,
                                            output=result.stdout, stderr=result.stderr)
    record = json.loads((root / "docs/submission_provenance/copy_manifest.json").read_text())
    state = record["original_states_before"]["/home/zxro/teammate_ant_rl"]
    commit = state["head"].strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid preserved project HEAD in submission provenance")
    return dict(git_commit=commit, git_commit_source="preserved_submission_provenance",
                repository_snapshot=True)
