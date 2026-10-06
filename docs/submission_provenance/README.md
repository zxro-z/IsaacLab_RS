# Submission copy provenance

The original Phase 2 copy inventory is preserved unchanged. This document describes the source tiers and the narrowly approved submission compatibility changes; the root README describes the verified final runtime.

| Tier | Source | Verification |
|---|---|---|
| A — Official framework | Isaac Lab official v2.3.0 committed tree; commit 3c6e67bb5c7ada942a6d1884ab69338f57596f77 | Exported with git archive from /home/zxro/arena/IsaacLab; working-tree changes excluded |
| B — Current project source | /home/zxro/teammate_ant_rl, observation-ablation, HEAD 4513d7cae5d4fdc6af9288349531e6b49422f7b1 | Copied bytes verified; canonical results/logs preserved |
| C — Directly provenance-verified custom runtime files | ant_contact_observations.py; ant_terrain_heightscan_env_cfg.py | SHA256 matches canonical experiment source mappings |
| D — Transitive runtime dependencies | ant_friction_random_env_cfg.py; ant_ood_terrain_env_cfg.py | Recovered from the same validated runtime checkout; no direct experiment source hash was recorded |

All four custom files came from /home/zxro/arena/IsaacLab, not from the old /home/zxro/IsaacLab_RS repository. Their exact hashes, copy paths, and protected original inventories are recorded in copy_manifest.json.

The current project namespace source/ant remains separate from the framework Ant namespace. The RL script group was copied to scripts/reinforcement_learning/rsl_rl without changing file contents. Dedicated evaluators remain under scripts/. Historical command logs and manifests retain their original paths unchanged.

The HeightScan transfer evaluation in validation_compare remains supplementary validation rather than a canonical training experiment.

The old /home/zxro/IsaacLab_RS repository is EXCLUDED AS A SUBMISSION SOURCE. Its contents were not used or modified. New submission validation has reproduced all three canonical HeightScan evaluations; historical evidence remains unchanged.

## Approved submission changes

- Evaluator Git metadata fallback only: `scripts/submission_provenance.py` and four manifest call sites. Git checkouts retain their current HEAD; no-Git evaluation snapshots identify preserved project provenance explicitly.
- Modified evaluator config paths only: two source-hash-verified YAMLs copied from `logs/rsl_rl/ant/v3_depth/params/` to `logs/rsl_rl/ant/modified/params/`. See [mapping](modified_config_mapping.json).
- Submission documentation and generated-junk ignore rules. Task/observation/reward/terrain/PPO/rollout/metrics logic is unchanged.

Historical source mappings still identify the original script hashes and paths. Compare approved patch records when verifying the current evaluator hashes; do not rewrite those historical records to hide layout changes.

The upstream `.gitattributes` is preserved in [official_gitattributes](official_gitattributes). Submission LFS filter declarations were removed so required checkpoints/videos are staged as ordinary Git blobs; no LFS tooling or remote was configured.

## HeightScan submission cleanup

The current project is a progressive HeightScan ablation. See [cleanup audit](heightscan_cleanup/README.md). The original copy inventory, integrity manifests, protected-file lists, source mappings, original README snapshot, validation reports and saved training configurations remain byte-for-byte historical evidence. They can name intentionally removed files and contain obsolete sensor interfaces. Their recorded hashes and PASS statuses apply to the earlier snapshot, not the current repository. Saved `logs/rsl_rl/ant/modified/params/env.yaml` supplies reward provenance to the final HeightScan evaluator; its historical observation/sensor fields are not instantiated by that evaluator.
