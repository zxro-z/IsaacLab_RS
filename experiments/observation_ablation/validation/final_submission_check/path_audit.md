# Phase 3 path audit

The line-level inventory is in path_audit.json. Historical evidence is unchanged.

| Classification | Treatment |
|---|---|
| A — Historical provenance | Original command logs, manifests, source mappings, raw logs and archived documentation retain their paths |
| B — Active executable defaults/references | No source edits; supported ISAACLAB_RS/ISAACLAB_ROOT overrides select the submission copy. Hard-coded supplementary/finalizer paths remain limitations |
| C — README/documentation | Submission README uses relative links and portable setup variables. Original documentation archives retain provenance |
| D — Invalid/broken | Missing DepthCam checkpoint paths are documented as unavailable, not fabricated. Historical paths to excluded training runs remain archive references |

## Active executable references

| File / line | Treatment |
|---|---|
| `cross_eval_team1_policy/evaluate_depth_policy.py:5` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `cross_eval_team1_policy/run_sequence.py:3` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `cross_eval_team1_policy/finalize_results.py:3` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `cross_eval_team1_policy/finalize_results.py:70` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `cross_eval_team1_policy/evaluate_depth_policy_initial.py:5` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `cross_eval_team1_policy/evaluate_depth_policy_smoke_version.py:5` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `validation_compare/postprocess_source.py:3` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `validation_compare/postprocess_source.py:41` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `validation_compare/postprocess_source.py:43` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `validation_compare/postprocess_source.py:64` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `validation_compare/postprocess_source.py:70` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `validation_compare/postprocess_source.py:81` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `scripts/evaluate_depthcam_ablation.py:13` | Leave unchanged; set ISAACLAB_ROOT to the submission root. |
| `scripts/finalize_observation_ablation.py:197` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `scripts/stage2_observation_ablation.py:27` | Leave source unchanged; set ISAACLAB_RS to the submission root. |
| `scripts/finalize_stage2_observation_ablation.py:77` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `scripts/contact_stock_observation_ablation.py:20` | Leave source unchanged; set ISAACLAB_RS to the submission root. |
| `scripts/finalize_observation_ablation_budget.py:64` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `scripts/observation_ablation_budget.py:19` | Leave source unchanged; set ISAACLAB_RS to the submission root. |
| `scripts/observation_ablation.py:18` | Leave source unchanged; set ISAACLAB_RS to the submission root. |
| `scripts/evaluate_contact_teammate1.py:5` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |
| `scripts/evaluate_contact_teammate1.py:77` | Not portable as-is; historical finalizer/supplementary executable requires review before use. |

## Registration hash history

Four recorded source-mapping comparisons differ for source/ant/__init__.py. The saved hashes match historical project commits 9fb0c8720d8823c2057ead34829797231fd8ca2a and e74d3fe5f08b3f070ff314e6f0b6349985458b61. Current registration content extends those exact saved bytes; additional task registrations were appended. The current source remains an exact copy of the approved project HEAD. Historical manifests were not rewritten.

## Packaging and imports

Root setuptools namespace discovery includes 284 package paths after the framework overlay, while regular discovery identifies ant and ant.agents. Editable installation was not performed. Runtime checks must force module resolution to the submission source. Canonical evaluator entry points cannot be imported passively because their module-level code parses arguments, creates output directories and launches AppLauncher.
