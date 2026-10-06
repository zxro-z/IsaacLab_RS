# Submission whitespace policy

The full initial-import `git diff --cached --check` returns nonzero (exit code 2), with **3,020 findings** in preserved upstream framework files, historical project code, and raw runtime/evidence files. These are approved provenance-preserving exceptions, not submission blockers.

Submission-authored/modified content passes the scoped check: the root README, submission provenance helper, four evaluator files changed only for provenance/path handling, `docs/submission_provenance/`, and new validation/report content. Raw console logs (`*.log`) and preserved patch context (`*.patch`) remain evidence and are excluded from the authored-report scope.

No upstream, historical, raw, or provenance file was reformatted to suppress these warnings. The complete initial-import check output remains recorded in the final-cleanup validation directory.
