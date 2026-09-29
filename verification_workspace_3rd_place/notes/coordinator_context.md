# Coordinator context — 3rd place (paulonium)

## Global rules (non-negotiable, all reviewers)
- Submission, model files, logs, and competition data are confidential.
- Never upload competition data, model artifacts, predictions, or submission code to third-party APIs or external services. No web fetches of competitor-hosted URLs (e.g. Hugging Face).
- STATIC REVIEW ONLY. Do not run competitor code, install deps, build envs, run inference/training, execute notebooks, or run competitor scripts/tests (incl. build/verify_pkg.sh, tests/). Do not import competitor modules. Do not unpickle/torch.load competitor artifacts.
- Read-only inspection of files, metadata, documentation. Never modify files under `3rd Place/` or `references/`.
- All outputs go in `verification_workspace_3rd_place/`.
- Distinguish confirmed findings / suspicious patterns / unresolved questions.
- Escalate on evidence of cheating, test-label leakage, prohibited data use, or static mismatch between submitted code/artifacts and the leaderboard submission.

## Paths
- Source tree: `3rd Place/`
- Competition rules for this verification: `references/verification_comp_specific_rules.md` (organizer-filled; authoritative)
- Competition materials: `references/competition_materials.md`
- Intake: `references/submissions/3rd_place.md`
- Harness zip (from platform): `references/submissions/3rd_place/3rd_place_submission.zip` — code+json extracted (read-only copies) at `verification_workspace_3rd_place/extracted/harness_zip/`; do not extract weights.
- Scored predictions CSV: `references/submissions/3rd_place/3rd_place_submission.csv`
- Hashes: `verification_workspace_3rd_place/hashes/`
- Official training data (organizer copy): `/Users/ishashah/DrivenData/competition-sfmn-parkinsons/data/final/public/` (train_labels.csv, 1362 train UIDs)
- Official test UID list (NO labels needed): `/Users/ishashah/DrivenData/competition-sfmn-parkinsons/data/final/private/submission_format.csv` (524 test UIDs). Use for grepping test UIDs in competitor files. Do NOT open test_labels.csv.

## Coordinator findings so far (verified)
1. Harness zip: 58 files — main.py, canonize.py, datprep_box.py, datprep_iso.py, glandrm.py, assets/calibration.json {"a":0.78,"b":-0.1646,"member_logit_cap":6.0}, assets/canon_const.json, assets/ellipse.ts.pt, 50 × assets/fu_a{1..10}_dnet_f{0..4}.ts.pt (all 50 distinct sha256).
2. vs tree `3rd Place/inference/`: canonize.py, datprep_box.py, datprep_iso.py, glandrm.py, canon_const.json, ellipse.ts.pt byte-identical. main.py DIFFERS (harness 8762 B dated 2026-09-05; tree 10071 B) — tree adds a "MIXED ROSTER (2026-09-11)" path for ex_* members; diff at `notes/main_py_harness_vs_tree.diff`. With only fu_* assets the extra path is inert. calibration.json and all 50 fu_* weights are absent from the tree (README points to a private HF repo). README claims inference/main.py "is the exact program submitted" — inaccurate.
3. Scored CSV: 524 rows, uid order identical to submission_format, no dupes, values in [0.0078, 0.9892], no exact 0/1. 51 repeated values sit exactly at sigmoid(±0.78·6 − 0.1646), i.e. the member_logit_cap + calibration in the harness assets — consistent with the harness code. Log loss on the full test set recomputed by the Coordinator = 0.273213 (matches intake "private" score); public subset = 0.231707 vs intake public 0.231321 (small mismatch, open question).
