# Coordinator context — 2nd place (NavAttack)

## Global rules (non-negotiable, all reviewers)
- Submission, model files, logs, and competition data are confidential.
- Never upload competition data, model artifacts, predictions, or submission code to third-party APIs or external services. No web fetches of competitor-hosted URLs (Dropbox, Hugging Face, etc.). Do not download the weights bundles.
- STATIC REVIEW ONLY. Do not run competitor code, install deps, build envs, run inference/training, execute notebooks, or run competitor scripts/tests/tools (pipeline/*.sh, scripts/*, tools/*, src/check_package.py, tabular/verify_*.py, docs/figures/make_figures.py). Do not import competitor modules. Do not unpickle/torch.load/np.load competitor artifacts, and do not decompress the .pkl.xz files. Do not unzip `tabicl_lib.zip` into anywhere but the workspace; listing it (`unzip -l`) and reading its text members via `unzip -p` is fine.
- NO IMAGE VIEWING (organizer decision 2026-09-29): never open/view/render .png/.jpg/.svg/.nii/.npy-as-image etc., including `2nd Place/docs/figures/*`. Record path/size/hash; read the code that produced it (e.g. make_figures.py) to infer content. Mark as "not viewed; organizer to inspect" if it matters.
- Read-only inspection. Never modify files under `2nd Place/` or `references/`.
- All outputs go in `verification_workspace_2nd_place/`.
- Distinguish confirmed findings / suspicious patterns / unresolved questions.
- Escalate on evidence of cheating, test-label leakage, prohibited data use, or static mismatch between submitted code/artifacts and the leaderboard submission.

## Paths
- Source tree: `2nd Place/`
- Competition rules for this verification: `references/verification_comp_specific_rules.md` (organizer-filled; authoritative)
- Extra governing rules A–F (verbatim from official pages): `verification_workspace_2nd_place/notes/governing_rules.md`; page text in `verification_workspace_2nd_place/official_pages/*.txt`
- Intake: `references/submissions/2nd_place.md` (NOTE: its csv/zip paths wrongly point at 3rd_place; the correct files are below)
- Harness zip (from platform): `references/submissions/comp_subs/2nd_place/2nd_place_submission.zip` (sha256 88b5611e…7fd). Code/json/md/license/tabicl_lib.zip extracted read-only copies at `verification_workspace_2nd_place/extracted/harness_zip/`; weights NOT extracted. Member hashes: `hashes/harness_zip_members.sha256`; listing: `notes/zip_listing.txt`.
- Scored predictions CSV: `references/submissions/comp_subs/2nd_place/2nd_place_submission.csv` (sha256 293d7041…37a)
- Official training data (organizer copy): `/Users/ishashah/DrivenData/competition-sfmn-parkinsons/data/final/public/` (train_labels.csv, 1362 train UIDs)
- Official test UID list (NO labels): `/Users/ishashah/DrivenData/competition-sfmn-parkinsons/data/final/private/submission_format.csv` (524 test UIDs). Use for grepping test UIDs in competitor files/logs. Do NOT open test_labels.csv.

## Coordinator findings so far (verified)
1. Harness zip: 82 members — 22 code/doc files at root + tabicl_licenses/LICENSE, and 60 assets (34 3D-CNN .pt, 18 TorchScript 2D .ts.pt, linear_lin.json, linear_lin4.json, template_v6.npy, tabicl-classifier-v2-20260212.ckpt, tabular_recipe.json, tabular_raw.pkl.xz, tabular_log.pkl.xz, inference_config.json). main.py at zip root.
2. All 22 code/doc files in the zip are byte-identical to `2nd Place/inference/` (incl. main.py, inference_main.py, tabicl_lib.zip). Assets are not in the tree (`inference/assets/` holds only .gitkeep); all 60 zip asset hashes match `2nd Place/models/SHA256SUMS.scored.txt` exactly. `SHA256SUMS.public.txt` differs for 18 TorchScript files (competitor says debug_pkl records scrubbed by tools/scrub_torchscript_debug.py).
3. Scored CSV: 524 rows, uid order identical to submission_format, 0 dupes, values in [0.005, 0.995], 115 rows at exactly 0.995, 4 at 0.005, 407 distinct values. Clip 0.005 comes from harness assets/inference_config.json:502 and inference/inference_main.py:253,475-477. Coordinator-recomputed log loss full test = 0.273088 (matches intake private 0.2731).
4. README links two public Dropbox bundles (`navattack_v17_weights.zip`, `navattack_v17_cnn_raw_checkpoints.zip`) for comp-trained weights. models/README.md says the frozen TabICL contexts let training labels be recovered. Organizer ruling: weights trained on comp data may be released only in a private location only organizers can access. Owner: Release Readiness / Data Source.
5. docs/figures has drift.png, score_progression.png etc. (not viewed). Reviewers should read make_figures.py to see whether any figure uses test-set data or leaderboard feedback.
