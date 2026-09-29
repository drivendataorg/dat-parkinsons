# Reviewer specs

Each reviewer performs static, read-only inspection. No competitor code is executed.

## 1. Solution Explanation Reviewer

Owns a readable technical explanation of how the submitted solution works. Runs first so the other reviewers can use `solution_overview.md` as a map of the codebase.

**Checklist coverage**

- Before beginning code review, read `verification_comp_specific_rules.md` and extract forbidden datasets, allowed datasets, and competition-specific checks.
- Add the extracted competition-specific checks to `solution_overview.md` so downstream reviewers can apply them.
- Review the codebase and submitted documentation to explain the overall modeling solution.
- Identify the main training, inference, feature engineering, post-processing, and ensembling paths by static inspection only.
- Write an independent markdown summary that future reviewers can use before reading the full codebase.

**Inputs**

- `verification_comp_specific_rules.md`.
- Full source tree.
- Submitted README, notebooks, writeup, model card, and config files.
- Documented training and inference entrypoints.
- Model artifacts and prediction files when provided and needed to understand the workflow.

**Outputs**

- `solution_overview.md` in the verification workspace.
- A `Competition-Specific Checks` section listing checks extracted from `verification_comp_specific_rules.md`.
- A `Dataset Policy` section listing forbidden datasets, allowed datasets, and unresolved dataset-policy questions.
- Diagram or ordered pipeline summary when useful.
- List of important files and what each one does.
- Plain-language explanation of model architecture, data preprocessing, training procedure, inference procedure, ensembling, post-processing, and known nondeterminism.
- Assumptions or gaps where the implementation is unclear.

**Suggested checks**

- Read `verification_comp_specific_rules.md` first, before inspecting the submitted solution.
- Convert each specific note into an actionable check with an owner when possible: Code Integrity, Data Source, Eligibility, or Release Readiness.
- Trace code paths from documented entrypoints by reading code, configs, notebooks, and scripts.
- Separate what the competitor claims from what the code appears to do.
- Note code paths that appear unused, stale, or inconsistent with the submitted writeup.
- Flag suspicious or rule-relevant behavior for the Code Integrity and Data Source reviewers.

**Escalation triggers**

- `verification_comp_specific_rules.md` is missing, ambiguous, or conflicts with official competition rules.
- The actual solution differs materially from the submitted explanation.
- The solution appears to use a dataset listed as forbidden in `verification_comp_specific_rules.md`.
- Key modeling or post-processing logic is hidden, missing, generated dynamically, or only available in binary artifacts.
- The solution cannot be explained from the submitted code and artifacts.

## 2. Eligibility and Disqualifier Reviewer

**Checklist coverage**

- Review top submission for obvious disqualifiers.
- Check for non-OSI models or code when the competition requires OSI-compatible components.
- Check for clear cheating indicators.

**Inputs**

- Competition rules.
- `verification_comp_specific_rules.md` and the `Competition-Specific Checks` section of `solution_overview.md`.
- Submission package.
- Model and dependency manifests.
- Submitted writeup, README, dependency notes, or model card.

**Outputs**

- License and eligibility summary.
- List of external models, packages, and code with license notes.
- Any clear disqualifier evidence with exact references.

**Escalation triggers**

- Non-OSI dependency where OSI compatibility is required.
- Undisclosed model or dataset that materially affects eligibility.
- Code or documentation explicitly describing prohibited behavior.

## 3. Code Integrity and Anti-Cheating Reviewer

**Checklist coverage**

- Read code to check for cheating.
- Verify test data was not used in training, pseudo-labeling, hand-labeling, tuning, or threshold selection.
- Verify test observations are processed independently.
- Verify code appears to match the leaderboard submission when leaderboard materials or submitted prediction files are provided.

**Inputs**

- `verification_comp_specific_rules.md` and the `Competition-Specific Checks` section of `solution_overview.md`.
- Full source tree.
- Saved models and checkpoints when provided.
- Prediction files when provided.
- Submission metadata and leaderboard file when provided.

**Outputs**

- Code path map for training and inference.
- Evidence table for any suspicious references to test data, labels, IDs, folds, manual rules, or leaderboard-specific constants.
- Assessment of whether inference is independent per test observation.
- Static assessment of whether the submitted code appears to match the leaderboard submission.
- Comparison between provided prediction files and submitted leaderboard metadata when available.

**Suggested checks**

- Search for terms such as `test`, `label`, `pseudo`, `manual`, `hand`, `leaderboard`, `fold`, `id`, `submission`, and known test filenames.
- Search for dataset names, source names, file names, and rule-specific terms listed in `verification_comp_specific_rules.md`.
- Inspect data joins, caches, hard-coded dictionaries, and post-processing rules.
- Check timestamps and artifact provenance where available.

**Escalation triggers**

- Training reads test labels or uses test-set-derived labels.
- Inference batches test samples together in a way that allows cross-observation leakage.
- Hard-coded predictions, labels, subject IDs, or leaderboard-specific post-processing.

## 4. Data Source and Model Provenance Reviewer

**Checklist coverage**

- Apply the forbidden and allowed dataset lists from `verification_comp_specific_rules.md`.
- Verify prohibited data sources were not used.
- Verify external models were not trained on prohibited sources when that matters for the competition.
- Verify external datasets or models do not prevent broad release when release expectations are provided.
- Verify data was not uploaded to third-party APIs.

**Inputs**

- Competition data rules.
- `verification_comp_specific_rules.md` and the `Dataset Policy` section of `solution_overview.md`.
- Dependency files.
- Download scripts.
- Training notebooks.
- Model cards, citations, and URLs when provided.
- Environment variables and API usage references.

**Outputs**

- Inventory of all external datasets, pretrained models, APIs, and downloaded artifacts.
- Release compatibility notes for each external dependency when release expectations are provided.
- Evidence of any network calls or third-party upload paths.

**Suggested checks**

- Review `README`, notebooks, scripts, config files, shell scripts, and CI files.
- Search for API clients, cloud storage uploads, remote inference calls, and hosted notebook assumptions.
- Search for all forbidden and allowed dataset names from `verification_comp_specific_rules.md`, including aliases and common abbreviations.
- Inspect pretrained model names against prohibited source lists.

**Escalation triggers**

- Use of prohibited source data.
- External model trained on prohibited data when competition rules forbid it.
- Dependency license or terms that block broad release when release expectations are provided.
- Upload of competition data to third-party APIs, including hosted LLM APIs.

## 5. Release Readiness Reviewer

**Checklist coverage**

- Note tweaks to code or docs for releasing when release expectations are provided.
- When release expectations are not provided, record only generic release risks such as secrets, private paths, missing provenance, or undocumented dependencies.

**Inputs**

- Findings from all other reviewers.
- `verification_comp_specific_rules.md` and any release-specific notes extracted into `solution_overview.md`.
- Source tree and documentation.
- License and dependency inventory.
- Release expectations when provided.

**Outputs**

- Release-blocking changes when release expectations are provided.
- Non-blocking cleanup suggestions.
- Documentation gaps.
- Reproducibility improvements.
- Packaging or artifact-size concerns.

**Suggested checks**

- Confirm instructions are sufficient for a fresh verifier.
- Identify hard-coded local paths, credentials, usernames, absolute paths, and private bucket references.
- Check whether model artifacts and dependencies can be redistributed when release expectations are provided.

**Escalation triggers**

- Secrets, credentials, private URLs, or personal data in the submission.
- Release terms conflict with competition obligations when release terms were provided.
- Documentation omits required data or model provenance.
