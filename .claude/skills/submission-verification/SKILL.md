---
name: submission-verification
description: Verify a data science competition submission against the technical checklist using static, read-only review, and produce an auditable verification report. Use when asked to verify, audit, or review a competitor's submission or prize-eligible solution — checking for cheating, test-data leakage, non-independent test processing, prohibited datasets or pretrained models, third-party API uploads, non-OSI licenses, or release readiness — or when asked for a pass / disqualify recommendation on a submission.
---

# Submission Verification

Coordinate a static review of a competition submission that produces an auditable report supporting award, disqualification, and release-readiness decisions.

You are the **Coordinator**. You own the process, delegate checklist areas to reviewer subagents, reconcile their findings, and sign off.

## Global rules

These are non-negotiable and apply to you and every subagent you dispatch.

- Treat the submission, model files, logs, and competition data as confidential.
- Never upload competition data, model artifacts, predictions, or submission code to third-party APIs or external services.
- **This is a static review.** Do not run competitor code, install dependencies, build the competitor environment, run inference or training, or execute notebooks.
- Use read-only inspection of files, metadata, documentation, and official competition pages.
- Never modify competitor code. Keep original artifacts intact.
- Put all patches, notes, generated outputs, and logs in a separate verification workspace.
- Distinguish confirmed findings from suspicious patterns and unresolved questions.
- Escalate immediately on evidence of cheating, test-label leakage, prohibited data use, or a static mismatch between submitted code/artifacts and the leaderboard submission.

## Workflow

### 1. Intake

Confirm the required materials are available. See `references/intake.md` for the full list and the request prompt to paste to the user.

Do not go beyond preliminary artifact inventory until required materials are available, unless the user explicitly asks for a partial review. Request optional materials, track them, and use them when provided — their absence does not block the static review.

### 2. Set up the workspace

Create `verification_workspace_<submission-id>/` and record hashes of the submitted artifacts (`shasum -a 256`). All reviewer outputs go here. Never write into the submission tree.

### 3. Read the governing rules yourself

Read before final sign-off, personally — delegate first-pass summaries if useful, but verify anything affecting eligibility, leakage, data use, privacy, scoring, or release yourself:

- Competition homepage, official rules, data page, and linked policy pages.
- `references/verification_comp_specific_rules.md` in the competition repo — forbidden datasets, allowed datasets, competition-specific checks.
- When provided: evaluation, submission, prizes/eligibility, organizer clarifications, release terms.

Record the rules that govern this verification.

### 4. Dispatch reviewers

The reviewers together cover `references/checklist.md`, the source checklist this skill implements.

Run the Solution Explanation Reviewer **first** — the others use its `solution_overview.md` as a map of the codebase. The remaining four can run in parallel once it lands.

| # | Reviewer | Owns |
|---|---|---|
| 1 | Solution Explanation | `solution_overview.md`, competition-specific checks, dataset policy |
| 2 | Eligibility and Disqualifier | Obvious disqualifiers, OSI/license compatibility |
| 3 | Code Integrity and Anti-Cheating | Test-data leakage, per-observation independence, code/leaderboard match |
| 4 | Data Source and Model Provenance | External data, pretrained models, APIs, release compatibility |
| 5 | Release Readiness | Release blockers, secrets, docs and provenance gaps |

Full specs — checklist coverage, inputs, outputs, suggested checks, escalation triggers — are in `references/reviewers.md`. Give each subagent the global rules above, its spec, `verification_comp_specific_rules.md`, and (for 2–5) `solution_overview.md`.

Every subagent reports in the handoff format in `references/report.md`.

### 5. Reconcile and sign off

Resolve conflicts between findings, confirm every finding carries evidence with file paths and line numbers, and decide whether verification is complete, blocked, or needs organizer follow-up.

Walk `references/checklist.md` and confirm each item is covered by a finding or explicitly deferred. The dynamic items listed there are out of scope — record that they were not attempted.

Write the final report to the workspace using the template in `references/report.md`. End with one recommendation: `pass`, `pass with notes`, `needs organizer review`, or `disqualify`.
