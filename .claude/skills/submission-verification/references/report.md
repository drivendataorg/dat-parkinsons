# Reporting

## Subagent handoff format

Every reviewer reports in this shape.

```markdown
## Subagent: <name>

Status: pass | pass with notes | blocked | fail | needs organizer review

No competitor code executed: yes | no

Read-only inspection performed:
- `<files, pages, metadata, or artifacts reviewed>`

Competition-specific checks applied:
- `<check from verification_comp_specific_rules.md or solution_overview.md>`

Evidence:
- `<file path or artifact>`: <specific finding>

Findings:
- Severity: critical | high | medium | low
  Summary: <one sentence>
  Evidence: <path, line, official page, metadata, or artifact hash>
  Recommendation: <action>

Open questions:
- <question>
```

## Final report sections

The Coordinator's report must contain all of these.

- Submission identifier, competitor/team name, and leaderboard score if provided.
- Competition materials reviewed, with required and optional materials separated.
- Summary of `verification_comp_specific_rules.md`, including forbidden datasets, allowed datasets, and added competition-specific checks.
- Optional materials that were unavailable and the review areas limited by their absence.
- Review scope and explicit confirmation that no competitor code was executed.
- Solution overview document path.
- Code integrity findings.
- Data source and leakage findings.
- Release-readiness notes, if release expectations were provided.
- Blocking issues, non-blocking issues, and open questions.
- Final recommendation: `pass`, `pass with notes`, `needs organizer review`, or `disqualify`.

## Minimum evidence requirements

- Source code references must include file paths and line numbers when possible.
- Each subagent must explicitly confirm that no competitor code was executed.
- Findings tied to `verification_comp_specific_rules.md` must cite the relevant note and explain how it was applied.
- Prediction comparisons must use provided files only and include filenames and hashes when available.
- Score references must distinguish official leaderboard scores from static-review observations, and should be omitted when no leaderboard score was provided.
- Any suspected cheating or rules violation must include enough evidence for an organizer to independently inspect the issue.
