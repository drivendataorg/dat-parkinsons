# Intake

Solicit these from the user or organizer before verification starts.

## Required

- Competition homepage URL.
- Official rules page or copied rules text.
- Data page, data dictionary, and file manifest.
- `references/verification_comp_specific_rules.md` in the competition repo, containing forbidden datasets, allowed datasets, and competition-specific checks. It ships as a stub — an unfilled section counts as missing, not as `None`.
- Submitted source tree, archive, repository, or package to review.
- Competition-specific restrictions on external data, pretrained models, OSI licenses, third-party APIs, privacy, and submission limits.
- Any submitted writeup, model card, README, or dependency/provenance notes.

## Optional

- Evaluation metric, scoring script, and submission format.
- Public leaderboard score, private leaderboard score if available, and submitted prediction file.
- Model artifacts and checkpoints.
- Official train, validation, and test data locations.
- Release expectations, including target license, artifacts to publish, and broad-use constraints.

## Request prompt

Use this when required materials are missing.

```markdown
Please provide or confirm the materials needed for verification:

Required:
- Competition page:
- Official rules:
- Data page or data manifest:
- `verification_comp_specific_rules.md`:
- Submitted source tree, archive, repository, or package:
- External data/model policy:
- Privacy restrictions:
- Submitted writeup, model card, README, or provenance notes:

Optional:
- Evaluation metric and scoring script:
- Leaderboard score and submitted prediction file:
- Model artifacts and checkpoints:
- Train, validation, and test data locations:
- Release requirements:
- Any known organizer clarifications:
```

Track which optional materials were unavailable — the final report must state which review areas their absence limited.
