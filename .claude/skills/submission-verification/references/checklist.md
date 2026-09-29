# Technical verification checklist

The source checklist this skill implements. `reviewers.md` assigns each item to a reviewer; the Coordinator confirms every item is covered or explicitly deferred before sign-off.

Competition-specific items — privacy restrictions, forbidden datasets, allowed datasets — live in `verification_comp_specific_rules.md`, not here.

## Static review

- [ ] Review the top submission for obvious disqualifiers
  - [ ] Non-OSI models or code, where the competition requires OSI compatibility
  - [ ] Clear cheating
- [ ] Read the code to check for cheating. Confirm the competitor has:
  - [ ] Not used test data in training (e.g. pseudo-labeling, hand-labeling)
  - [ ] Processed test observations independently
  - [ ] Not used prohibited data sources
  - [ ] Not exceeded the submission limit by using different accounts
  - [ ] Submitted code that matches the leaderboard submission
  - [ ] Respected the competition-specific restrictions listed in `verification_comp_specific_rules.md`
- [ ] Data sources
  - [ ] External models trained on a prohibited source
  - [ ] External dataset or model that would prevent the model from being released for broad use
  - [ ] Competition data uploaded to a third-party API (e.g. ChatGPT)
- [ ] Note any tweaks to code or docs needed for release

## Out of scope for this skill

Dynamic verification. This skill never runs competitor code — a human performs these steps separately, and the final report states they were not attempted.

- [ ] Set up the competitor's environment and install dependencies
- [ ] Run inference using the competitor's provided model and check the score is within range of their competition score
- [ ] Run training code to confirm it runs without error
