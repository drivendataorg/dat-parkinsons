# Competition-specific verification rules

Rules unique to the DaT Parkinson's Challenge (DrivenData competition 311) that the `submission-verification` skill applies on top of its generic checklist. Fill each section in from the official rules, the design document, and any organizer clarifications; cite the source so a reviewer can check it.

Leave a section as `None` when it does not apply — an empty section reads as "not yet filled in" and blocks verification.

> Drafted 2026-09-29 by the verification Coordinator from the public competition pages (see Sources). Quotes were taken from the pages as fetched and are to be re-checked verbatim against the live pages before sign-off. Items marked **TODO (organizer)** need an answer from the organizers that the website does not give; replace each with the answer or `None`.

## Forbidden datasets and sources

Datasets, corpora, and data sources competitors may not train on, including as the training data of an external pretrained model. List aliases and common abbreviations so reviewers can grep for them.

- No dataset is named as forbidden on the public pages. The About page describes the competition data as newly collected from ten French centers (CHU Toulouse, CHRU Nancy, CHRU Tours, Centre Henri Becquerel, CHRU Brest, CHU Grenoble Alpes, Hospices Civils de Lyon, Centre Jean Perrin, AP-HM Marseille, Institut Godinot) and gives no indication it derives from a public resource. [About]
- Any external data that is **not** freely and publicly available to all participants is forbidden for prize-eligible solutions. This includes private clinical or institutional scan archives and annotations a competitor has access to through their employer or clinical practice. [Homepage]
- Any external data or pretrained model under a non-commercial license is forbidden for prize-eligible solutions: "no NC, CC NC, or CC BY-NC licenses". [Homepage]
- Any external data the competitor does not hold "all rights, licenses and permissions" to use is forbidden. [Rules]
- Is any part of the competition data (train or test) also present in a public or semi-public dataset, or has any of it been published by the contributing centers? If so, name it here with aliases; training on it would be test-label leakage. - No, none of it is present in a public dataset or been published.
- **TODO (organizer):** Does data behind an application or data use agreement count as "freely and publicly available"? Relevant DaT SPECT sources to rule on, with grep aliases: PPMI (Parkinson's Progression Markers Initiative, `ppmi`, `loni`, `ida.loni.usc.edu`); any other public DaT SPECT / ioflupane / DaTscan cohort.: As long as the DUA says the data is open source then it's ok.

## Allowed datasets and sources

External data explicitly permitted, with any conditions attached.

- Competition data: `data/niftis/{uid}.nii.gz` training scans and `train_labels.csv` (`uid`, `is_pathologic` ∈ {0.0, 1.0}). [Problem description]
- External data and pretrained models are allowed, provided that, for prize eligibility, (1) the external data is "freely and publicly available to all participants", and (2) the data and models are licensed so the resulting model "can be released for broad use … including for commercial purposes". [Homepage]
- The competitor must have "all rights, licenses and permissions" to use the external data "without the need for additional approvals from any third party". [Rules]
- "All external data must be shared with the challenge organizers, regardless of prize eligibility, to allow for independent result verification." A reviewer must confirm each external source was disclosed. [Homepage]
- **TODO (organizer):** List any datasets organizers explicitly approved during the competition (forum or email), or `None`. None

## Pretrained models and licensing

Whether external pretrained models are allowed, and whether OSI-compatible licenses are required for models, code, and dependencies.

- Pretrained models are allowed, under the same license condition as external data: no NC, CC NC or CC BY-NC licenses. [Homepage]
- The pretraining data of any pretrained model is subject to the same rules. A model trained on a forbidden or non-commercially licensed dataset is not allowed. [Homepage; generic checklist]
- The winning solution must be licensed under the MIT License. [Rules; Homepage]
- "Any software required to generate the solution must also be available under an open source license that does not prohibit free commercial use." [Rules] Solutions must be "reproducible using only free open source software". [About]

## Privacy restrictions

Inferences, attributes, or re-identification the solution must not perform on competition data.

- "Participants explicitly agree to make no attempt to utilize the Data to identify specific individuals," and must immediately delete the relevant data and notify DrivenData if they discover an identity inadvertently. [Rules]
- Reviewers should flag any code that reads or derives patient-identifying header fields from the NIfTI files, or links scans across UIDs to individuals.
- **TODO (organizer):** Is there any other attribute the solution must not infer (for example site, scanner or demographics), or `None`? Note that the rule on independent processing (below) already constrains test-time use of site or scanner information shared across test cases. None

## Third-party APIs

Whether competition data may be sent to hosted APIs, and any permitted exceptions.

- Competition data may not be shared with third-party APIs or services. [Problem description]
- The code execution container has no network access at test time. [Code submission format] Any network call in the inference code is a finding. Reviewers must also check training and preprocessing code, including any annotation or segmentation tooling, for uploads of competition data to hosted services.
- Participants are prohibited from privately sharing code or data developed with or based on the competition data. [Rules]

## Submission limits

Per-competitor submission cap and the rule on multiple accounts. Note who can check account-level activity.

- "The number of submissions allowed is restricted to a fixed value or rate on a per-competition basis. Any attempt to circumvent this limit will result in disqualification." [Rules]
- Multiple accounts are prohibited, and so is privately sharing code or data outside a team. [Rules]
- The static review cannot check account-level activity; only DrivenData can, from platform records. **TODO (organizer):** What was the submission cap for this competition, and has DrivenData checked the winners' accounts for multi-account activity? Or record "checked by DrivenData, no issues". Checked by DrivenData, no issues

## Release expectations

Target license, artifacts to publish, and broad-use constraints the winning solution must satisfy. Omit if the solution will not be released.

- License: MIT for the winning solution; all required software must be under open-source licenses that allow commercial use. [Rules; Homepage]
- Winners must deliver their code and documentation to DrivenData, complete the Winning Model Documentation Template, and sign the required documents. [Rules]
- Winning models will be released as open source and indexed in BOAS (Bibliothèque Ouverte d'Algorithmes en Santé), France's open health algorithm library. [Homepage]
- The competition dataset itself is expected to be released as open data later. That future release does not retroactively allow external data derived from it during the competition. [Homepage]
- **TODO (organizer):** Which artifacts must be published (code only, or code plus weights), and where (DrivenData's GitHub, Hugging Face, BOAS)? Can model weights trained on the competition data be released before the dataset is published? Code only will be published when winners are announced. Model weights trained on comp data can be released before dataset is published if in a private location only organizers can access.

## Other competition-specific checks

Anything else a reviewer must verify for this competition. Name the reviewer that should own each one — Code Integrity, Data Source, Eligibility, or Release Readiness.

- **Code Integrity:** "Each test data sample should be processed independently during inference without the use of information from other cases in the test set." Pseudo-labeling and unsupervised learning on test data are prohibited. Check for batch-level normalization, test-set statistics, cross-case calibration, clustering or TTA that pools test cases. [Code submission format]
- **Code Integrity:** The submitted code-execution package must have `main.py` at the zip root and write `submission.csv`. It reads from `/code_execution/data/niftis/` and `submission_format.csv`, runs offline within 3 h on a single A100 80 GB, 24 vCPUs and 220 GB RAM, and bundles all model assets. Compare the harness zip provided under `references/submissions/<N>_place/` with the source tree and the README's claims. [Code submission format]
- **Code Integrity:** Predictions must be probabilities in [0, 1], and the metric is log loss. Check the scored CSV for format, UID coverage against `submission_format.csv` if available, and signs of hand editing. [Problem description]
- **Data Source:** Every external dataset, annotation set and pretrained model must be disclosed to organizers. Cross-check the winner's documentation (`driven_data.md` / the Winning Model Documentation) against what the code actually loads. [Homepage]
- **Data Source:** Competitor annotations. "Participants may annotate provided training data as long as they are included with solutions to enable reproduction and do not overfit to the test set; Participants may not add any manual annotations to the provided test data. Eligible solutions need to be able to run on test samples automatically using the test data as provided." [Rules, Data use and code sharing] Check: (1) annotations are on training scans only, never test scans; (2) the annotations were delivered to organizers with the solution; (3) nothing suggests the annotation set was chosen or tuned against the test set or leaderboard.
- **Eligibility:** Rules exclude employees and officers of the sponsor. **TODO (organizer):** Confirm each winner is not a sponsor employee or officer, and not a member of the 11-person expert annotation panel or a contributing center with access to the test data or labels. Only organizers can confirm identity against these lists. Confirmed by DrivenData
- **Eligibility:** Open to natural persons aged 18 or over in permitted territories. Residents of the Russian Federation are ineligible for prize money. [Rules]

## Verification handling restrictions

Rules for the reviewers themselves (Coordinator and every reviewer subagent), on top of the skill's global rules.

- **No image viewing.** Reviewers must not open, view or render any image file, including `.png`, `.jpg`, `.gif`, `.svg`, `.tif`, `.dcm`, `.nii`/`.nii.gz` slices, notebook image outputs, and figures in docs, because they may depict competition scans. Reviewers may record an image file's path, size, hash and filename, and read any code that produced it (for example the plotting script) to infer what it shows. Record any image whose content matters to a finding as "not viewed; organizer to inspect". [Organizer decision, 2026-09-29]

## Sources

Official rules pages, design document sections, and organizer clarifications these rules came from.

- [Homepage] https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/987/: external data and model conditions, NC prohibition, disclosure, MIT, BOAS.
- [Rules] https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/rules/: external data rights, MIT and open-source dependencies, privacy, submission limits, multiple accounts, private sharing, winner obligations, eligibility.
- [Problem description] https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/990/: data manifest and dictionary, log loss, no third-party APIs.
- [Code submission format] https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/: zip layout, runtime paths, no network, limits, independent test processing.
- [About] https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/988/: data provenance (ten French centers), expert annotation panel, reproducibility with free OSS.
- Runtime repository: https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime (not yet reviewed).
- **TODO (organizer):** Organizer clarifications (forum posts, emails), with dates, or `None`. None
