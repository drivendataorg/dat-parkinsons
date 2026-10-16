# MODE=full retrain report

Root: `<run>`  
Written 2026-09-28 06:27:25 EDT

| area | status | result |
|---|---|---|
| run | PASS | train_all.sh MODE=full exited 0 after 22199 s (6.17 h) |
| stages 01-03 | PASS | 13 of 13 files byte-identical to the documented hashes |
| stage 04 | PASS | crops_v3_b.npz sha256 equal to the 25 Sep rebuild |
| stage 04 | PASS | crops_v3_f2.npz sha256 equal to the 25 Sep rebuild |
| side tables | PASS | 6 of 6 byte-identical to the original tree (qc_v3_b.csv, qc_v3_f2.csv, eda_real_geometry.csv, hdr_fingerprint.csv, scale_f2.csv, mixture_shape.csv) |
| stage 07 recipe | PASS | tabular_recipe.json and context.npz byte-identical to the scored recipe and the original context |
| stage 07 order | PASS | feature_order_v17.json equals the scored configuration's 479 names |
| checkpoints 3d | PASS | 34 of 34 retrained checkpoints tensor-identical to the originals (34 expected); largest tensor difference 0 |
| checkpoints 2d | PASS | 18 of 18 retrained checkpoints tensor-identical to the originals (18 expected); largest tensor difference 0 |
| per-run OOF | PASS | 9 five-fold runs; OOF log loss change vs original: mil48a +0.0000, mil48b +0.0000, m31000 +0.0000, m33000 +0.0000, m35000 +0.0000, m37000 +0.0000, e2dmipf2 +0.0000, e2dmipf22 +0.0000, e2db3f2 +0.0000 |
| stacker | PASS | cross-fit log loss 0.20695 at T=1 (original 0.20695, +0.00000), 0.21671 at T=1.4 (original 0.21671); AUROC 0.97228; scale 0.531211 (orig 0.531211), intercept 0.184314 (orig 0.184314); ensemble file byte-identical |
| package | PASS | 82 entries, static gates passed; 52 of 82 entries identical to the scored archive, 30 differ (CRC32 or size) |
| parity | PASS | rebuilt vs scored package on 150 scans: max abs dp 0, mean 0, 0 above 1e-3, 0 above 1e-2; log loss 0.09811 vs 0.09811, AUROC 0.99873 vs 0.99873 |
| environment | PASS | scored package in this inference environment vs the 14 Sep predictions: max abs dp 0 |
| template | PASS | template_v6 rebuilt from this run's crops_v3_b.npz: sha256 identical to the scored asset, 0 of 110592 voxels differ |
| TorchScript | PASS | 18 retrained .ts.pt vs released: max abs logit diff 0 on 256 cached images |
| TabICL contexts | PASS | re-frozen contexts vs released, 3 outputs on the 1,362 training rows: max abs dp 0 |
| logs | INFO | 58 log/text files scanned, 0 contain scan ids (never copy these into the release): [] |

## Stage timings

| stage | start (s) | duration (s) |
|---|---:|---:|
| 00 verify staged data | 0 | 0 |
| 01 folds | 0 | 1 |
| 02 registered feature tables | 1 | 494 |
| 03 linear members lin / lin4 | 495 | 19 |
| 04 CNN input caches | 514 | 791 |
| 05 3D hemisphere-MIL CNN (34 checkpoints) | 1305 | 11791 |
| 06 2D EfficientNet members (18 checkpoints) | 13096 | 8760 |
| 07 TabICL block | 21856 | 301 |
| 08 stacker + image package spec | 22157 | 19 |
| 09 package submission.zip | 22176 | 23 |
| done: outputs/v17/submission.zip | 22199 |  |

## Per-run results

| run | OOF log loss | original | change | AUROC | original | time (s) |
|---|---:|---:|---:|---:|---:|---:|
| mil48a | 0.26428 | 0.26428 | +0.00000 | 0.95638 | 0.95638 | 1686 |
| mil48b | 0.26399 | 0.26399 | +0.00000 | 0.95702 | 0.95702 | 1684 |
| m31000 | 0.26389 | 0.26389 | +0.00000 | 0.95663 | 0.95663 | 1684 |
| m33000 | 0.26670 | 0.26670 | +0.00000 | 0.95556 | 0.95556 | 1683 |
| m35000 | 0.26458 | 0.26458 | +0.00000 | 0.95683 | 0.95683 | 1683 |
| m37000 | 0.26056 | 0.26056 | +0.00000 | 0.95758 | 0.95758 | 1684 |
| e2dmipf2 | 0.28655 | 0.28655 | +0.00000 | 0.95170 | 0.95170 | 1494 |
| e2dmipf22 | 0.28637 | 0.28637 | +0.00000 | 0.95354 | 0.95354 | 1494 |
| e2db3f2 | 0.28077 | 0.28077 | +0.00000 | 0.95621 | 0.95621 | 3891 |

Full-data runs (s): mf21000 422, mf23000 421, mf25000 421, mf27000 421, e2dmipf2_full 373, e2dmipf22_full 374, e2db3f2_full 969
