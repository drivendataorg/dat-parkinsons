[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://drivendata-prod-public.s3.amazonaws.com/comp_images/sfmn-scans-cropped.png' width='600'>](https://www.drivendata.org/competitions/311/)

# DaT Parkinson's Challenge

## Goal of the Competition

Dopamine transporter (DaT) imaging is a key tool for helping clinicians distinguish Parkinsonian syndromes from other causes of similar symptoms. Interpreting these scans is time-consuming and depends on expert readers, and scans come from many different cameras, sites, and reconstruction settings.

The **DaT Parkinson's Challenge**, run in partnership with the Société Française de Médecine Nucléaire (SFMN), asked participants to build models that read a single DaT brain scan and predict the probability that a nuclear medicine reader would report it as abnormal. Submissions were scored by log loss on a hidden test set.

## What's in this Repository

This repository contains code from winning competitors in the [DaT Parkinson's Challenge](https://www.drivendata.org/competitions/311/) on DrivenData. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Log Loss | Private Log Loss | Summary of Model
--- | --- | --- | --- | ---
1st | Team The Avengers: [dvtle](https://www.drivendata.org/users/dvtle/), [SamuelAnsel](https://www.drivendata.org/users/samuelansel/) | 0.2249 | 0.2532 | Trained a multi-view Siamese CNN on tight crops of the striatum to capture left-right asymmetry, using soft-labeled synthetic DaT PET scans as additional training data, and averaged an ensemble of 100 models.
2nd | [NavAttack](https://www.drivendata.org/users/NavAttack) | 0.2409 | 0.2731 | Rescaled each scan against its own background intensity, then blended regression models on striatal measurements, a 3D hemisphere-wise multiple-instance CNN, EfficientNet models on 2D projections, and a TabICLv2 tabular model using fixed logit weights and temperature scaling.
3rd | [paulonium](https://www.drivendata.org/users/paulonium/) | 0.2317 | 0.2732 | Reduced each scan to parameter-free 2D projection maps, read by DenseNet121 with left-right sagittal fusion at the first convolution, trained with synthetic mild-abnormal scans made by simulating dopamine loss in normal scans, and ensembled 50 models (10 members × 5 folds) with flip test-time augmentation and temperature calibration.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winners Blog Post**: [Meet the winners of the DaT Parkinson's Challenge](https://drivendata.co/blog/dat-parkinsons-winners)