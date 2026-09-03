# Experiment Protocol (Task A / Task B)

## Task A: Recognition

Goal:
- Given a clip, predict coarse or fine action label.

Inputs:
- Trimmed clips from annotated segments.

Outputs:
- `coarse` prediction
- `fine` prediction

Metrics:
- Top-1 Accuracy
- Macro F1
- Balanced Accuracy

## Task B: Temporal Localization

Goal:
- Given a full video, predict `[start, end, class]`.

Inputs:
- Full videos and temporal annotations.

Outputs:
- Temporal segments + labels.

Metrics:
- mAP @ IoU (0.3, 0.5, 0.75)
- Recall @ IoU

## Task C (Application Extension): Skill-Level Classification

Goal:
- Predict performer skill level (`beginner` / `intermediate` / `expert`) from hand-motion keypoint sequences.

Inputs:
- Per-video keypoint sequence JSON (from Gesture Coach extraction).
- `skill_level` metadata at video level in `master_annotations.json`.

Outputs:
- Skill-level class prediction per sample.

Metrics:
- Macro F1 (primary)
- Balanced Accuracy
- Top-1 Accuracy

Coordinate-mode ablation (fixed split/classifier):
1. `2d` (x, y, conf)
2. `3d_image` (x, y, z, conf)
3. `3d_world` (x, y, z, conf)
4. `3d_mano` (MANO-compatible joints)

Recommended protocol:
- Subject-wise group k-fold (avoid same subject leakage between train/test).
- Same folds across all coordinate modes.
- Report mean/std across folds (and optional multi-seed repeat).

## Split Policy

Default recommendation:
- split by `subject_id` (not random segment split)
- keep all clips from one subject in one subset
- for publication-quality results, prefer subject-wise `k-fold` evaluation

Default ratios:
- train: 0.7
- val: 0.2
- test: 0.1

K-fold recommendation:
- `k=5` when subject count allows
- each fold: one subject-group partition as test
- validation sampled from remaining subject groups (fixed seed)

## Ablation Order

1. Video only
2. Single-view vs multi-view
3. Add weight feature
4. Add gyro feature (with missing-modality masking)

## Reporting

Always report:
- split seed
- subject distribution per subset
- class distribution per subset
- mean/std across at least 3 seeds when possible
- per-class precision / recall / F1
- confusion matrix (Task A)
- for Task C, include 2D/3D/MANO ablation table and best mode by Macro F1
