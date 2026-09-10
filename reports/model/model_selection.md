# Model Selection Report

**Generated**: 2026-09-09 10:13:31 UTC

---

## Experiment Comparison

| Experiment | Strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN |
|------------|----------|-----------|--------|------|---------|--------|------|------|
| baseline_smote_scale_weight | Baseline: SMOTE + scale_pos_weight | 0.0318 | 0.9988 | 0.0617 | 0.9998 | 0.9866 | 74,894 | 3 |
| smote_only | SMOTE only, no scale_pos_weight | 0.7940 | 0.9980 | 0.8844 | 0.9997 | 0.9887 | 638 | 5 |
| scale_weight_only | scale_pos_weight only, no SMOTE | 0.8693 | 0.9963 | 0.9285 | 0.9997 | 0.9889 | 369 | 9 |

---

## Threshold Analysis (Selected Experiment)

| Threshold | Precision | Recall | F1 | FP | FN |
|-----------|-----------|--------|------|------|------|
| 0.50 | 0.8693 | 0.9963 | 0.9285 | 369 | 9 |
| 0.60 | 0.8790 | 0.9963 | 0.9340 | 338 | 9 |
| 0.70 | 0.8892 | 0.9963 | 0.9397 | 306 | 9 |
| 0.80 | 0.9036 | 0.9963 | 0.9477 | 262 | 9 |
| 0.90 | 0.9099 | 0.9959 | 0.9510 | 243 | 10 |

---

## Selected Model

- **Experiment**: `scale_weight_only`
- **Selected Threshold**: `0.90`
- **Precision at threshold**: `0.9099`
- **Recall at threshold**: `0.9959`
- **F1 at threshold**: `0.9510`

---

## Selection Reasoning

Three imbalance strategies were compared: baseline_smote_scale_weight, smote_only, scale_weight_only.
- **baseline_smote_scale_weight**: Recall=0.9988, PR-AUC=0.9866, Precision=0.0318, F1=0.0617, FP=74,894, FN=3
- **smote_only**: Recall=0.9980, PR-AUC=0.9887, Precision=0.7940, F1=0.8844, FP=638, FN=5
- **scale_weight_only**: Recall=0.9963, PR-AUC=0.9889, Precision=0.8693, F1=0.9285, FP=369, FN=9

**scale_weight_only** was selected because it achieved the best combination of Recall (0.9963) and PR-AUC (0.9889) while maintaining 9 false negatives. 

Threshold 0.90 was selected because it achieves Recall=0.9959 (≥ 0.95) with F1=0.9510, balancing fraud detection against 243 false positives.
