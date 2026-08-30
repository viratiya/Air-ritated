# Data cleaning report

- Raw train: 7485 rows x 15 cols
- Raw test:  1872 rows x 14 cols
- Raw test_labels: 1872 rows

## Dropped column
- `NMHC(GT)`: 87.8% missing in train, 100% missing in test -> no signal, unusable as a feature. Also removed from `data_dictionary.csv` reference below.

## Target-missing rows dropped
- train: removed 1592 rows where `CO(GT)` was -200 (sentinel for a sensor/analyzer outage) -> 5893 rows remain.
- test / test_labels: removed 91 rows (kept aligned) where the sealed `CO(GT)` ground truth was -200 -> 1781 rows remain.

## Remaining missing feature values
- train: 2932 missing cells linearly time-interpolated (data is unbroken hourly, so this is safe) -> 0 missing cells remain.
- test: 867 missing cells linearly time-interpolated -> 0 missing cells remain.

## Final integrity checks
- Duplicate DateTime in train: 0
- Duplicate DateTime in test: 0
- train/test DateTime overlap (leakage check): 0
- test/test_labels still row-aligned: True
- Any -200 sentinel remaining anywhere: train=0, test=0, labels=0

## Final shapes
- train_clean.csv: 5893 rows x 15 cols
- test_clean.csv: 1781 rows x 14 cols
- test_labels_clean.csv: 1781 rows
