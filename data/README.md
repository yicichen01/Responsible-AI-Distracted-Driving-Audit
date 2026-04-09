# Data Access

This repository does not include the full image dataset because of file size constraints.

## Primary dataset

The project uses the **State Farm Distracted Driver Detection** dataset from Kaggle:

https://www.kaggle.com/competitions/state-farm-distracted-driver-detection

## Expected structure

```text
data/
├── train/
│   ├── c0/
│   ├── c1/
│   ├── c2/
│   ├── c3/
│   ├── c4/
│   ├── c5/
│   ├── c6/
│   ├── c7/
│   ├── c8/
│   └── c9/
└── test/
```

## Subgroup audit setup

For the interpretability and fairness audit, a smaller manually selected subset was also constructed for each driving behavior category:

- White drivers
- Non-White drivers

These subgroup folders were used for Grad-CAM comparison and qualitative attention analysis.

