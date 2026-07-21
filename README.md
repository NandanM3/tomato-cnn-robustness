# LeafVision

CNN-based plant disease detection, with a focus on how well these models perform on field images.

This repo contains the model, training code, and experiments for the paper
*"Evaluating the Reliability and Practical Viability of CNN-Based Early-Stage
Plant Disease Detection Under Simulated Real-World Conditions"* (in progress).

## Research Question

To what extent can CNN-based image classification systems reliably detect
early-stage plant diseases under simulated real-world conditions, and what
practical limitations affect their viability for agricultural deployment?

## What's in this repo

- A tomato leaf disease classifier (Blight, Early Blight, Healthy) built with
  transfer learning (EfficientNet-B0)
- A from-scratch baseline (no pretrained weights) for comparison
- A pipeline for simulating realistic field conditions (lighting, blur, noise,
  resolution loss) on clean dataset images
- Evaluation scripts measuring how much accuracy/precision/recall/F1 degrade
  under those simulated conditions
- A small economic discussion connecting model robustness to real deployment
  cost tradeoffs for farmers

## Repo Structure

```
leafvision/
├── data/                    # datasets (gitignored — see Data section below)
├── models/
│   ├── baseline_transfer/   # EfficientNet-B0 + transfer learning
│   ├── baseline/    # same architecture, trained from scratch
│   └── common/              # shared model def, dataloaders, utils
├── experiments/
|   |── phase1_dataprep/        # Splitting dataset
│   ├── phase2_clean_eval/       # baseline metrics on clean data(data set below)
│   ├── phase3_degradation/      # scripts that generate degraded image sets(from same data set)
│   ├── phase4_robustness_eval/  # eval on degraded sets + aggregation
│   └── transfer_vs_scratch/     # transfer learning comparison
├── results/
│   ├── tables/               # CSVs backing every number in the paper
│   └── figures/               # plots used in the paper
├── paper/
│   ├── citations.bib
│   ├── lit_review_notes/
│   └── drafts/
└── notebooks/                # exploratory only — nothing load-bearing lives only here
```

## Setup

```bash
git clone https://github.com/NandanM3/leafvision.git
cd leafvision
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The tomato leaf dataset is not committed to this repo (see `.gitignore`).
[Add: where to download it / source dataset name, e.g. PlantVillage subset]

To reproduce:
1. Download the dataset to `data/raw/`
2. Run `experiments/phase2_clean_eval/prepare_data.py` to generate train/val/test splits
3. Run `experiments/phase3_degradation/generate_conditions.py` to build the degraded test sets

## Reproducing Results

| Result | Script |
|---|---|
| Clean baseline metrics (Phase 2) | `experiments/phase2_clean_eval/` |
| Degraded condition generation (Phase 3) | `experiments/phase3_degradation/` |
| Robustness evaluation (Phase 4) | `experiments/phase4_robustness_eval/` |
| Transfer learning vs. scratch comparison | `experiments/transfer_vs_scratch/` |

Each experiment folder has its own short README with exact run instructions.

## Status

Work in progress — model training, robustness evaluation, and paper
writing are ongoing. See `paper/drafts/` for the current paper draft.

## Citation

If you use this code, please cite:
```
[Add citation block ]
```

## License

[Add a license]
