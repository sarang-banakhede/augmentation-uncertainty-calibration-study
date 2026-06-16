# Effect of Data Augmentation Strategies on Uncertainty Quantification and Model Calibration

> Does the way you train a model change how honest it is about what it doesn't know?  
> This project says — yes, quite a lot.

Most augmentation research stops at Dice scores. We go further: we ask whether the *choice of augmentation strategy* affects how well-calibrated a model's confidence is, and whether its uncertainty actually means something — i.e., does high uncertainty correlate with high error?

We evaluate five augmentation strategies on a U-Net with MC-Dropout, testing not just segmentation performance but calibration quality and out-of-distribution reliability on an unseen dataset (PH2).

The short answer: **better Dice does not mean better calibration**. Mixing augmentation achieves the 2nd-best Dice but the worst ECE. Occlusion achieves the best calibration *and* is the only strategy to improve DSC under domain shift.

---

## Datasets

**ISIC-2016** — Primary training and in-distribution test set. Dermoscopic skin lesion images with binary segmentation masks.

**PH2** — Out-of-distribution evaluation only. Never seen during training. Acquired under a different dermoscopy protocol, making it a natural domain shift testbed.

---

## Project Structure

```
.
├── src/
│   ├── config.py          # hyperparameters and dataset paths
│   ├── model.py           # U-Net with embedded MC-Dropout
│   ├── datasets.py        # ISIC-2016 and PH2 dataset classes
│   ├── augmentations.py   # five augmentation strategies
│   ├── metrics.py         # Dice, ECE, Brier, HD95, entropy
│   └── engine.py          # train loop, eval loop, MC-Dropout inference
│
├── train.py               # train all five strategies on ISIC-2016
├── evaluate.py            # OOD evaluation on PH2
├── visualize.py           # uncertainty map visualization
│
├── results/               # auto-generated after training
│   └── {strategy}/
│       ├── best_weights.pth
│       ├── training_history.csv
│       ├── isic_inference.csv
│       └── ph2_inference.csv
│
├── figures/               # plots go here
├── requirements.txt
└── README.md
```

---

## The Five Strategies

| Strategy | What it does |
|---|---|
| Baseline | No augmentation. Resize + normalize only. |
| Geometric | Flips, rotation (±30°), elastic deformation. Preserves intensity, changes shape. |
| Intensity | Brightness, contrast, Gaussian noise, gamma. Preserves shape, changes appearance. |
| Mixing | MixUp + CutMix (50/50 per batch). Creates ambiguous, interpolated training signal. |
| Occlusion | GridDropout + CoarseDropout. Forces the model to reason from partial information. |

All strategies use the same U-Net backbone, same hyperparameters, same number of MC-Dropout forward passes (T=20).

---

## What We Measure

**Segmentation** — Dice (DSC), IoU, Hausdorff Distance (HD95)

**Calibration** — Expected Calibration Error (ECE), Brier Score. These measure whether the model's confidence matches its actual accuracy.

**Uncertainty quality** — predictive variance (σ²) and predictive entropy (H) from MC-Dropout, evaluated via Spearman correlation with per-image error (1 − DSC). A high correlation means uncertainty actually flags bad predictions.

**OOD generalization** — all of the above repeated on PH2, which was never seen during training.

---

## Setup

```bash
git clone https://github.com/your-username/uq-augmentation.git
cd uq-augmentation
pip install -r requirements.txt
```

Download the datasets and place them as follows:

```
data/
├── ISIC2016/
│   ├── train/
│   │   ├── images/   # .png files
│   │   └── masks/    # .png files
│   └── test/
│       ├── images/
│       └── masks/
└── PH2Dataset/
    ├── image/        # .bmp files
    └── label/        # .bmp files named {id}_lesion.bmp
```

- **ISIC-2016** 
- **PH2** 

Update the paths in `src/config.py` if you place the data elsewhere.

---

## Quickstart

```bash
# 1. Train all five strategies on ISIC-2016 (saves to results/)
python train.py

# 2. Evaluate trained models on PH2 (out-of-distribution)
python evaluate.py

# 3. Visualize MC-Dropout uncertainty maps for a single image
python visualize.py --image /path/to/image.png
```

Each script accepts `--help` for available arguments.

---

## Paper

For complete methodology, experiments, and results, refer to the manuscript:

> **Effect of Data Augmentation Strategies on Uncertainty Quantification and Model Calibration in Skin Lesion Segmentation**  
> [Under review]
