DIV.repository-content {
    display: table
}
DIV.js-repo-meta-container {
    display: table-caption
}
DIV.readme {
    display: table-header-group
}

# Bacterial & Fungal Detection — ML Pipeline (v1 → v85)

End-to-end development of machine learning models for **bacterial and fungal detection**, focused on **robust classification**, **feature engineering**, **ensembles/meta-models**, **Bayesian hyperparameter optimization**, **hard negative mining**, and **dynamic model weighting**.

> **Status:** Iterative development through versions 1–85, aiming to improve stable “perfect precision” from **1/6** to **2/6** evaluation runs.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Development Pipeline Overview](#development-pipeline-overview)
  - [1) Data Preprocessing & Feature Engineering](#1-data-preprocessing--feature-engineering)
  - [2) Model Selection & Training](#2-model-selection--training)
  - [3) Optimization & Fine-Tuning](#3-optimization--fine-tuning)
  - [4) Evaluation & Performance Monitoring](#4-evaluation--performance-monitoring)
- [Key Achievements](#key-achievements)
- [Evolution: Version 1 → 85](#evolution-version-1--85)
- [Future Directions & Next Steps](#future-directions--next-steps)
- [Reproducibility & Suggested Repo Structure](#reproducibility--suggested-repo-structure)
- [Appendices (Concept Reference)](#appendices-concept-reference)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

This project documents a multi-iteration journey building ML models that classify bacterial vs. fungal samples. The goals have been to:

- Improve **classification accuracy** and especially **precision** in critical runs
- Reduce **bias** caused by class imbalance
- Increase **generalization** via augmentation and stronger feature representations
- Stabilize performance using **ensembles** and **confidence-aware** weighting
- Optimize efficiently using **Bayesian hyperparameter optimization**
- Address difficult edge cases using **hard negative mining**

---

## Development Pipeline Overview

The development follows a structured pipeline with four major steps:

### 1) Data Preprocessing & Feature Engineering

**Missing data handling, normalization, and augmentation**
- Missing values are handled via interpolation to maintain consistency.
- Normalization (e.g., Min-Max scaling, standardization) improves convergence and stability.
- Augmentation includes bootstrapping and synthetic sample generation to expand the dataset and improve generalization.

**Class balancing (SMOTE and over/under-sampling)**
- **SMOTE** is used to generate synthetic samples for minority classes.
- Over-sampling and under-sampling are tested across iterations to reduce bias toward majority classes.

**Feature extraction (spectral analysis and statistical features)**
- Spectral analysis is used to extract characteristic features for bacteria and fungi.
- Statistical features include mean, variance, skewness, and kurtosis to capture distributional structure.
- Feature selection is explored (e.g., PCA and RFE) to keep the most informative features while reducing noise.

---

### 2) Model Selection & Training

**Early models: CNN-based classification**
- Convolutional Neural Networks (CNNs) are used due to strong pattern recognition in spectral/high-dimensional data.
- Transfer learning with pre-trained CNN architectures (e.g., ResNet, VGG) is tested to improve feature extraction.

**Later improvements: ensembles and gradient boosting**
- CNN predictions are aggregated using ensemble methods to increase robustness.
- Gradient boosting techniques (XGBoost, LightGBM) are introduced as meta-models to refine predictions.

**Hybrid models via late fusion**
- Late fusion combines outputs from multiple models in a final decision stage.
- Weighting strategies assign higher influence to models with stronger confidence or historical reliability.
- Hybrid CNN + classical ML improves interpretability while maintaining strong performance.

---

### 3) Optimization & Fine-Tuning

**Bayesian hyperparameter tuning**
- Bayesian optimization is applied to efficiently explore hyperparameters with fewer evaluations than grid search.
- The search space is iteratively refined based on early signals, focusing compute on the most impactful parameters.

**Hard negative mining**
- Misclassified samples are identified and reintroduced into training with increased weighting.
- Feature enhancement and targeted training help the model separate similar but distinct classes/species/strains.

**Dynamic model weighting and confidence-based stacking**
- Confidence-aware stacking assigns dynamic weights to models based on reliability for specific samples.
- The ensemble adapts over time to favor models that generalize better, improving stability.

---

### 4) Evaluation & Performance Monitoring

**Metrics**
- **Accuracy:** overall correctness
- **Precision:** fraction of predicted positives that are true positives
- **Recall:** fraction of actual positives correctly identified
- **F1-score:** balance between precision and recall
- **ROC-AUC:** discrimination capability across thresholds

**Iterative refinements via confusion matrix analysis**
- Confusion matrices identify recurring misclassification patterns.
- Adjustments are made based on failure modes, enabling targeted improvements.

**Failure case identification and resolution**
- Error analysis highlights scenarios where the model performs poorly.
- Underperforming configurations are retrained with more data, improved features, or refined fusion strategies.

---

## Key Achievements

- Achieved up to **99.1% classification accuracy** using advanced stacking/ensemble methods.
- Improved precision stability from **1/6** runs achieving 100% precision toward a target of **2/6**.
- Implemented **hard negative mining**, increasing robustness on difficult classifications.
- Introduced **dynamic ensemble weighting**, improving prediction consistency.
- Applied **Bayesian hyperparameter tuning**, yielding ~**20% training time reduction** without sacrificing performance.

---

## Evolution: Version 1 → 85

The model progression can be grouped into five phases:

### Phase 1: Initial CNN-Based Models (v1–10)

**Approach**
- CNNs with minimal preprocessing using raw spectral input.
- Baseline architectures to establish initial performance.

**Challenges**
- Overfitting on limited data reduced generalization.
- Class imbalance biased predictions toward dominant classes.
- High feature variability caused unstable classification.

**Key learnings**
- Better feature representations were required beyond raw spectra.
- Augmentation was necessary to improve robustness.
- Addressing imbalance was critical for consistent results.

---

### Phase 2: Feature Representation & Data Augmentation (v11–30)

**Approach**
- Engineered features (mean, variance, skewness) improved separability.
- SMOTE balanced the dataset.
- t-SNE visualization assessed class separation in feature space.

**Results**
- Approximately **5–8% accuracy** improvement over Phase 1.
- Better generalization, though some misclassifications persisted.

---

### Phase 3: Ensemble Learning & Meta-Models (v31–50)

**Approach**
- Hybrid ensembles combining CNNs with XGBoost for decision refinement.
- Gradient boosting used as a meta-model on CNN outputs.
- Late fusion explored to improve stability and accuracy.

**Trade-offs**
- Higher compute cost due to multiple model training.
- Extensive hyperparameter tuning required for optimal ensembles.

---

### Phase 4: Hard Negative Mining & Dynamic Weighting (v51–70)

**Approach**
- Hard negative mining prioritized persistent misclassifications.
- Dynamic weighting adjusted model influence based on confidence.
- Bayesian optimization further tuned model performance.

**Results**
- Achieved **100% precision in 1/6** evaluation runs.
- Identified ensemble-weighting bottlenecks requiring additional refinement.

---

### Phase 5: Refinements & Stabilization (v71–85)

**Approach**
- Focused on increasing perfect-precision stability from **1/6** to **2/6**.
- Implemented multi-stage ensemble refinement to reduce variance.
- Optimized feature selection to improve decision boundaries.

**Notes**
- Later versions (v86–v88) showed performance degradation due to overfitting in ensemble configurations.
- Late fusion and dynamic weighting emerged as key levers for continued improvements.

---

## Future Directions & Next Steps

To push beyond the current milestone and stabilize perfect-precision performance, the next steps focus on hybrid modeling, better feature engineering, smarter fusion/weighting, and continual learning.

### 1) Hybrid Model Approaches

**Rationale**
CNNs are strong for local patterns but limited in capturing long-range dependencies. Transformers—especially Vision Transformers (ViTs)—can improve representation via self-attention.

**Implementation strategy**
- **Vision Transformers (ViTs):**
  - Split spectral inputs into patches and process via self-attention.
  - Pretrain on synthetic spectral datasets before fine-tuning.
- **Hybrid CNN + Transformer:**
  - CNN for low-level spectral pattern extraction.
  - Transformer for higher-level interactions and global dependencies.

**Expected impact**
- Better feature representation and correlation modeling across the spectrum.
- Improved generalization to unseen strains.
- Reduced run-to-run variance seen in CNN-only approaches.

**Challenges**
- High compute requirements.
- Need for careful fusion design and adequate data/transfer learning.

---

### 2) Feature Engineering Improvements

**GAN-based synthetic data augmentation**
- Train GANs (preferably conditional GANs, cGANs) on real spectra to generate realistic synthetic samples.
- Use synthetic data to boost minority classes (beyond SMOTE).
- Validate synthetic realism using t-SNE and anomaly detection (e.g., Isolation Forest).

**Recursive Feature Elimination (RFE)**
- Iteratively remove low-importance features to improve separability and reduce overfitting.
- Use gradient boosting feature importance as a starting point, then apply RFE with cross-validation.

**Expected impact**
- Stronger class balance, better minority-class performance.
- Reduced noise and redundancy in features.
- More stable decision boundaries and improved precision on edge cases.

**Challenges**
- GAN training instability and risk of unrealistic spectra (mode collapse).
- RFE can be computationally expensive, especially at scale.

---

### 3) Smarter Model Weighting & Late Fusion

**Rationale**
Static ensemble weights can be suboptimal. Dynamic weighting informed by attention mechanisms or meta-learning can prioritize the most reliable model per sample.

**Implementation strategy**
- **Attention-based weighting:**
  - Use self-attention layers to dynamically weight CNN vs. boosting vs. transformer predictions.
  - Incorporate per-class/per-strain historical reliability.
- **Meta-learning / Mixture of Experts (MoE):**
  - Train a meta-learner to decide which base model to trust for a given input.
- **Optimized late fusion:**
  - Compare weighted averaging, max pooling, and neural fusion layers.

**Expected impact**
- Higher precision and improved stability across runs.
- Better handling of ambiguous cases through adaptive model selection.

**Challenges**
- More complex training pipeline and need for meta-data/confidence calibration.
- Data requirements for generalizable meta-learning.

---

### 4) Continual Learning & Robust Generalization

**Rationale**
Static models don’t adapt to new strains or concept drift. Continual learning (CL) and transfer learning (TL) can improve long-term robustness.

**Implementation strategy**
- **Incremental/online learning:**
  - Update on new data without full retraining.
  - Use Elastic Weight Consolidation (EWC) to reduce catastrophic forgetting.
- **Transfer learning from larger spectral datasets:**
  - Pretrain CNN/transformer components on broader biomedical spectral data.
  - Fine-tune on the project-specific dataset.
- **Drift detection and adaptive learning rates:**
  - Detect distribution shifts and trigger controlled updates.

**Expected impact**
- Better adaptation to new bacterial/fungal profiles.
- Reduced need for frequent full retraining.
- Improved stability across datasets and time.

**Challenges**
- Avoiding knowledge erosion on older classes.
- Maintaining a reliable stream of validated new samples.

---

## Reproducibility & Suggested Repo Structure

A practical structure for strong provenance and reproducible research:

```text
.
├── data/
│   ├── raw/                 # raw spectra + labels (often not committed)
│   ├── interim/             # intermediate artifacts
│   └── processed/           # features / splits / normalized data
├── notebooks/               # EDA, t-SNE, error analysis
├── src/
│   ├── preprocessing/       # interpolation, normalization, augmentation
│   ├── features/            # spectral + statistical features, PCA, RFE
│   ├── models/              # CNN, XGBoost/LGBM, ensembles, late fusion
│   ├── training/            # training loops, callbacks, logging
│   └── evaluation/          # metrics, confusion matrices, misclass logs
├── experiments/
│   ├── v001/ ... v085/      # configs, logs, checkpoints, notes per version
│   └── registry.csv         # run registry (seed, split, params, metrics)
├── reports/
│   ├── figures/             # plots: t-SNE, ROC, confusion matrices
│   └── summaries/           # version summaries, milestone reports
├── requirements.txt
├── README.md
└── LICENSE
