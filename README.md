# GPP Prediction and Compound Drought Mechanism Analysis Code

This repository contains MATLAB code used for GPP prediction, PSO-LSTM-Transformer model training, post-training evaluation, SHAP-based mechanism analysis, and drought-recovery analysis for the Huai River Basin, China.

The processed input data required by these scripts are archived separately on Zenodo:

https://doi.org/10.5281/zenodo.20306852

## Repository contents

```text
code/
  train/
    main_workflow_huaih.m          Main PSO-LSTM-Transformer training workflow.
    run_pso_optimization.m         PSO hyperparameter-search wrapper.
    PSO.m                          Particle swarm optimization implementation.
    ys.m                           PSO particle-position initialization helper.
  after_train/
    chek_withpso_v5.m              Post-training prediction and model-performance evaluation.
    SHAP_Spatial_Analysis.m        Global SHAP importance and beeswarm-style analysis.
    SHAP_Spatial_GEO.m             Spatial SHAP analysis and driver-regime export.
    shapley_lstm_3d.m              Leave-one-feature-out SHAP approximation for sequence models.
    dry_recover_v1.m               Lightweight drought-recovery-time calculation.
    dry_recover_reason.m           Multi-factor drought-recovery mechanism analysis.
models/
  Final_Model_Huaihe_v5.mat        Trained model artifact and normalization parameters.
data/
  README_place_zenodo_files_here.txt
outputs/
  .gitkeep
```

## Data preparation

Download the Zenodo dataset and place the following files under `data/`:

```text
Ready_for_Train_3D_Huaihe_v5.mat
Pixel_Data_GPP.mat
Pixel_Data_petPM.mat
Pixel_Data_soil.mat
Pixel_Data_spei.mat
Pixel_Data_tmpmax.mat
Pixel_Data_vpd.mat
```

The main scripts use relative paths derived from the script location, so they do not require user-specific Windows paths.

## MATLAB requirements

The scripts were prepared for MATLAB workflows and require toolboxes that provide:

- Deep learning layers and training functions, including `trainNetwork`, `sequenceInputLayer`, `lstmLayer`, `selfAttentionLayer`, `positionEmbeddingLayer`, and `trainingOptions`.
- `mapminmax` normalization functions.
- Statistics functions such as `knnsearch`, `prctile`, and `histcounts`.
- Optional parallel execution for `parpool` and `parfor`.
- Optional Mapping Toolbox functions for boundary display in `dry_recover_v1.m`; the script still runs without the optional shapefile.

GPU execution is optional. Training and prediction scripts use `ExecutionEnvironment`, and most post-training prediction code falls back to CPU when GPU execution fails.

## Recommended run order

1. Place the Zenodo `.mat` input files in `data/`.
2. Run `code/train/main_workflow_huaih.m` to train the PSO-LSTM-Transformer model and write `models/Final_Model_Huaihe_v5.mat`.
3. Alternatively, use the included `models/Final_Model_Huaihe_v5.mat` to skip retraining and run the post-training scripts directly.
4. Run `code/after_train/chek_withpso_v5.m` for prediction, global metrics, spatial-error analysis, and best-pixel diagnostics.
5. Run `code/after_train/SHAP_Spatial_Analysis.m` for global SHAP ranking and feature-contribution visualization.
6. Run `code/after_train/SHAP_Spatial_GEO.m` for spatial SHAP maps and driver-regime exports under `outputs/`.
7. Run `code/after_train/dry_recover_v1.m` to calculate lightweight drought-recovery-time outputs.
8. Run `code/after_train/dry_recover_reason.m` after step 7 to analyze environmental gradients associated with recovery time.

## Reproducibility notes

- The scripts set `rng(42)` where stochastic sampling or optimization is used.
- The data archive provides monthly, 1-km, MATLAB-formatted model-ready inputs for 2001-01 to 2022-12.
- The included trained model file contains the trained network and normalization parameters used by the post-training evaluation and SHAP scripts.
- Large processed input data are not duplicated in this code package; use the Zenodo DOI above.

## Citation

If using the processed input data, cite:

Chen, L., & Ning, S. (2026). Processed Monthly 1-km GPP and Hydroclimatic Dataset for Compound Drought Mechanism Analysis in the Huai River Basin China (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20306852

For the code, cite the archived software DOI once a software release DOI has been created.
