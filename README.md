# GPP Prediction and Drought Mechanism Analysis Framework
A Spatiotemporal Deep Learning Framework for GPP Prediction and Compound Drought Mechanism Analysis 
# GPP Prediction and Drought Mechanism Analysis 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2023a-blue.svg)](https://www.mathworks.com/products/matlab.html)

Official repository for the paper: A Spatiotemporal Deep Learning Framework for GPP Prediction and Compound Drought Mechanism Analysis in the Huai River Basin.

This repository contains the MATLAB source code for the unified analytical framework integrating the **PSO-LSTM-Transformer** hybrid architecture, **Monte-Carlo SHAP** interpretability, and ecosystem resilience assessment.

## 📖 Overview / 研究概述

In this study, we propose an integrated end-to-end framework to analyze Gross Primary Productivity (GPP) dynamics under drought stress in the climate transition zone (Huai River Basin, China). The workflow comprises four main modules:
1.  **Spatiotemporal Data Preprocessing**: Constructing pixel-based 3D tensors (14 channels, 288 months from 2001-2024).
2.  **Hybrid Deep Learning Model**: A PSO-optimized LSTM-Transformer architecture (`PSO.m`) for accurate long-term GPP prediction.
3.  **Interpretability Framework**: Monte-Carlo sampling-based SHAP analysis to decouple hydrometeorological drivers (e.g., `vpd`, `Ante_prec_5`).
4.  **Resilience Assessment**: Calculating the Standardized GPP Anomaly Index (SGAI) to quantify the Time-to-Recovery (TtR).

## 🗂️ Repository Structure / 项目结构

```text
├── data/                           % Sample data directory / 示例数据目录
│   └── sample_Train_3D.mat         % A small subset of 3D tensor data for testing
├── models/                         % Model architecture / 模型架构核心代码
│   ├── PSO.m                       % Particle Swarm Optimization algorithm
│   └── lstm_transformer_build.m    % Hybrid network construction 
├── scripts/                        % Main execution scripts / 执行脚本
│   ├── check_before_train.m        % Data loading, memory check, and visualization
│   ├── main_train.m                % Main script for model training
│   ├── shap_analysis.m             % Monte-Carlo SHAP calculation
│   └── resilience_analysis.m       % SGAI and TtR calculation
└── README.md
