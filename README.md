# CAHKF Vehicle Acceleration Dataset

**Research data supporting the study:**  
*Calibration-Aware Heteroscedastic Kalman Filtering for Vehicle Acceleration Estimation without External Ground Truth from Monocular Bounding-Box Width*

## Overview

This repository serves as a source of the research dataset used to investigate **vehicle acceleration estimation from monocular vision**. The study estimates vehicle distance from bounding-box width captured by a single webcam and addresses the amplification of measurement noise that occurs when distance measurements are differentiated twice to estimate acceleration.

The data support the evaluation of a **calibration-aware heteroscedastic Kalman filter (CAHKF)**, which models measurement covariance through camera-calibration error propagation, rather than assuming constant measurement noise. A generalized non-uniform **Savitzky–Golay filter** is used as a comparison baseline.

## Data Context

The study considers the following data settings:

| Data setting | Purpose | Acceleration ground truth |
| --- | --- | --- |
| Controlled simulated trajectories | Quantitative evaluation of acceleration estimates under known conditions | Available in simulation |
| On-road monocular observations | Assessment using webcam-based vehicle measurements in real driving conditions | **Not available** (no external reference sensor) |
| Field-informed simulation | Evaluation under conditions informed by the road test | Available from the simulation model |

> **Note:** The table summarizes the data settings in the study; it is not a verified inventory of files in this repository. Consult the uploaded dataset files for their exact contents and formats.

## Research Workflow

1. Observe vehicles using a single monocular camera and obtain bounding-box width measurements.
2. Estimate distance using camera calibration and observed bounding-box width.
3. Propagate calibration uncertainty to construct measurement covariance for the Kalman filter.
4. Estimate vehicle acceleration with the heteroscedastic Kalman filter.
5. Compare simulated-trajectory results against a generalized non-uniform Savitzky–Golay baseline.

## Key Findings and Interpretation

In the controlled simulation with uniform **15 FPS** sampling, the Kalman filter achieved a reported **mean absolute bias of 0.03 m/s²**, compared with **8.8 m/s²** for the baseline using cross-validated windows. With oracle-selected windows, the baseline RMSE was approximately **0.8–2.9 times** the Kalman RMSE.

A **provisional operating range of approximately 25 m** was identified for a **4-second window** under a relative RMSE criterion of **≤25%**.

These findings are **preliminary**. In particular, the on-road test lacked an independent reference sensor; therefore, the reported quantitative accuracy relates to **simulated trajectories**, not directly validated real-world acceleration accuracy.

## Intended Use

This dataset is intended for research on monocular vehicle-motion estimation, camera-calibration uncertainty, noise-aware filtering, and simulation-based comparison of acceleration-estimation methods. Users should distinguish simulated ground-truth data from field observations without reference measurements when interpreting results.

## Citation

If you use this dataset, please cite the associated research study:

*Calibration-Aware Heteroscedastic Kalman Filtering for Vehicle Acceleration Estimation without External Ground Truth from Monocular Bounding-Box Width.*

Full author, venue, year, and DOI information can be added once available.

## Data Availability and License

The exact file inventory, variable definitions, and reuse license should be documented alongside the files uploaded to this repository. **No open-data license is asserted by this README.**
