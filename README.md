#  Euro Area Sovereign Yield Curve Modeling & Stress Testing

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Data Source](https://img.shields.io/badge/Data-ECB%20Data%20Portal%20API-003399?logo=europeanunion&logoColor=white)
![Domain](https://img.shields.io/badge/Domain-Quantitative%20Finance%20%26%20Central%20Banking-green)
![License](https://img.shields.io/badge/License-MIT-orange)

An end-to-end quantitative data science pipeline developed in Python to extract, fit, analyze, and stress-test the Euro Area AAA sovereign spot yield curve using real-time official data from the European Central Bank (ECB).

---

##  Executive Summary

Analyzing the term structure of interest rates is vital for central bank monetary policy design, macroeconomic forecasting, and balance sheet risk management. 

This project delivers an automated quantitative framework that:
* **Ingests Live Data:** Connects directly to the **ECB Data Portal REST API** to retrieve daily zero-coupon AAA spot yields across 10 benchmark maturities (3M to 30Y).
* **Parametric Modeling:** Implements the **Nelson-Siegel (1987)** model via non-linear optimization (*Nelder-Mead*) for continuous curve fitting.
* **Dimensionality Reduction:** Applies **Principal Component Analysis (PCA)** to demonstrate that 3 orthogonal factors capture **99.96%** of Eurozone yield dynamics.
* **Macro Stress Testing:** Simulates yield curve reactions under policy shock scenarios (*Hawkish Shift* & *Bear Flattening*).

---

## Key Empirical Findings

| Metric / Analysis | Quantitative Result | Financial / Economic Interpretation |
| :--- | :---: | :--- |
| **PC1 (Level Factor)** | **97.42%** | Captures overall parallel shifts in long-term macroeconomic expectations. |
| **PC2 (Slope Factor)** | **2.34%** | Captures short vs. long term yield spread dynamics (monetary policy stance / curve inversion). |
| **PC3 (Curvature Factor)** | **0.20%** | Captures medium-term hump effects (intertemporal expectations). |
| **Total Variance Explained** | **99.96%** | **Proves empirically** that 3 latent factors govern Euro Area sovereign yields. |

---

##  Quantitative Methodology

```text
┌────────────────────────┐      ┌──────────────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐
│ ECB Data Portal API    │ ───> │ Data Cleaning & Align    │ ───> │ Nelson-Siegel Fitting   │ ───> │ PCA & Stress Testing   │
│ (AAA Sovereign Yields) │      │ (10 Benchmark Maturities)│      │ (Nelder-Mead Optimizer) │      │ (Scenario Analysis)    │
└────────────────────────┘      └──────────────────────────┘      └─────────────────────────┘      └────────────────────────┘

## the Nelson-Siegel Model
The zero-coupon spot yield $y(\tau)$ for maturity $\tau$ is defined as:
              $$y(\tau) = \beta_0 + \beta_1 \left( \frac{1 - e^{-\tau/\tau_0}}{\tau/\tau_0} \right) + \beta_2 \left( \frac{1 - e^{-\tau/\tau_0}}{\tau/\tau_0} - e^{-\tau/\tau_0} \right)$$

beta zero (level): Long-term asymptotic interest rate level.
beta one(slope): Short-to-long term spread (governs yield curve inversion).
beta two(curvature): Medium-term hump magnitude.
tau zero(scale factor): Set to 2.0 (standard decay rate).

## Macroeconomic Stress Scenarios
Scenario 1 - HAWKINS PARALLEL SHIFT (+100 BPS): Models aggressive ECB policy rate hikes across all maturities in response to an inflation shock.
Scenario 2 - BEAR FLATTENING: Short-term rates surge (+150 BPS at 3M) while long-term rates remain anchored (+25 BPS at 30Y) ,modeling severe short-end liquidity tightening.


ecb-yield-curve-modeling/
│
├── main.py              # Complete Python execution pipeline (API, Model, PCA, Plots)
└── README.md            # Project documentation & quantitative findings



## Stack and Dependencies
Language: Python 3.10
Data Retrieval: requests (ECB RESTful API integration)
Data Processing: Pandas and Numpy
Quantitative & ML: scipy.optimize (Nelder-Mead), scikit-learn (PCA)
Visualization: matplotlib

## How to run

git clone [https://github.com/YOUR_USERNAME/ecb-yield-curve-modeling.git](https://github.com/YOUR_USERNAME/ecb-yield-curve-modeling.git)
cd ecb-yield-curve-modeling

## Author
Tommaso Grattacagio DATA SCIENCE FOR ECONOMICS STUDENT

## How to do previous command
git clone:download a copy of the repository on your computer
pip install: install required Python libraries(pandas,scipy,etc)
python main.py: run the script
 





