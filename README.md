# Threshold Dialectics: An Early Warning Signal Simulation Study

This repository contains the Python simulation and analysis code for a study on early warning signals (EWS) for system collapse, based on the theoretical framework of **Threshold Dialectics (TD)**. The simulation models a complex adaptive system, subjects it to various collapse scenarios, and evaluates the performance of a large suite of traditional and TD-specific EWS.

This work is the computational foundation for the concepts explored in the book, *"Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness"*.

> **Core Argument:** Systemic collapse is not primarily a scalar-threshold problem but a **coupled-velocity** problem: observe how fast rigidity ($\beta$) and slack ($F_{crit}$) drift *together*, and you can anticipate failure in time; watch only individual levels, and you will likely act too late.

## Table of Contents
- [Overview](#overview)
- [Key Concepts](#key-concepts)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Interpreting the Results](#interpreting-the-results)
- [Citation](#citation)
- [License](#license)

## Overview

This project simulates a complex adaptive system governed by the principles of Threshold Dialectics. The system's viability is determined by a dynamic **Tolerance Sheet** ($\Theta_T$), which is a function of three core adaptive "levers." When systemic "strain" exceeds this tolerance, the system collapses.

The primary goal of this simulation is to test the efficacy of various EWS in predicting these collapses across different scenarios. It compares traditional EWS (like variance and autocorrelation) with a novel suite of TD-specific EWS derived from the dynamics of the core levers, including an advanced **Ensemble EWS** that combines the most robust signals.

## Key Concepts

A brief glossary of core concepts from Threshold Dialectics used in this simulation:

- **Adaptive Levers:** The three core capacities a system manages to maintain viability:
    - **Perception Gain ("g"):** The system's sensitivity or responsiveness to new information or prediction errors.
    - **Policy Precision ("β"):** The rigidity or confidence with which the system adheres to its current operational rules or models.
    - **Energetic Slack ("F_crit"):** The system's buffer of resources (energy, capital, etc.) available to absorb shocks and fuel adaptation.

- **Tolerance Sheet ($\Theta_T$):** A dynamic, multi-dimensional boundary representing the system's maximum capacity to withstand strain. It is calculated as $\Theta_T = C \cdot g^{w_1} \cdot \beta^{w_2} \cdot F_{crit}^{w_3}$. Collapse occurs when strain exceeds $\Theta_T$.

- **TD Diagnostics (EWS):**
    - **Speed Index ("Speed"):** Measures the joint rate of change (velocity) of the "β" and "F_crit" levers. High speed indicates rapid structural drift.
    - **Couple Index ("Couple"):** Measures the correlation between the velocities of "β" and "F_crit". Detrimental coupling patterns (e.g., rising rigidity and falling slack) dramatically increase risk.
    - **H_UHB:** A composite indicator inspired by "Universal Horizon Bound" theories, formulated as "Speed² / (1 - Couple)". It is designed to capture the amplified risk from high speed combined with detrimental coupling.

- **Ensemble EWS ("Ensemble_AvgScore_Top4"):** A robust EWS created by averaging the scaled scores of the four top-performing TD-specific indicators ("Speed", "FACR", "RMA_norm", "H_UHB"). This approach mitigates the risk of any single indicator failing in a specific scenario.

## Features

- **Complex System Simulation:** A highly parameterized simulation engine ("simulate_system_run") models the core dynamics of Threshold Dialectics.
- **Multiple Collapse Scenarios:** The study includes 8 distinct, pre-configured collapse scenarios (e.g., "Resource Exhaustion," "Shock-Induced," "Lever Instability") and a stable baseline scenario.
- **Comprehensive EWS Calculation:** The simulation calculates over 20 different EWS, including traditional statistical measures and a large suite of novel TD-specific indicators.
- **Robust Performance Evaluation:** The script uses 5-fold stratified cross-validation to assess the performance of each EWS, calculating the Area Under the ROC Curve (AUC) and lead time statistics.
- **Automated Result Generation:** Automatically generates a detailed Excel report, a summary text file, and publication-quality plots (ROC curves, heatmaps) in the "results/" directory.

## Installation

To set up the project, follow these steps. A Python version of 3.8 or higher is recommended.

1.  **Clone the repository:**
    """bash
    git clone https://github.com/your-username/threshold-dialectics.git
    cd threshold-dialectics
    """

2.  **Create and activate a virtual environment (recommended):**
    """bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    """

3.  **Install the required packages:**
    A "requirements.txt" file should be created with the following content.

    """
    # requirements.txt
    numpy
    pandas
    matplotlib
    scikit-learn
    seaborn
    openpyxl
    """

    Install the packages using pip:
    """bash
    pip install -r requirements.txt
    """

## Usage

The simulation is run from the command line. The main script (provided as "simulation_study.py" in this example) can be executed directly.

**To run the full simulation and generate all outputs:**
"""bash
python simulation_study.py
"""
This will run all scenarios, perform the analysis, and save the results. This can take a significant amount of time.

**Command-Line Arguments:**

-   "--quick": Runs the simulation without generating the "matplotlib" plots. This is useful for faster execution when you only need the data output.
    """bash
    python simulation_study.py --quick
    """
-   "--debug": In addition to the standard outputs, this flag writes per-scenario summary CSV files to the "results/" directory, which can be useful for detailed analysis of a specific collapse type.
    """bash
    python simulation_study.py --debug
    """

## Project Structure

"""
.
├── simulation_study.py     # The main Python script for simulation and analysis
├── README.md               # This file
└── requirements.txt        # Python package dependencies
"""

After a successful run, a "results/" directory will be created:

"""
.
└── results/
    ├── EWS_summary.xlsx        # Detailed Excel report with all metrics
    ├── summary.txt             # Text file with key summary tables
    ├── auc_heatmap.png         # Heatmap of Mean AUC scores
    ├── leadtime_heatmap.png    # Heatmap of Mean Lead Times
    ├── roc_curves_*.png        # ROC curves for each scenario
    └── example_*.png           # Example time-series plots for selected scenarios
"""

## Interpreting the Results

The primary outputs are located in the "results/" directory. The goal is to identify EWS that consistently provide high predictive power (AUC) and long lead times across different scenarios.

1.  **Start with the Heatmaps:** "auc_heatmap.png" is the best starting point. Brighter colors indicate a higher AUC, meaning the EWS is better at distinguishing between pre-collapse and stable states. Look for rows that are consistently bright across all scenarios.

2.  **Consult the Summary File:** "EWS_summary.xlsx" (or "summary.txt") provides the quantitative data. The "Overall\_Summary" sheet/section gives the mean performance of each EWS across all collapse types.

3.  **Key Findings from the Included Example "summary.txt":**
    -   **TD-Specific EWS Outperform Traditional EWS:** Notice that indicators like "Speed", "FACR", and "RMA_norm" have a Mean AUC of 1.0, while traditional indicators like "AR1_Y" have an AUC of ~0.47 (worse than random).
    -   **The Ensemble is Most Robust:** The "Ensemble_AvgScore_Top4" achieves a perfect Mean AUC of 1.0 and provides one of the longest overall lead times. This is the key takeaway: combining strong, mechanism-based signals yields superior and more reliable early warnings.
    -   **Context Matters:** Some strong individual indicators (like "H_UHB") may fail in specific scenarios (e.g., "Resource Exhaustion Collapse"). This highlights the strength of the ensemble approach.

| EWS Name                     | Mean AUC (Overall) | Mean Lead Time (s) |
| ---------------------------- | ------------------ | ------------------ |
| "Variance_Y" (Traditional)   | 0.706              | 72.2               |
| "Speed" (TD)                 | 1.000              | 62.4               |
| **"Ensemble_AvgScore_Top4"** | **1.000**          | **81.3**           |

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
