# 🎓 Data Science Research Capstone & Whitepaper

[![Paper](https://img.shields.io/badge/Whitepaper-PDF%20(8%20Pages)-red.svg)](Data_Science_Research_Capstone_Whitepaper.pdf)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](notebooks/capstone_predictive_modeling.ipynb)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()
[![Evaluation](https://img.shields.io/badge/Mentor-Evaluation%20Ready-green.svg)]()

> **Capstone Deliverables Submitted:**
> 1. 📄 **Formal Research Report / Whitepaper (PDF):** [`Data_Science_Research_Capstone_Whitepaper.pdf`](Data_Science_Research_Capstone_Whitepaper.pdf) *(8-Page Academic Report complete with Abstract, Literature Review, Methodology, ADF Stationarity Proofs, Holdout Evaluation, 30-Day Projections, and Supply Chain Business Implications)*.
> 2. 📓 **Predictive Modeling Code Repository:** [`notebooks/capstone_predictive_modeling.ipynb`](notebooks/capstone_predictive_modeling.ipynb) *(Complete executable workflow with pre-rendered decomposition plots and forecast projections)*.
> 3. 🐍 **Production Python Pipeline:** [`src/predictive_modeling_pipeline.py`](src/predictive_modeling_pipeline.py).

---

## 📌 1. Research Question Formulation
> **Research Question (RQ):**
> *"How do deterministic linear trends and high-amplitude weekly promotional events induce non-stationarity in retail daily demand series, and which predictive modeling architecture (SARIMAX vs. Facebook Prophet) achieves optimal out-of-sample forecast accuracy to minimize safety stock holding costs and stockout risk?"*

### Sub-Questions:
- **RQ1 (Stationarity & Differencing):** Does first-order differencing ($d=1$) sufficiently eliminate unit roots caused by sustained upward drift, as verified by the Augmented Dickey-Fuller (ADF) test?
- **RQ2 (Model Performance):** Between classical statespace autoregression (SARIMA) and Bayesian generalized additive models (Prophet), which model yields lower generalization error on holdout promotional shocks?
- **RQ3 (Business & Economic Impact):** What are the quantitative safety stock reductions and holding cost savings unlocked by achieving a sub-1% MAPE forecast pipeline?

---

## 📄 2. Formal Research Whitepaper Structure
The included 8-page research paper ([`Data_Science_Research_Capstone_Whitepaper.pdf`](Data_Science_Research_Capstone_Whitepaper.pdf)) contains:
1. **Executive Abstract & Keywords** (250-word synthesis of research problem, findings, and supply chain relevance).
2. **Section 1: Introduction & Research Problem Formulation** (The bullwhip effect, retail inventory challenges, and formal research questions).
3. **Section 2: Literature Background & Theoretical Foundations** (Box-Jenkins ARIMA/SARIMA, additive decomposition, and generalized additive models).
4. **Section 3: Data Methodology & Exploratory Data Analysis** (Dataset schema, descriptive statistics, and weekly promotional dynamics).
5. **Section 4: Statistical Diagnostics & Stationarity Analysis** (Augmented Dickey-Fuller test formulation, unit-root rejection, and ACF/PACF plots).
6. **Section 5: Predictive Modeling Architecture & Hyperparameter Tuning** (Grid search optimization over $(p, d, q) \times (P, D, Q)_7$ minimizing AIC).
7. **Section 6: Empirical Results & Holdout Performance Evaluation** (80/20 train-test split benchmark, RMSE, MAPE, and MAE comparison).
8. **Section 7: 30-Day Out-of-Sample Horizon Projections** (July 1 to July 30, 2026 forecast table with dynamic 95% confidence intervals).
9. **Section 8: Supply Chain Implications & Economic Impact Analysis** (Safety stock dimensioning formula, reorder point optimization, and cost savings).
10. **Section 9 & 10: Limitations, Future Work, Conclusion & Academic References**.

---

## 📊 3. Key Empirical Findings & Metrics

| Dimension | Finding | Statistical / Operational Value |
| :--- | :--- | :--- |
| **Historical Scope** | 30 Continuous Daily Records (June 2026) | Mean: 142.10 units (Min: 118, Max: 174) |
| **Decomposition** | Additive Superposition: $Y_t = T_t + S_t + R_t$ | Trend: +1.03 units/day; Seasonality: Sat +21.15, Fri +7.90 |
| **Raw Series ADF Test** | Fails to reject unit-root null hypothesis ($H_0$) | Statistic: -2.6436, $p$-value: **0.0843** (Non-Stationary) |
| **Differenced ADF Test** | Successfully rejects $H_0$ at 99% confidence | Statistic: -4.0110, $p$-value: **0.0013** (Stationary) |
| **Selected Model** | Optimal SARIMAX Specification | **SARIMA(1, 1, 0) × (0, 1, 1)₇** (AIC: 68.42) |
| **Holdout Test RMSE** | Unseen 6-day evaluation (June 25–30) | **1.57 units** (~87% error reduction over baselines) |
| **Holdout Test MAPE** | Unseen 6-day evaluation (June 25–30) | **0.88%** (Sub-1% generalization error) |
| **Safety Stock Reduction**| Lead time $L=3$ days, 95% service level | **86.1% reduction** in required buffer stock (36 $\rightarrow$ 5 units) |

---

## 📈 4. Visual Evidence & Research Figures

### Figure 1: Time-Series Additive Decomposition (Period = 7 Days)
![Time Series Decomposition](plots/02_time_series_decomposition.svg)

### Figure 2: Stationarity Analysis (Raw vs First-Differenced ADF Comparison)
![Stationarity and Differencing](plots/03_stationarity_and_differencing.svg)

### Figure 3: Holdout Performance Evaluation vs Ground Truth (RMSE: 1.57, MAPE: 0.88%)
![Holdout Performance Evaluation](plots/04_model_evaluation_holdout.svg)

### Figure 4: 30-Day Out-of-Sample Projections (July 2026 with 95% Confidence Intervals)
![30-Day Forecast Projections](plots/05_30_day_forecast_projections.svg)

---

## 📁 5. Repository Directory Architecture
```
data-science-research-capstone/
├── Data_Science_Research_Capstone_Whitepaper.pdf   # Formal 8-page Capstone Research Report
├── Data_Science_Research_Capstone_Whitepaper.md    # Full academic text in Markdown format
├── README.md                                       # Capstone documentation and executive brief
├── requirements.txt                                # Python library environment dependencies
├── .gitignore                                      # Standard Python / Jupyter gitignore rules
├── data/
│   ├── daily-demand-series.csv                     # Historical input data with event indicators
│   └── demand_forecast_30_days.csv                 # Generated 30-day projection dataset
├── notebooks/
│   └── capstone_predictive_modeling.ipynb          # Jupyter Notebook with executable code & plots
├── src/
│   └── predictive_modeling_pipeline.py             # Modular production modeling script
└── plots/                                          # Vector charts embedded in paper & notebook
    ├── 01_raw_demand_series.svg
    ├── 02_time_series_decomposition.svg
    ├── 03_stationarity_and_differencing.svg
    ├── 04_model_evaluation_holdout.svg
    └── 05_30_day_forecast_projections.svg
```

---

## 🚀 6. Reproducibility & Execution Instructions

```bash
# 1. Clone the repository
git clone https://github.com/<YOUR_USERNAME>/data-science-research-capstone.git
cd data-science-research-capstone

# 2. Set up virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Execute the predictive modeling pipeline
python3 src/predictive_modeling_pipeline.py

# 4. Launch the Jupyter Notebook
jupyter notebook notebooks/capstone_predictive_modeling.ipynb
```
