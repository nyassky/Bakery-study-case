# Contextual Factors in Retail Demand
### Author
**Ranis Vakhitov** — 3rd-year student, Peter the Great St. Petersburg Polytechnic University (SPbPU)

### A Case Study of Bakery Sales: SARIMA + Random Forest Analysis of Temporal vs. Product-Driven Demand
---

## About the Project

A widespread assumption in small-format retail holds that product assortment is the primary lever for revenue optimization. This project tests that assumption empirically using transaction records from a South Korean bakery.

SARIMA is used for time-series decomposition and Random Forest Regression for feature importance analysis, comparing the predictive power of **temporal context** (hour of day, day of week) against **product features** (sales volume by item).

**Dataset:** [Bakery Sales Dataset (South Korea)](https://www.kaggle.com/datasets/hosubjeong/bakery-sales) — timestamped POS records with item names, quantities, and store location.

**Preprocessing:** Records with missing timestamps or null item fields are removed. Hour of day and day of week are extracted from each transaction timestamp. The categorical `place` attribute is integer-encoded for use in tree-based models. A cluster of atypical late-night (after 10 PM) transactions with unusually high per-transaction values was identified and flagged separately (present on fewer than 3% of days).

---

## Research Questions

1. Do temporal features predict daily revenue more reliably than product features?
2. Do atypical late-night transactions introduce meaningful bias into standard forecasting approaches?

---

## Methodology

**SARIMA** (seasonal period s=7) is fit to the daily aggregated sales series to characterize weekly cyclicity and provide a linear forecast baseline.

**Random Forest** (200 trees, unconstrained depth) is trained on the full feature set — temporal, locational, and product dummies — to predict daily revenue. Random Forest is preferred over gradient boosting here for its more stable feature attribution on small transaction logs. Feature importances (mean impurity decrease, normalized) are the primary output, enabling direct comparison of temporal versus product contributions to revenue variance.

Model evaluation compares forecasts against a withheld test set, with visual inspection of time-series alignment to assess how each model captures periodic demand.

---

## Results

| Model | Metric | Result |
|---|---|---|
| SARIMA | Weekly seasonality | Captured with stable error margin; smooths over intra-week spikes |
| Random Forest | R² | **0.89** |

---

## Conclusion

Revenue variance in this bakery is driven primarily by **inter-year growth** and the sales volume of its **dominant product** — not by assortment breadth. Once flagship product demand is accounted for, **hour of day** is the strongest remaining predictor, ranking above all other product categories. This means *when* customers arrive predicts total revenue better than *what else* is on the menu.

A small number of late-night transactions further complicate planning: they inflate the daily mean enough to bias inventory estimates. Switching to trimmed-mean or median-based estimation would correct this at no added operational cost.

**Limitations:** This study covers a single location over a short time window. Multi-year, multi-location data would help disentangle the trend and product effects observed here more cleanly.

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/nyassky/Bakery-study-case.git
cd Bakery-study-case

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas scikit-learn statsmodels matplotlib jupyter
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/hosubjeong/bakery-sales) and place it in the project root.

---

## Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-2C3E50)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243)
