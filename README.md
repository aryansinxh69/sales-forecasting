
# Sales Forecasting — Superstore Furniture Dataset

A machine learning project that predicts the sales amount of a furniture order using Linear Regression, built as part of an internship assignment.

---

## Dataset

**File:** `stores_sales_forecasting.csv` — 2,121 real Furniture orders from a US Superstore

| Feature | Type |
|---|---|
| Category | Text |
| Sub-Category | Text |
| Region | Text |
| Ship Mode | Text |
| Segment | Text |
| Quantity | Number |
| Discount | Number |
| Profit | Number |
| **Sales** | Target |

> Dataset contains Furniture orders only — Sub-Categories: Bookcases, Chairs, Furnishings, Tables.

---

## Project Structure

```
sales-forecasting/
│
├── sales_forecasting.ipynb        # Main notebook
├── stores_sales_forecasting.csv   # Dataset (keep in same folder)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook sales_forecasting.ipynb
```

Then run **Kernel → Restart & Run All**.

---

## What It Does

1. Loads and previews the CSV
2. EDA — sales distribution, category/region trends, correlation heatmap
3. Label encodes all text columns and saves mappings
4. Splits data 80% training / 20% testing
5. Trains a `LinearRegression` model on 8 features
6. Evaluates using MAE, RMSE, and R² Score
7. Predicts sales for a custom new order

---

## Making a Prediction

Edit these values in the final notebook cell:

```python
new_category    = 'Furniture'     # Only available category
new_subcategory = 'Chairs'        # Bookcases / Chairs / Furnishings / Tables
new_region      = 'West'          # Central / East / South / West
new_shipmode    = 'First Class'   # First Class / Same Day / Second Class / Standard Class
new_segment     = 'Corporate'     # Consumer / Corporate / Home Office
new_quantity    = 5
new_discount    = 0.1
new_profit      = 120.0
```

---

## Results

| Metric | Description |
|---|---|
| MAE | Average dollar error per prediction |
| RMSE | Penalises large errors more than MAE |
| R² Score | % of sales variation explained by the model |

R² may appear low due to outlier orders in the dataset (range: $1.89 – $4,416). This is a known limitation of Linear Regression on skewed data.

---

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn

---

