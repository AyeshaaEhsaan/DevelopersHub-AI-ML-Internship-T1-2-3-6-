# AI/ML Engineering Internship — DevelopersHub Corporation

This repository contains my work from the AI/ML Engineering track of the DevelopersHub Corporation Internship Program. Each task involved independent research, hands-on implementation, and drawing conclusions from real datasets — not just running code, but actually understanding what the results mean.

---

## Task 1 — Exploratory Data Analysis on the Iris Dataset
**Notebook:** `iris_eda_task1.ipynb`

The Iris dataset is a classic starting point in machine learning, but I wanted to go beyond just plotting it. The goal here was to understand *why* certain features matter more than others for distinguishing flower species.

Through pairplots, histograms, box plots, and a correlation heatmap, a clear pattern emerged: petal length and petal width are by far the most separable features across species. Setosa is completely isolated from the other two — you could classify it with a single threshold. Versicolor and Virginica, however, overlap enough that you'd need a proper model to separate them reliably.

The sepal width was the most surprising finding — it actually has the least discriminating power, even though intuitively "width" sounds like it should matter. This kind of insight is exactly what EDA is for: checking assumptions before modeling.

**Tools:** `pandas`, `seaborn`, `matplotlib`

---

## Task 2 — Stock Price Prediction (AAPL)
**Notebook:** `stock_price_prediction_task2.ipynb`

Predicting stock prices is one of those problems that sounds straightforward until you actually try it. I used three years of Apple (AAPL) daily trading data fetched live from Yahoo Finance via `yfinance`, and set up a next-day closing price prediction problem.

One important design decision: time series data must *not* be shuffled before splitting. The train set is the older 80% of the data; the test set is the most recent 20%. Shuffling would leak future information into training, making results look far better than they really are.

I built features that traders actually use — 5-day and 20-day moving averages, daily price range, and intraday momentum — then compared Linear Regression against Random Forest. Random Forest won by a clear margin, which makes sense: stock price relationships are non-linear and Random Forest handles that naturally.

The honest takeaway: even a good model struggles during sudden market events. The error spikes exactly where there's volatility — earnings calls, macro news. That's not a flaw in the model; it's just the nature of financial data.

**Tools:** `yfinance`, `scikit-learn`, `pandas`, `matplotlib`

---

## Task 3 — Heart Disease Prediction
**Notebook:** `heart_disease_prediction_task3.ipynb`

This was the task I found most meaningful, because the stakes are real. The dataset comes from the UCI ML Repository — 303 patients with 13 clinical features, and a binary label indicating presence or absence of heart disease.

I trained both Logistic Regression and a Decision Tree, and evaluated them not just on accuracy but on ROC-AUC and confusion matrices. In a medical context, accuracy alone is misleading — a false negative (missing a sick patient) is far more dangerous than a false positive (flagging a healthy one). So recall for the disease class was the metric I cared about most.

The most influential features were chest pain type, thalassemia, and ST depression on exercise — which aligns with what cardiologists actually look for. Interestingly, maximum heart rate achieved had an *inverse* relationship with disease risk: patients with lower peak heart rates were more likely to have heart disease, not less. That was counterintuitive at first, but it makes sense physiologically — a diseased heart has reduced capacity.

Logistic Regression hit ~85% accuracy with a ROC-AUC of ~0.91, which is strong for a linear model on clinical data. The Decision Tree was slightly lower but far more explainable — every split maps to a clinical decision, which matters when a doctor needs to understand why.

**Tools:** `scikit-learn`, `pandas`, `seaborn`, `matplotlib`

---

## Task 6 — House Price Prediction
**Notebook:** `house_price_prediction_task6.ipynb`

I used the California Housing dataset (built into scikit-learn) to predict median house values across census blocks. The first thing that stood out in EDA was the geographic plot — when you map price by latitude/longitude, California's expensive coastal corridor is immediately visible. Location isn't just one feature; it's doing a lot of the heavy lifting.

I engineered two additional features: bedroom ratio (bedrooms as a fraction of total rooms) and rooms per person (spaciousness per occupant). Both added signal beyond what the raw features provided.

For modeling, I wrapped everything in scikit-learn Pipelines — scaling and model bundled together — which prevents a subtle but common mistake: fitting the scaler on the full dataset before splitting, which leaks test data statistics into training.

Ridge Regression gave a decent R² of ~0.62 but clearly struggled with the non-linear relationship between location and price. Gradient Boosting jumped to ~0.84 R², with median income and location (latitude/longitude) as the top three features by importance. The "location, location, location" rule in real estate turned out to be statistically provable.

Both models underperformed at the high end (above $400k), partly because the dataset has a hard cap at $500,001 — a known quirk that compresses the upper price range and limits what any model can learn there.

**Tools:** `scikit-learn`, `pandas`, `seaborn`, `matplotlib`

---

## How to Run

All notebooks are designed for **Google Colab** — no local setup or file downloads required.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → GitHub tab → paste this repo URL
3. Select the notebook you want to run
4. Runtime → Run all

Each notebook installs its own dependencies in the first cell and loads data directly from public sources.

---

*DevelopersHub Corporation — AI/ML Engineering Internship*
