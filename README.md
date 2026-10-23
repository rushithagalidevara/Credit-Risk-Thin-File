# Credit Risk Modeling for Thin-File Borrowers

Can we fairly predict loan repayment for people with little or no credit history?

This project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset, which focuses on applicants underserved by traditional credit bureaus, to build and evaluate credit risk models, measure how much predictive power remains *without* bureau-style data, and test the model for fairness.

## Highlights

| Model | Test AUC | Test Gini | KS |
|---|---|---|---|
| Logistic regression (baseline) | 0.752 | 0.504 | 0.374 |
| LightGBM | 0.769 | 0.538 | 0.402 |
| LightGBM, no bureau data | 0.716 | 0.433 | 0.322 |

- **Thin-file finding:** with all bureau-style variables removed, the model retains about **80%** of the full model's predictive power (Gini 0.433 vs 0.538). Application, affordability, employment and behavioral information still meaningfully separate good and risky borrowers.
- **Simple models go far:** LightGBM improves on logistic regression by only +0.034 Gini, so a transparent model captures most of the signal.
- **Top risk drivers:** external credit scores (`EXT_SOURCE_*`), organization type, and engineered affordability features. Five of the top 20 features were engineered in this project, including credit term (#3) and credit-to-goods ratio (#5).
- **Fairness (approve the safest 80%):**
  - *Gender* (not a model input): women approved 83.1% vs men 74.1%, largely tracking actual default rates (7.0% vs 10.2%). However, men who would have repaid were falsely rejected 22.2% of the time vs 14.5% for women.
  - *Age:* the largest disparity. Applicants under 30 who would have repaid were falsely rejected **30.7%** of the time, versus **5.7%** for those 60+. Young applicants are often exactly the thin-file population that financial inclusion aims to serve.
- **Removing age features made the age gap worse, not better.** Retraining without explicit age features cost almost no accuracy (Gini -0.005), but the false rejection gap across age groups *widened* from 25.0 to 27.7 points, and false rejections for applicants under 30 rose from 30.7% to 33.7%. Features strongly correlated with age act as proxies, including the employment anomaly flag (0.62), the employer phone flag (0.62), the external score `EXT_SOURCE_1` (0.60), employment length (0.36) and registration history (0.32). Simply hiding a sensitive variable ("fairness through unawareness") does not remove bias and can even worsen it.

## What's in the notebook

1. Data exploration and cleaning (including the `DAYS_EMPLOYED` anomaly)
2. Affordability and stability features (credit-to-income, annuity-to-income, credit term, employment stability)
3. Logistic regression baseline and LightGBM challenger with early stopping
4. **Thin-file experiment:** model performance with bureau-style variables removed
5. Calibration curve and risk-decile table
6. Explainability with feature importance and SHAP
7. Fairness analysis by gender and age group under a simulated approval policy
8. Conversion of probabilities to a points-based credit score (base 600 at 50:1 odds, PDO 20)

## Why this matters

About 1.3 billion adults worldwide have no financial account, and many more have no credit history (World Bank Global Findex 2025). Lenders that cannot measure risk exclude these people. Models that use alternative data, built and tested for fairness, can expand access to affordable credit.

## How to run

```bash
pip install pandas numpy scikit-learn lightgbm shap matplotlib
```

1. Download `application_train.csv` from the Kaggle competition's Data tab (accept the rules first).
2. Place it in `data/application_train.csv`.
3. Open `credit_risk_home_credit.ipynb` and run all cells.

The dataset is not included in this repository, in line with Kaggle's competition rules.

## Limitations and next steps

- No application dates, so validation is a random split rather than out-of-time.
- Only approved loans have outcomes; reject inference is not addressed.
- Fairness tested on two attributes only.
- With a single approval cutoff, groups with higher average default rates see higher false rejection rates. Reducing the age gap without hurting accuracy is the main open question.
- Next: test fairness-aware training and threshold methods (where legally permitted), add alternative data that gives young applicants a track record (for example bureau and previous-application tables as a stand-in), and try monotonic constraints.

## Author

Rushitha Galidevara · MS Finance (Investments and Quantitative Analysis) · [LinkedIn](https://www.linkedin.com/in/rushithagalidevara)
