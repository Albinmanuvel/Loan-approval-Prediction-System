# Loan approval prediction

Decision tree classifier that predicts whether a loan application is approved or rejected. An early learning project, kept as a record.

## Data
`loan_approval_dataset.csv` - 4,269 applications, 13 columns (income, loan amount, term, CIBIL score, asset values, education, self-employment) and `loan_status`.

## What the script does (`DT1.py`)
1. Welch ANOVA and Pearson correlation of each feature against `loan_status`.
2. Drops features with no relationship to approval, keeping the ones that matter (mainly CIBIL score and loan term).
3. 75/25 train/test split (`random_state=42`).
4. `DecisionTreeClassifier` (entropy, max depth 3).

## Results (test set, 1,068 rows)
- Accuracy: 0.955
- Rejected class: precision 0.93, recall 0.95
- Approved class: precision 0.97, recall 0.96

The CIBIL score is by far the strongest signal, which is why a depth-3 tree already does this well. This is a well-known, fairly clean Kaggle-style dataset, so high accuracy is expected.

## Run it
```
pip install -r requirements.txt
python DT1.py
```
