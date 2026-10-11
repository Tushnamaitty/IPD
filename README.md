# IPD — Inequalities in Hospital Care Utilisation: Cesarean Delivery in India

## Current manuscript analysis (updated 11 October 2026)

**Current manuscript question:** Among births in selected government/municipal hospitals (NFHS delivery-place `m15=21`) and private hospitals/maternity homes/clinics (`m15=31`), how do survey-weighted cesarean (`m17`) proportions and covariate-standardized public–private differences vary between NFHS-4 and NFHS-5, across residence groups and states/UTs?

**Interpretation:** Observational, descriptive and model-standardized *associations*. Neither delivery-sector assignment nor state policy effects are causally identified. SHAP pertains to prediction only.

**Primary narrow-facility samples** after the recorded 60-month birth-recall/date criteria: NFHS-4 all births n=114,263 (first delivery n=52,851), NFHS-5 all births n=115,197 (first delivery n=51,854). Broad-sector NFHS-5 n=200,794 and original predictive NFHS-5 n=201,311 refer to **different analytic cohorts** and must not be substituted for current manuscript denominators. First deliveries include twins where recorded.

| Study/wave | Observed private−public gap | Model-standardized private−public gap |
|---|---:|---:|
| NFHS-4 narrow, all births | 22.42 pp | 19.17 pp |
| NFHS-4 narrow, first deliveries | 23.62 pp | 21.21 pp |
| NFHS-5 narrow, all births | 25.16 pp | 23.76 pp |
| NFHS-5 narrow, first deliveries | 25.62 pp | 25.24 pp |

National models above standardize within their own wave/cohort. **Cross-wave comparisons instead use a shared, support-restricted reference** in 33 comparable states/UTs: all-birth change +4.66 pp and first-delivery change +4.00 pp. These must not be reconstructed by subtracting the rounded own-cohort national gaps.

Main figure/supplement plan agreed with Professor Murthy: show an all-birth state-wise figure in the main manuscript, detailed state/UT estimates in supplements; consider a WHO-style state heat map optionally. State-wise estimates are **unadjusted** and are not interchangeable with standardized national estimates. Survey uncertainty is approximate and limitations are recorded.

**Reproducibility and review:** Raw DHS NFHS data are not distributed in this repository; the review below uses committed aggregate outputs and supplied research-update PDFs, not independent microdata reruns. Review documents:
- [Definitive results inventory](docs/manuscript_results_inventory_2026-10-11.md)
- [Uncertainty and design review](docs/uncertainty_review_2026-10-11.md)
- [Code, output and PDF consistency audit](docs/consistency_audit_2026-10-11.md)

The older prediction/AIPW development work below is preserved as **historical project documentation**, not the current primary manuscript specification.

---

## Archived original ML-focused README (historical research phase)

# Explainable Machine Learning Prediction of Cesarean Section Delivery in India

## Overview

This project develops an explainable machine learning model to predict whether an institutional delivery in India will result in a Cesarean section using data from the National Family Health Survey-5 (NFHS-5).

The project emphasizes careful feature selection, prevention of target leakage, respondent-level data splitting, exploratory data analysis, machine learning model comparison, hyperparameter tuning, SHAP-based explainability, probability calibration, and classification-threshold optimization.

The final selected model was a tuned XGBoost classifier.

## Objectives

The main objectives of this project are:

- Predict the likelihood of Cesarean section delivery using maternal, socioeconomic, obstetric, antenatal-care, and pregnancy-related factors.
- Identify the most important predictors associated with Cesarean section delivery.
- Compare linear and tree-based machine learning models.
- Prevent respondent-level data leakage during model development.
- Optimize the strongest candidate models using grouped cross-validation.
- Explain global and individual model predictions using SHAP.
- Evaluate the reliability of predicted probabilities.
- Determine an appropriate classification threshold.
- Examine how healthcare, maternal, socioeconomic, and reproductive factors influence model predictions.

## Dataset

This project uses the **National Family Health Survey-5 (NFHS-5) Birth Recode dataset**, provided through the DHS Program.

The original NFHS-5 microdata is not included in this repository because it is distributed under the DHS Program data usage agreement.

Anyone wishing to reproduce this project must independently request access from the DHS Program and comply with all applicable data usage policies.

After restricting the analysis to institutional deliveries, the final modeling dataset contained:

- **201,311 delivery records**
- **158,429 unique respondents**
- **36 predictor variables**
- **22.23% Cesarean deliveries**
- **77.77% vaginal deliveries**

## Project Structure

```text
.
├── data/
│   ├── raw/                              # Original NFHS-5 data (ignored)
│   └── processed/                        # Processed datasets (ignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_machine_learning_preprocessing.ipynb
│   ├── 04_baseline_machine_learning_models.ipynb
│   ├── 05_hyperparameter_tuning.ipynb
│   ├── 06_shap_explainability.ipynb
│   ├── 07_calibration_threshold_optimization.ipynb
│   └── 08_results_discussion_and_conclusion.ipynb
├── artifacts/                            # Saved models and analysis objects (ignored)
├── outputs/                              # Safe figures and report-ready outputs
├── src/                                  # Reusable source code
├── docs/                                 # Project documentation and references
├── requirements.txt
├── README.md
└── .gitignore
```

## Methodology

The project follows the workflow below:

1. Data acquisition
2. Metadata inspection and variable verification
3. Study population selection
4. Manual predictor selection
5. Target leakage analysis
6. DHS special-code cleaning
7. Final modeling dataset creation
8. Exploratory Data Analysis
9. Statistical comparison of predictors
10. Correlation and missing-value analysis
11. Respondent-level train-test splitting
12. Machine learning preprocessing
13. Baseline model development
14. Hyperparameter tuning with grouped cross-validation
15. Final model comparison and selection
16. SHAP-based global and local explainability
17. Probability calibration
18. Classification-threshold optimization
19. Final results interpretation
20. Discussion, limitations, future work, and conclusion

## Data Preparation

The analysis was restricted to institutional deliveries using the NFHS-5 place-of-delivery variable.

Variables were manually reviewed using:

- DHS variable labels
- DHS value labels
- Observed values
- Missing-value patterns
- Special response codes
- Timing relative to delivery
- Potential target leakage

Variables describing events occurring during or after delivery were excluded from the predictor set.

The final dataset was saved locally as:

```text
data/processed/df_model_v1.parquet
```

This file is ignored by Git and is not distributed through the repository.

## Exploratory Data Analysis

The EDA phase included:

- Cesarean outcome distribution
- Maternal demographic analysis
- Socioeconomic analysis
- Obstetric-history analysis
- Maternal BMI analysis
- ANC timing and visit analysis
- ANC-provider analysis
- ANC-quality analysis
- Pregnancy counselling analysis
- Pregnancy-complication analysis
- Facility-type analysis
- Mann–Whitney U tests for numeric predictors
- Chi-square tests for categorical predictors
- Effect-size analysis
- Spearman correlation analysis
- Missing-value analysis

The EDA showed that Cesarean delivery was associated with several maternal, socioeconomic, reproductive, antenatal-care, and facility-related characteristics.

## Machine Learning Preprocessing

The final 36 predictors were grouped into:

- 10 numeric features
- 5 categorical features
- 21 binary features

Two preprocessing pipelines were created.

### Linear-model preprocessing

Used for Logistic Regression:

- Median imputation
- Numeric missing-value indicators
- Standard scaling
- One-hot encoding

### Tree-model preprocessing

Used for Decision Tree, Random Forest, and XGBoost:

- Median imputation
- Numeric missing-value indicators
- One-hot encoding
- No numeric scaling

The original 36 predictors expanded to **99 transformed features** after encoding and missing-indicator creation.

A respondent-level train-test split was used to ensure that multiple birth records from the same respondent were not divided between training and test sets.

Final split:

- Training set: **161,048 records**
- Test set: **40,263 records**
- Respondent overlap: **0**

## Baseline Models

The following baseline classifiers were trained:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Baseline performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.7259 | 0.4325 | 0.7479 | 0.5481 | 0.8035 | 0.5333 |
| Logistic Regression | 0.7383 | 0.4443 | 0.7071 | 0.5457 | 0.7943 | 0.5109 |
| Random Forest | 0.7964 | 0.5866 | 0.2846 | 0.3833 | 0.7904 | 0.5040 |
| Decision Tree | 0.7229 | 0.3780 | 0.3821 | 0.3800 | 0.6012 | 0.2818 |

XGBoost achieved the strongest overall baseline performance.

## Hyperparameter Tuning

The three strongest candidate models were tuned:

- Logistic Regression
- Random Forest
- XGBoost

Hyperparameter tuning used:

- `GridSearchCV` for Logistic Regression
- `RandomizedSearchCV` for Random Forest
- `RandomizedSearchCV` for XGBoost
- `StratifiedGroupKFold` with three folds
- Respondent ID as the grouping variable
- PR-AUC (`average_precision`) as the primary optimization metric

### Best cross-validated PR-AUC

| Model | Best CV PR-AUC |
|---|---:|
| XGBoost | 0.5291 |
| Random Forest | 0.5204 |
| Logistic Regression | 0.5085 |

### Tuned model performance

| Model | Test Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.7248 | 0.4313 | 0.7471 | 0.5469 | 0.8033 | 0.5330 |
| Random Forest | 0.7383 | 0.4444 | 0.7101 | 0.5467 | 0.8002 | 0.5262 |
| Logistic Regression | 0.7384 | 0.4444 | 0.7073 | 0.5459 | 0.7943 | 0.5108 |

Random Forest showed the largest improvement after tuning, particularly in Cesarean recall.

XGBoost remained the strongest overall model based on Recall, F1-score, ROC-AUC, PR-AUC, and generalization performance.

## Final Model

The tuned XGBoost model was selected as the final predictive model.

### Final XGBoost performance

| Metric | Value |
|---|---:|
| Training accuracy | 0.7352 |
| Test accuracy | 0.7248 |
| Precision | 0.4313 |
| Recall | 0.7471 |
| F1-score | 0.5469 |
| ROC-AUC | 0.8033 |
| PR-AUC | 0.5330 |
| Best cross-validated PR-AUC | 0.5291 |

The small difference between training and test performance suggests good generalization with limited overfitting.

## SHAP Explainability

SHAP analysis was performed using a stratified sample of **5,000 unseen test observations**.

The SHAP sample preserved the test-set outcome distribution:

- Vaginal delivery: 77.78%
- Cesarean delivery: 22.22%

Global and local explanations were generated using:

- SHAP global bar plots
- SHAP beeswarm plots
- Original-variable importance aggregation
- Numeric dependence plots
- Individual waterfall plots
- High-risk and low-risk case comparisons
- False-positive and false-negative explanations

### Most influential original predictors

The leading predictors identified by SHAP were:

1. Facility type
2. Total children ever born
3. Maternal BMI
4. Wealth index
5. Age at first birth
6. Doctor-provided ANC
7. Years of education
8. Number of ANC visits
9. Social group
10. Preceding birth interval

Facility type was the most influential predictor in the final XGBoost model.

SHAP findings describe associations learned by the model and must not be interpreted as proof of causation.

## Probability Calibration and Threshold Optimization

The reliability of the final XGBoost probability estimates was evaluated using:

- Calibration curves
- Brier score
- Sigmoid calibration
- Isotonic calibration

The original XGBoost model was compared with sigmoid-calibrated and isotonic-calibrated versions.

Calibration produced only limited changes, indicating that the final XGBoost probabilities were already reasonably reliable.

Classification thresholds ranging from 0.05 to 0.95 were then evaluated using:

- Accuracy
- Precision
- Recall
- Specificity
- F1-score
- Youden’s J statistic
- False-positive count
- False-negative count

The following threshold strategies were compared:

- Default threshold of 0.50
- Threshold maximizing F1-score
- Threshold maximizing Youden’s J
- Recall-focused threshold

The F1-optimal threshold was retained as the main recommended operating threshold because it provided a balanced trade-off between Precision and Recall.

The chosen threshold remains a statistical operating point and should not be treated as a clinical decision threshold without external and prospective validation.

## Interpretation and Discussion

The final results showed that:

- XGBoost provided the strongest overall balance of discrimination, recall, minority-class performance, and generalization.
- Logistic Regression remained a stable and interpretable benchmark.
- Random Forest benefited substantially from hyperparameter tuning.
- Facility type was the strongest predictor in the final model.
- Maternal BMI, reproductive history, wealth, education, and ANC utilization also contributed strongly to model predictions.
- False-positive and false-negative cases likely reflect clinical and emergency factors not available in NFHS-5.
- SHAP explanations describe learned associations and do not establish causality.

## Study Limitations

Important limitations include:

- Retrospective survey-data design
- Lack of detailed clinical and fetal variables
- Lack of reliable previous Cesarean history
- Possible recall and reporting bias
- No external validation
- No prospective clinical validation
- Strong influence of facility type
- Small size of the other-facility category
- Statistical rather than clinically validated threshold selection
- Inability to establish causal relationships

## Future Work

Future extensions could include:

- External validation using another maternal-health dataset
- Validation using another NFHS survey round
- State-specific and district-level models
- Models developed separately for public and private facilities
- Inclusion of previous Cesarean history
- Addition of richer clinical and fetal variables
- Fairness analysis across demographic and socioeconomic groups
- Survey-weight-aware model development
- Prospective hospital-based validation
- Development of an explainable research prototype
- Comparison with alternative gradient-boosting algorithms
- Sensitivity analysis without facility type
- Sensitivity analysis excluding the other-facility category

## Technologies

The project uses:

- Python
- Pandas
- NumPy
- Pyreadstat
- PyArrow
- SciPy
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Matplotlib
- Seaborn
- Jupyter Notebook

## Reproducibility

To reproduce the project:

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required packages.
4. Obtain the NFHS-5 Birth Recode dataset from the DHS Program.
5. Place the dataset inside the local `data/raw/` directory.
6. Run the notebooks in numerical order.

```bash
pip install -r requirements.txt
```

The original data, processed datasets, trained models, and serialized artifacts are not included in the repository.

## Data Confidentiality

NFHS-5 microdata is not included in this repository.

Users must independently obtain access through the DHS Program and comply with its data usage agreement.

The following are excluded from version control:

- Raw NFHS-5 data
- Processed record-level datasets
- Variable-mapping files generated locally
- Serialized preprocessing objects
- Trained model files
- SHAP value arrays
- Calibration objects
- Threshold artifacts
- Other potentially large or record-level derived artifacts

Only safe, aggregated figures and result summaries may be included under `outputs/`.

## Project Progress

### Completed

- Project definition
- NFHS-5 dataset loading
- Metadata inspection
- Institutional-delivery selection
- Manual feature selection
- Target-leakage analysis
- DHS special-code cleaning
- Final modeling dataset creation
- Exploratory Data Analysis
- Statistical testing
- Correlation analysis
- Missing-value analysis
- Respondent-level train-test split
- Machine learning preprocessing
- Baseline model development
- Model evaluation
- Hyperparameter tuning
- Overfitting assessment
- Final model selection
- SHAP global explainability
- SHAP local explainability
- High-risk and low-risk case analysis
- False-positive and false-negative analysis
- Probability calibration
- Classification-threshold optimization
- Results interpretation
- Discussion
- Study-strength analysis
- Limitation analysis
- Future-work planning
- Final project conclusion

### Remaining Work

- Literature review
- Research-paper formatting
- Reference management
- Final presentation or poster
- Final repository cleanup