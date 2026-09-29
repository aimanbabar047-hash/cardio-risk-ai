❤️ CardioRisk AI

AI-powered cardiovascular disease risk screening: a calibrated XGBoost pipeline with a screening threshold, per-patient explanations and a full analytics dashboard, delivered as a Streamlit app.

GIBC V2 – Track 02: Applied (Medical Technology)

Research prototype. Not a medical diagnosis. Never used on real patients.

What it does
Risk Assessment – enter 11 inputs, get a model-estimated probability and a screening flag
Why this prediction? – SHAP-style contributions (log-odds) for each input
Model Analytics – test metrics, ROC/PR curves, threshold slider, 5-fold CV, bootstrap CIs, calibration, subgroup results
Explainability, Methodology, Responsible AI pages
How it works
Public dataset: Cardiovascular Disease dataset (Kaggle), 70,000 raw records
Cleaning: implausible blood pressure, height and weight removed; age converted to years (30–65)
Preprocessing + Logistic Regression, Random Forest, XGBoost; XGBoost tuned with randomized search (3-fold CV)
5-fold cross-validation and bootstrap confidence intervals
Isotonic calibration (kept only if it lowers Brier score)
Screening threshold chosen from out-of-fold training predictions for ~80% recall
Model, metadata, metrics and test predictions saved to models/ and read by the app
Project structure
app.py                                  Streamlit application
Cardiovascular_Disease_data_analysis_updated.ipynb   Full analysis and training
requirements.txt
models/
  cardiorisk_xgboost_pipeline.joblib
  model_metadata.json
  model_metrics.json
  test_predictions.npz
Setup and run

Prerequisites: Python 3.10+

bash
git clone https://github.com/aimanbabar047-hash/cardiorisk-ai.git
cd cardiorisk-ai
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

To retrain: open the notebook, run all cells (it downloads the dataset with kagglehub), and it will regenerate everything in models/. For exact compatibility, use the same scikit-learn and XGBoost versions that were used for training (recorded in models/model_metadata.json).

Limitations

Public dataset, no clinical validation, trained on ages 30–65 only, gender codes undocumented in the dataset, and the screening threshold favours recall so false alarms are expected.

Team

Aiman Babar (solo participant)
