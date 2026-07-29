# PredictX: Industrial IoT Predictive Maintenance Platform

End-to-end Machine Learning platform for Industrial IoT predictive maintenance using Random Forest, SHAP, and Streamlit to predict machine failures and optimize maintenance decisions.

## Project Structure

```
PredictX/
├── app/                    # Streamlit pages (Home, Dashboard, Prediction, Model Comparison, Explainability, Business Insights)
├── src/                    # ML lifecycle modules
│   ├── preprocess.py       # Data ingestion, validation, feature encoding
│   ├── eda.py              # EDA charts saved to screenshots/
│   ├── train.py            # Model training (LR, DT, RF)
│   ├── evaluate.py         # Confusion matrix, ROC curve, classification report
│   ├── explain.py          # Feature importance + SHAP explanations
│   ├── predict.py          # Single-machine failure scoring engine
│   └── results_summary.py  # Full pipeline runner
├── data/raw/               # AI4I dataset (auto-generated if absent)
├── data/processed/         # Quality reports
├── models/                 # Trained model artifacts (.joblib, .json)
├── screenshots/            # EDA and evaluation charts
├── notebooks/              # EDA Jupyter notebook
├── tests/                  # pytest unit tests
├── streamlit_app.py        # Streamlit entry point
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.results_summary   # Run full pipeline
streamlit run streamlit_app.py  # Launch dashboard
```

## Features

- **Data**: AI4I 2020 dataset with deterministic offline fallback
- **Models**: Logistic Regression, Decision Tree, Random Forest (class-weighted, stratified split)
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Explainability**: Feature importance + SHAP (global + local)
- **Dashboard**: 6-page Streamlit app
- **Business Insights**: ROI calculator with adjustable cost parameters
- **Tests**: 7 pytest unit tests

## License

MIT License
