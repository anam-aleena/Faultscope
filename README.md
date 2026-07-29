# 🏭 FaultScope

> **Industrial IoT Predictive Maintenance Platform** — Real-time machine failure prediction using Random Forest, SHAP explainability, and an interactive 6-page Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-url.streamlit.app)

---

## 🎯 Problem Statement

Unplanned machine breakdowns cost Indian manufacturers **₹50,000–₹5,00,000 per incident** in downtime, repairs, and lost production. Most small and mid-size factories have zero predictive capability — they run machines until failure.

**FaultScope** gives factory operators and maintenance teams an instant AI-powered risk score for any machine, so they can act before breakdown — not after.

---

## 🖥️ Live Demo

🔗 **[Launch FaultScope →](https://your-streamlit-url.streamlit.app)**

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 Real-time prediction | Enter sensor readings → get failure probability instantly |
| 🔴 Risk stratification | LOW / MEDIUM / HIGH / CRITICAL with maintenance advice |
| 📊 EDA Dashboard | 5 auto-generated charts: histograms, heatmaps, boxplots |
| 🔬 Model comparison | Logistic Regression vs Decision Tree vs Random Forest |
| 💡 Explainability | SHAP global + local feature importance |
| 📈 Business ROI | Adjustable cost calculator — quantify maintenance savings |
| ✅ Test suite | 7 pytest unit tests across pipeline and prediction |

---

## 🏗️ Project Structure

```
FaultScope/
├── app/                          # Streamlit page modules
│   ├── Home.py                   # Landing page + architecture overview
│   ├── Dashboard.py              # EDA charts + data quality report
│   ├── Prediction.py             # Real-time failure scoring UI
│   ├── Model_Comparison.py       # Side-by-side model metrics + ROC curve
│   ├── Explainability.py         # SHAP global + local explanations
│   └── Business_Insights.py      # ROI calculator + business Q&A
│
├── src/                          # ML pipeline modules
│   ├── preprocess.py             # Data ingestion, validation, feature encoding
│   ├── eda.py                    # EDA chart generation → screenshots/
│   ├── train.py                  # Model training (LR, DT, Random Forest)
│   ├── evaluate.py               # Confusion matrix, ROC curve, classification report
│   ├── explain.py                # Feature importance + SHAP explanations
│   ├── predict.py                # Single-machine failure scoring engine
│   └── results_summary.py        # One-command full pipeline runner
│
├── tests/                        # pytest unit tests
│   ├── test_preprocess.py        # Data pipeline tests
│   ├── test_train.py             # Model training tests
│   └── test_predict.py           # Prediction engine tests
│
├── streamlit_app.py              # App entry point
├── requirements.txt
├── pytest.ini
└── .gitignore

# Auto-generated on first run (git-ignored):
# data/raw/        ← AI4I dataset or synthetic fallback
# data/processed/  ← Quality reports
# models/          ← Trained .joblib artifacts
# screenshots/     ← EDA and evaluation charts
```

---

## 🤖 ML Pipeline

```
Raw Sensor Data
      ↓
Data Ingestion + Validation
      ↓
Feature Encoding (LabelEncoder for machine type)
      ↓
Train/Test Split (stratified, 80/20)
      ↓
┌─────────────────────────────────┐
│  Logistic Regression            │
│  Decision Tree (max_depth=8)    │
│  Random Forest ← Best Model     │
└─────────────────────────────────┘
      ↓
Evaluation (ROC-AUC, F1, Precision, Recall)
      ↓
SHAP Explainability (TreeExplainer)
      ↓
Streamlit Dashboard
```

---

## 📡 Sensor Features

| Feature | Description | Unit |
|---|---|---|
| Machine Type | Grade: L (Light), M (Medium), H (Heavy) | — |
| Air Temperature | Ambient temperature | Kelvin |
| Process Temperature | Operational temperature | Kelvin |
| Rotational Speed | Motor speed | RPM |
| Torque | Applied torque | Nm |
| Tool Wear | Cumulative tool usage | Minutes |
| **Machine Failure** | **Binary target** | **0 = OK, 1 = Fail** |

Dataset: **AI4I 2020 Predictive Maintenance** (UCI ML Repository) — with deterministic synthetic fallback if unavailable.

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/anam-aleena/Faultscope.git
cd Faultscope
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full ML pipeline
```bash
python -m src.results_summary
```
This trains all models, generates EDA charts, evaluates the best model, and saves SHAP explanations — all in one command.

### 4. Launch the dashboard
```bash
streamlit run streamlit_app.py
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

7 unit tests covering data preprocessing, model training, and prediction engine.

---

## 📸 Dashboard Pages

| Page | What you see |
|---|---|
| 🏠 Home | Architecture overview, dataset info |
| 📊 Dashboard | EDA charts, data quality metrics |
| 🎯 Prediction | Live risk scorer with gauge + advice |
| 🔬 Model Comparison | Bar chart comparison + confusion matrix + ROC |
| 💡 Explainability | SHAP global + local feature importance |
| 📈 Business Insights | ROI calculator, operational Q&A |

---

## 💰 Business Impact

FaultScope quantifies the financial value of predictive maintenance:

- **Detected failures** → avoided downtime cost
- **False alarms** → cost of unnecessary maintenance
- **Missed failures** → residual risk exposure
- **Net savings** → real ROI of the model

All parameters are adjustable in the Business Insights page.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn, XGBoost, SHAP |
| Data | pandas, numpy, AI4I 2020 dataset |
| Visualization | matplotlib, seaborn |
| App | Streamlit |
| Testing | pytest |
| Serialization | joblib |

---

## 👩‍💻 Author

**Aleena Anam**
B.Sc. Computer Science · SRTMU, Nanded · Maharashtra

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aleena--anam-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/aleena-anam-2056a4368)
[![GitHub](https://img.shields.io/badge/GitHub-anam--aleena-181717?style=flat&logo=github)](https://github.com/anam-aleena)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

<p align="center">
  Built with Python · scikit-learn · SHAP · Streamlit
</p>
