# DeepGuard AI — Deep Learning Fraud Detection MVP

A lightweight, end-to-end credit-card fraud detection application built with **PyTorch, FastAPI, Streamlit, pandas, and scikit-learn**.

The project demonstrates how to train a deep learning model on an imbalanced dataset, evaluate its performance, expose predictions through a REST API, and visualize results in an interactive dashboard.

> **Project status:** MVP / educational prototype  
> **Development environment:** macOS + VS Code  
> **Primary language:** Python  
> **Model:** Feed-forward neural network  
> **Dataset:** Public Credit Card Fraud Detection dataset

---

## 1. Project Objectives

DeepGuard AI aims to demonstrate a practical machine learning workflow:

1. Load and validate transaction data.
2. Preprocess transaction features.
3. Train a PyTorch neural network to identify potentially fraudulent transactions.
4. Evaluate the model using metrics suitable for imbalanced classification.
5. Save the trained model and preprocessing artifacts.
6. Serve predictions through a FastAPI REST API.
7. Display predictions in a Streamlit dashboard.
8. Prepare the architecture for future enterprise-scale improvements.

## 2. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Primary programming language |
| PyTorch | Deep learning model and training |
| pandas | Dataset loading and manipulation |
| NumPy | Numerical computation |
| scikit-learn | Data splitting, scaling, and evaluation |
| joblib | Saving the preprocessing scaler |
| FastAPI | REST API for fraud scoring |
| Uvicorn | API server |
| Streamlit | Interactive dashboard |
| pytest | Automated tests |

## 3. Project Structure

```text
DeepGuardAI/
├── app/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   ├── api.py
│   └── dashboard.py
├── data/
│   └── raw/
│       └── creditcard.csv
├── artifacts/
│   ├── fraud_model.pt
│   ├── scaler.joblib
│   └── metadata.json
├── tests/
│   └── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

### File responsibilities

- `app/model.py`: Defines the PyTorch neural network.
- `app/train.py`: Loads the dataset, preprocesses data, trains the model, evaluates performance, and saves artifacts.
- `app/api.py`: Loads the trained artifacts and exposes prediction endpoints.
- `app/dashboard.py`: Provides the interactive Streamlit user interface.
- `tests/test_model.py`: Contains basic automated tests.
- `data/raw/`: Stores the original dataset.
- `artifacts/`: Stores trained model weights, the feature scaler, and metadata.
- `requirements.txt`: Lists Python dependencies.
- `.gitignore`: Excludes local environments and generated files from version control.

## 4. Prerequisites

Before starting, install or verify:

- macOS
- Python 3.11 recommended
- Visual Studio Code
- VS Code Python extension
- Git, recommended but optional
- Internet access for downloading the dataset and dependencies

Check your Python version:

```bash
python3 --version
```

Check Git:

```bash
git --version
```

If Python 3.11 is not installed, install it from the official Python website:

https://www.python.org/downloads/

## 5. Create the Project

Open VS Code and select **Terminal → New Terminal**.

Create the project directory:

```bash
mkdir -p ~/Projects/DeepGuardAI
cd ~/Projects/DeepGuardAI
code .
```

If `code .` is unavailable, open VS Code manually and select **File → Open Folder**. Choose `~/Projects/DeepGuardAI`.

Create the directories and initial files:

```bash
mkdir -p app data/raw artifacts tests

touch app/__init__.py
touch app/model.py
touch app/train.py
touch app/api.py
touch app/dashboard.py
touch tests/test_model.py
touch requirements.txt
touch .gitignore
touch README.md
touch artifacts/.gitkeep
```

Populate each Python file with the implementation maintained for this project.

## 6. Create a Python Virtual Environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

When the environment is active, your terminal should display `(.venv)`.

Each new terminal session must activate the environment before running the project.

## 7. Install Dependencies

Add the following to `requirements.txt`:

```text
torch
numpy
pandas
scikit-learn
joblib
fastapi
uvicorn[standard]
streamlit
requests
pytest
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

Verify the important imports:

```bash
python -c "import torch, pandas, sklearn, fastapi, streamlit; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

If MPS is available, PyTorch can potentially use Apple's Metal acceleration. Otherwise, the project can run on CPU.

## 8. Download the Dataset

Download the public Credit Card Fraud Detection dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Steps:

1. Sign in to Kaggle.
2. Download the dataset ZIP file.
3. Extract the ZIP.
4. Locate `creditcard.csv`.
5. Copy the CSV into:

```text
DeepGuardAI/data/raw/creditcard.csv
```

Verify the file:

```bash
ls -lh data/raw/creditcard.csv
```

Inspect its columns:

```bash
python -c "import pandas as pd; d=pd.read_csv('data/raw/creditcard.csv', nrows=5); print(d.columns.tolist()); print(d.head(2))"
```

The expected columns are:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class`

`Class = 0` represents a legitimate transaction, while `Class = 1` represents a fraudulent transaction.

The dataset contains 284,807 transactions and 492 labeled fraud cases. Fraud therefore represents approximately 0.172% of all records.

## 9. Understand the Deep Learning Model

The model is defined in `app/model.py`.

Its architecture is:

```text
Input: 30 transaction features
             |
             v
      Linear: 30 → 64
             |
            ReLU
             |
          Dropout
             |
             v
       Linear: 64 → 32
             |
            ReLU
             |
          Dropout
             |
             v
       Linear: 32 → 1
             |
             v
          Raw logit
             |
             v
     Sigmoid at inference
             |
             v
    Fraud risk score (0–1)
```

The model learns patterns that distinguish fraudulent and legitimate transactions.

The output score is interpreted as a model-generated fraud risk score. It should not automatically be treated as a calibrated real-world probability.

## 10. Train the Model

From the project root, activate the environment and execute:

```bash
source .venv/bin/activate
python -m app.train
```

The training process performs the following operations:

1. Loads and validates the CSV.
2. Separates transaction features from the target label.
3. Splits the dataset into training, validation, and test subsets.
4. Fits the feature scaler on the training subset only.
5. Applies the trained scaler to validation and test data.
6. Trains the PyTorch neural network.
7. Compensates for class imbalance using a weighted binary classification loss.
8. Selects an alert threshold using validation data.
9. Evaluates the final model on the held-out test subset.
10. Saves the trained model, scaler, and metadata.

The configured split is:

- 70% training data
- 15% validation data
- 15% test data

The split is stratified so the minority fraud class is represented in each subset.

Training can take several minutes depending on the machine.

### Expected artifacts

After successful training, the `artifacts/` directory should contain:

```text
artifacts/
├── fraud_model.pt
├── scaler.joblib
└── metadata.json
```

- `fraud_model.pt`: Model weights and model configuration.
- `scaler.joblib`: Fitted feature scaler.
- `metadata.json`: Feature names, selected threshold, and evaluation metrics.

Do not start the API before these files have been generated successfully.

## 11. Understand Model Evaluation

Fraud detection is an imbalanced classification problem. Accuracy alone is insufficient because a model could predict every transaction as legitimate and still achieve approximately 99.83% accuracy on this dataset.

DeepGuard AI evaluates the model using:

### Precision

Of all transactions flagged by the model, how many were actually fraudulent?

Higher precision generally means fewer false alarms.

### Recall

Of all fraudulent transactions, how many did the model identify?

Higher recall generally means fewer missed fraud cases.

### F1 score

The harmonic mean of precision and recall.

It provides a single measure balancing both metrics at a chosen threshold.

### Precision-Recall AUC / Average Precision

Measures performance across different thresholds and is especially useful when the positive class is rare.

### ROC-AUC

Measures the model's ability to rank positive examples above negative examples across thresholds.

### Confusion matrix

The model reports:

- True negatives (TN)
- False positives (FP)
- False negatives (FN)
- True positives (TP)

Review these results before claiming that the model performs well. No particular score is guaranteed.

The selected alert threshold is chosen on validation data rather than optimized against the test labels. In a real financial application, threshold selection should also account for investigation capacity, fraud losses, false-decline costs, and operational requirements.

## 12. Start the FastAPI Service

After training, start the API from the project root:

```bash
source .venv/bin/activate
uvicorn app.api:app --reload
```

The default API address is:

```text
http://127.0.0.1:8000
```

### Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Application status and documentation link |
| GET | `/health` | Check API and model readiness |
| POST | `/predict` | Score one transaction |
| GET | `/metrics` | Retrieve saved evaluation metrics |
| GET | `/docs` | Interactive Swagger API documentation |

Open the API documentation:

http://127.0.0.1:8000/docs

Check health:

http://127.0.0.1:8000/health

A successful health response should indicate that the model is loaded.

### Test a prediction

The `/predict` endpoint expects all 30 named features. The simplest way to test it is through the Streamlit dashboard or Swagger UI.

The API validates feature names, checks numeric values, applies the saved scaler, and runs the trained neural network.

The response includes:

- `risk_score`
- `threshold`
- `decision`
- `flagged_for_review`
- `model`
- `disclaimer`

The initial decision labels are `REVIEW` and `LOWER_RISK`. They are demonstration labels, not authorization to approve or decline real transactions.

## 13. Start the Streamlit Dashboard

Keep the API running in the first terminal.

Open a second VS Code terminal:

```bash
cd ~/Projects/DeepGuardAI
source .venv/bin/activate
streamlit run app/dashboard.py
```

Streamlit typically opens:

http://localhost:8501

The dashboard allows you to:

1. Select a transaction from the dataset.
2. View the transaction amount and elapsed time.
3. Inspect the 30 model input features.
4. Submit the transaction to the FastAPI service.
5. View the risk score and review threshold.
6. See whether the transaction was flagged for review.
7. Inspect the historical dataset label after scoring.

The dashboard displays the actual label separately for retrospective demonstration. That label must never be included among the model's input features.

## 14. Run Automated Tests

From the project root:

```bash
source .venv/bin/activate
pytest -q
```

The initial tests verify that the neural network accepts correctly shaped inputs and returns finite outputs of the expected shape.

These tests do not prove that the model detects fraud accurately. Model quality is evaluated separately through the training and test metrics.

## 15. End-to-End Validation Checklist

Before demonstrating the application, verify each step.

- [ ] Python virtual environment is active.
- [ ] All dependencies install successfully.
- [ ] `creditcard.csv` exists in `data/raw/`.
- [ ] `python -m app.train` completes successfully.
- [ ] Model artifacts exist in `artifacts/`.
- [ ] Test metrics are saved and printed.
- [ ] `pytest -q` passes.
- [ ] FastAPI starts without errors.
- [ ] `/health` confirms the model is loaded.
- [ ] `/docs` opens successfully.
- [ ] Streamlit starts on port 8501.
- [ ] The dashboard can submit a transaction and display a prediction.
- [ ] Missing or malformed API features are rejected.
- [ ] No actual financial or customer data is used in the demonstration.

## 16. Troubleshooting

### Dataset file not found

Verify:

```bash
ls data/raw/creditcard.csv
```

If it is missing, download and extract the Kaggle dataset into the specified directory.

### Module import error

Run commands from the project root and use module execution:

```bash
python -m app.train
```

Avoid executing `train.py` directly from inside the `app/` directory.

### API cannot find model artifacts

Run training first:

```bash
python -m app.train
```

Then restart Uvicorn.

### Dashboard cannot connect to API

Confirm that Uvicorn is running on port 8000 and that this URL works:

```text
http://127.0.0.1:8000/health
```

### Port already in use

Run the API on another port:

```bash
uvicorn app.api:app --reload --port 8001
```

If you change the API port, update `API_URL` in `app/dashboard.py` to match.

### PyTorch installation problems

Confirm that the virtual environment is active, then consult the official installation instructions:

https://pytorch.org/get-started/locally/

### Unexpected evaluation metrics

Check the dataset schema, class distribution, stratified splits, feature scaling, threshold selection, and confusion matrix. Do not tune the threshold using the held-out test labels.

## 17. Security and Limitations

DeepGuard AI is an educational MVP, not a production fraud prevention system.

Important limitations:

- The dataset is historical and anonymized.
- The model has not been validated on current banking transactions.
- Risk scores may not be calibrated probabilities.
- The initial application has no authentication or authorization.
- The API is intended for local development and must not be exposed publicly as-is.
- No database-backed audit history or production monitoring is included.
- The model has no guarantee of detecting new fraud patterns.
- Model artifacts and the scaler must be versioned and deployed together.
- The application does not include regulatory, privacy, or financial decision governance controls.

Only use trusted model artifacts. Loading serialized Python objects such as joblib files from untrusted sources can execute unsafe code. Keep the dataset and generated artifacts out of public Git repositories unless there is a deliberate, reviewed reason to publish them.

## 18. Future Enhancement Roadmap

### Phase 1 — MVP

- PyTorch fraud classification
- FastAPI prediction endpoint
- Streamlit dashboard
- Evaluation metrics
- Local automated tests

### Phase 2 — Enterprise application

- Java 21 and Spring Boot services
- React dashboard
- PostgreSQL transaction and investigation records
- Authentication and authorization
- Audit history and structured logging

### Phase 3 — Real-time processing

- Kafka transaction ingestion
- Asynchronous scoring
- Retry and idempotency handling
- Monitoring and alerting
- Model and data drift detection

### Phase 4 — Model improvement

- Hyperparameter tuning
- Threshold optimization based on business costs
- Explainability with SHAP or other suitable techniques
- Model versioning and experiment tracking
- Comparison with logistic regression and tree-based baselines

### Phase 5 — Advanced AI

- Graph neural networks for suspicious account relationships
- Behavioral anomaly detection
- Investigator assistance using an LLM
- Evidence-backed investigation summaries
- Human review and approval workflows

### Phase 6 — Cloud deployment

- Docker containers
- AWS deployment
- Managed model and artifact storage
- Secrets management
- CI/CD
- Security monitoring and production observability

Each phase should be added only after the previous phase is working and tested.

## 19. Useful References

- Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- PyTorch installation: https://pytorch.org/get-started/locally/
- PyTorch tutorials: https://docs.pytorch.org/tutorials/
- FastAPI documentation: https://fastapi.tiangolo.com/
- Streamlit documentation: https://docs.streamlit.io/
- scikit-learn documentation: https://scikit-learn.org/stable/

---

**DeepGuard AI objective:** Demonstrate a complete deep learning lifecycle—from data preparation and model training to API-based inference and an interactive user interface—with a clear path toward a scalable enterprise fraud detection platform.
