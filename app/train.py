"""Load data, train and evaluate the model, and save artifacts.

Run from the project root with: python -m app.train
"""

from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.model import FraudNet


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "creditcard.csv"
ARTIFACT_DIR = ROOT / "artifacts"

SEED = 42
BATCH_SIZE = 2048
EPOCHS = 10
LEARNING_RATE = 0.001


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing: {DATA_PATH}\n"
            "Download creditcard.csv into data/raw/."
        )

    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain the Class target column.")

    feature_columns = [c for c in df.columns if c != "Class"]
    if len(feature_columns) != 30:
        raise ValueError(
            f"Expected 30 input features, got {len(feature_columns)}."
        )

    X = df[feature_columns].astype("float32").to_numpy()
    y = df["Class"].astype("int64").to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values.")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Class must contain only 0 and 1.")

    # Split first. The scaler must be fitted on training data only.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=SEED, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")

    print(f"Rows: train={len(y_train):,}, "
          f"validation={len(y_val):,}, test={len(y_test):,}")
    print(f"Training fraud cases: {int(y_train.sum())}")

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype("float32")),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )

    # Use Apple's Metal accelerator when available; otherwise CPU.
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Training device: {device}")

    model = FraudNet(input_size=len(feature_columns)).to(device)

    positives = max(int(y_train.sum()), 1)
    negatives = int(len(y_train) - y_train.sum())
    pos_weight = torch.tensor(
        [negatives / positives], dtype=torch.float32, device=device
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"loss={total_loss / len(train_ds):.5f}"
        )

    def scores_for(data):
        model.eval()
        all_scores = []
        with torch.no_grad():
            for start in range(0, len(data), BATCH_SIZE):
                xb = torch.from_numpy(
                    data[start:start + BATCH_SIZE]
                ).to(device)
                logits = model(xb)
                all_scores.extend(
                    torch.sigmoid(logits).cpu().numpy().tolist()
                )
        return np.asarray(all_scores)

    # Tune the alert threshold on validation data, never on the test set.
    val_scores = scores_for(X_val)
    precision, recall, thresholds = precision_recall_curve(
        y_val, val_scores
    )

    if len(thresholds):
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        best_idx = int(np.argmax(f1_values))
        threshold = float(thresholds[best_idx])
    else:
        threshold = 0.5

    # Final evaluation uses the held-out test set.
    test_scores = scores_for(X_test)
    test_preds = (test_scores >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "average_precision_pr_auc": float(
            average_precision_score(y_test, test_scores)
        ),
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(
            y_test, test_preds, labels=[0, 1]
        ).ravel().tolist(),
        "classification_report": classification_report(
            y_test, test_preds, labels=[0, 1],
            target_names=["legitimate", "fraud"],
            zero_division=0, output_dict=True
        ),
    }

    print("\nHeld-out test results")
    print(f"Selected threshold: {threshold:.6f}")
    print(f"PR-AUC / average precision: "
          f"{metrics['average_precision_pr_auc']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Confusion matrix [TN, FP, FN, TP]:",
          metrics["confusion_matrix_tn_fp_fn_tp"])
    print(classification_report(
        y_test, test_preds, labels=[0, 1],
        target_names=["legitimate", "fraud"], zero_division=0
    ))

    # Persist all inference dependencies.
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "input_size": len(feature_columns),
        },
        ARTIFACT_DIR / "fraud_model.pt",
    )
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")

    metadata = {
        "feature_columns": feature_columns,
        "threshold": threshold,
        "metrics": metrics,
        "seed": SEED,
        "epochs": EPOCHS,
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
