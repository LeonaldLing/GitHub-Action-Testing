import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_PATH = PROJECT_ROOT / "logs" / "training.log"
PARAMS = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))
RANDOM_STATE = PARAMS["data"]["random_seed"]


def configure_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - training - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def load_arrays():
    data_dir = ARTIFACTS_DIR / "data"
    return (
        np.load(data_dir / "X_train.npy"),
        np.load(data_dir / "y_train.npy"),
        np.load(data_dir / "X_test.npy"),
        np.load(data_dir / "y_test.npy"),
    )


def build_model():
    model_params = PARAMS["model"]["random_forest"]
    return RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        class_weight=model_params["class_weight"],
        random_state=RANDOM_STATE,
    )


def save_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    configure_logging()
    logging.info("=" * 70)
    logging.info("TRAINING RANDOM FOREST OBESITY CLASSIFIER")
    logging.info("=" * 70)

    x_train, y_train, x_test, y_test = load_arrays()
    logging.info("Loaded X_train=%s y_train=%s", x_train.shape, y_train.shape)
    logging.info("Loaded X_test=%s y_test=%s", x_test.shape, y_test.shape)

    model = build_model()
    logging.info("Fitting Random Forest model")
    model.fit(x_train, y_train)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    metrics = {
        "model_type": "random_forest",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_weighted_f1": float(f1_score(y_test, test_predictions, average="weighted")),
        "hyperparameters": PARAMS["model"]["random_forest"],
    }

    model_path = ARTIFACTS_DIR / "models" / "random_forest_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    save_json(metrics, ARTIFACTS_DIR / "metrics" / "training_history.json")

    logging.info("Saved model to %s", model_path)
    logging.info("Test accuracy: %.4f", metrics["test_accuracy"])
    print(json.dumps({"test_accuracy": metrics["test_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
