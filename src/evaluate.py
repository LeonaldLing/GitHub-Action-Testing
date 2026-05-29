import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_PATH = PROJECT_ROOT / "logs" / "evaluation.log"


def configure_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - evaluation - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def load_classes():
    metadata_path = ARTIFACTS_DIR / "preprocessing" / "feature_columns.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata["target_classes"]


def save_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    configure_logging()
    logging.info("=" * 70)
    logging.info("EVALUATING RANDOM FOREST OBESITY CLASSIFIER")
    logging.info("=" * 70)

    x_test = np.load(ARTIFACTS_DIR / "data" / "X_test.npy")
    y_test = np.load(ARTIFACTS_DIR / "data" / "y_test.npy")
    classes = load_classes()

    model_path = ARTIFACTS_DIR / "models" / "random_forest_model.pkl"
    model = joblib.load(model_path)
    y_pred = model.predict(x_test)

    metrics = {
        "model_type": "random_forest",
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "samples": int(len(y_test)),
        "classes": classes,
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    save_json(metrics, ARTIFACTS_DIR / "metrics" / "evaluation_metrics.json")
    logging.info("Saved evaluation metrics")
    logging.info("Accuracy: %.4f", metrics["accuracy"])
    print(json.dumps({"accuracy": metrics["accuracy"], "weighted_f1": metrics["weighted_f1"]}, indent=2))


if __name__ == "__main__":
    main()
