import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import chisquare, ks_2samp
from sklearn.metrics import accuracy_score, f1_score
import yaml

from preprocess_new_data import TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
LOG_PATH = PROJECT_ROOT / "logs" / "monitoring.log"
DRIFT_LOG_PATH = PROJECT_ROOT / "logs" / "drift_detection.log"
PARAMS = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

KS_PVALUE_THRESHOLD = PARAMS["monitoring"]["ks_pvalue_threshold"]
CHI_PVALUE_THRESHOLD = PARAMS["monitoring"]["chi_square_pvalue_threshold"]
MISSING_RATE_THRESHOLD = PARAMS["monitoring"]["missing_rate_threshold"]
ACCURACY_ALERT_THRESHOLD = PARAMS["monitoring"]["accuracy_alert_threshold"]


def configure_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - monitoring - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def save_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_monitoring_data():
    # Use new data when available; otherwise use test data so monitoring still produces reports.
    reference_path = PROJECT_ROOT / "train" / "train.csv"
    candidate_new_path = PROJECT_ROOT / "data" / "new_data.csv"
    fallback_path = PROJECT_ROOT / "test" / "test.csv"
    current_path = candidate_new_path if candidate_new_path.exists() else fallback_path
    logging.info("Loading reference data from %s", reference_path)
    reference = pd.read_csv(reference_path)
    logging.info("Reference data shape: %s", reference.shape)
    logging.info("Loading current data from %s", current_path)
    current = pd.read_csv(current_path)
    logging.info("Current data shape: %s", current.shape)
    return reference, current, current_path


def data_quality_report(df):
    # Basic data quality checks for missing values and duplicate rows.
    logging.info("Checking data quality")
    missing_rates = df.isna().mean().to_dict()
    report = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_rates": {key: float(value) for key, value in missing_rates.items()},
        "columns_above_missing_threshold": [
            key for key, value in missing_rates.items() if value > MISSING_RATE_THRESHOLD
        ],
    }
    logging.info(
        "Data quality summary: rows=%s columns=%s duplicate_rows=%s",
        report["rows"],
        report["columns"],
        report["duplicate_rows"],
    )
    if report["columns_above_missing_threshold"]:
        logging.warning("High missing rate columns: %s", report["columns_above_missing_threshold"])
    return report


def numeric_drift(reference, current, numeric_columns):
    # Numeric features use the Kolmogorov-Smirnov test to compare distributions.
    logging.info("Running numeric drift checks on %s columns", len(numeric_columns))
    results = {}
    for column in numeric_columns:
        ref_values = reference[column].dropna()
        cur_values = current[column].dropna()
        if ref_values.empty or cur_values.empty:
            continue
        statistic, pvalue = ks_2samp(ref_values, cur_values)
        results[column] = {
            "test": "kolmogorov_smirnov",
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "drift_detected": bool(pvalue < KS_PVALUE_THRESHOLD),
        }
        if results[column]["drift_detected"]:
            logging.warning("Numeric drift detected for %s (pvalue=%.6f)", column, pvalue)
        else:
            logging.info("Numeric drift ok for %s (pvalue=%.6f)", column, pvalue)
    return results


def categorical_drift(reference, current, categorical_columns):
    # Categorical features use chi-square to compare category frequencies.
    logging.info("Running categorical drift checks on %s columns", len(categorical_columns))
    results = {}
    for column in categorical_columns:
        ref_counts = reference[column].fillna("__missing__").value_counts()
        cur_counts = current[column].fillna("__missing__").value_counts()
        categories = sorted(set(ref_counts.index) | set(cur_counts.index))
        observed = np.array([cur_counts.get(category, 0) for category in categories], dtype=float)
        expected = np.array([ref_counts.get(category, 0) for category in categories], dtype=float)
        expected = expected / max(expected.sum(), 1) * max(observed.sum(), 1)
        expected = np.where(expected == 0, 1e-6, expected)
        statistic, pvalue = chisquare(f_obs=observed, f_exp=expected)
        results[column] = {
            "test": "chi_square",
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "drift_detected": bool(pvalue < CHI_PVALUE_THRESHOLD),
        }
        if results[column]["drift_detected"]:
            logging.warning("Categorical drift detected for %s (pvalue=%.6f)", column, pvalue)
        else:
            logging.info("Categorical drift ok for %s (pvalue=%.6f)", column, pvalue)
    return results


def performance_on_new_data(current):
    # If new data includes labels, calculate live performance against the current model.
    logging.info("Checking whether labelled new-data performance can be calculated")
    x_new_path = ARTIFACTS_DIR / "data" / "X_new.npy"
    y_new_path = ARTIFACTS_DIR / "data" / "y_new.npy"
    active_model_path = ARTIFACTS_DIR / "metadata" / "active_model.json"
    if TARGET_COLUMN not in current.columns or not x_new_path.exists() or not y_new_path.exists() or not active_model_path.exists():
        reason = "New data labels or model artefacts are unavailable."
        logging.info("Skipping new-data performance check: %s", reason)
        return {"available": False, "reason": reason}

    active_model = json.loads(active_model_path.read_text(encoding="utf-8"))
    model_path = PROJECT_ROOT / active_model["model_path"]
    x_new = np.load(x_new_path)
    y_new = np.load(y_new_path)
    logging.info("Loaded labelled new data arrays: X_new=%s y_new=%s", x_new.shape, y_new.shape)
    logging.info("Getting model predictions for new data")

    if active_model["model_type"] == "ann":
        import tensorflow as tf

        logging.info("Loading ANN model from %s", model_path)
        model = tf.keras.models.load_model(model_path)
        expected_features = model.input_shape[-1]
        if expected_features != x_new.shape[1]:
            reason = (
                f"Skipping new-data performance because the saved ANN expects "
                f"{expected_features} features but current preprocessing produced {x_new.shape[1]}."
            )
            logging.warning(reason)
            return {"available": False, "reason": reason}
        y_pred = np.argmax(model.predict(x_new, verbose=0), axis=1)
    else:
        logging.info("Loading Random Forest model from %s", model_path)
        model = joblib.load(model_path)
        expected_features = getattr(model, "n_features_in_", None)
        if expected_features is not None and expected_features != x_new.shape[1]:
            reason = (
                f"Skipping new-data performance because the saved Random Forest expects "
                f"{expected_features} features but current preprocessing produced {x_new.shape[1]}."
            )
            logging.warning(reason)
            return {"available": False, "reason": reason}
        y_pred = model.predict(x_new)

    accuracy = float(accuracy_score(y_new, y_pred))
    macro_f1 = float(f1_score(y_new, y_pred, average="macro"))
    logging.info("New-data performance: accuracy=%.4f macro_f1=%.4f", accuracy, macro_f1)
    return {
        "available": True,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "samples": int(len(y_new)),
        "alert": bool(accuracy < ACCURACY_ALERT_THRESHOLD),
    }


def write_dashboard(metrics):
    # Create a lightweight monitoring dashboard for report submission.
    logging.info("Writing monitoring dashboard HTML")
    drifted = metrics["summary"]["drifted_features"]
    alert_items = "\n".join(f"<li>{item}</li>" for item in metrics["alerts"]) or "<li>No active alerts</li>"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Monitoring Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 16px; max-width: 900px; }}
    .tile {{ border: 1px solid #d5d8dc; padding: 16px; border-radius: 6px; }}
    .value {{ font-size: 28px; font-weight: 700; }}
    pre {{ background: #f8f9f9; padding: 16px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Monitoring Dashboard</h1>
  <div class="grid">
    <div class="tile"><div>Current Rows</div><div class="value">{metrics["data_quality"]["rows"]}</div></div>
    <div class="tile"><div>Drifted Features</div><div class="value">{drifted}</div></div>
    <div class="tile"><div>Duplicate Rows</div><div class="value">{metrics["data_quality"]["duplicate_rows"]}</div></div>
  </div>
  <h2>Alerts</h2>
  <ul>{alert_items}</ul>
  <h2>Latest Metrics</h2>
  <pre>{json.dumps(metrics, indent=2)}</pre>
</body>
</html>
"""
    save_path = REPORTS_DIR / "monitoring_dashboard.html"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")


def write_retraining_flags(metrics):
    # Save simple trigger files, following the teacher example style.
    retrain_needed = metrics["summary"]["monitoring_status"] == "alert"
    flag_value = "true" if retrain_needed else "false"

    (PROJECT_ROOT / "retrain_needed.txt").write_text(flag_value, encoding="utf-8")
    (PROJECT_ROOT / "drift_status.txt").write_text(flag_value, encoding="utf-8")
    save_json(
        {
            "retrain_needed": retrain_needed,
            "reason": metrics["alerts"],
            "generated_at_utc": metrics["generated_at_utc"],
        },
        ARTIFACTS_DIR / "metadata" / "retraining_trigger.json",
    )
    logging.info("Saved retrain_needed.txt with value: %s", flag_value)
    logging.info("Saved drift_status.txt with value: %s", flag_value)
    logging.info("Saved retraining trigger metadata to artifacts/metadata/retraining_trigger.json")


def main():
    configure_logging()
    logging.info("=" * 70)
    logging.info("STARTING OBESITY DATA MONITORING")
    logging.info("=" * 70)
    logging.info("Thresholds: ks=%s chi_square=%s missing_rate=%s accuracy=%s",
                 KS_PVALUE_THRESHOLD, CHI_PVALUE_THRESHOLD, MISSING_RATE_THRESHOLD, ACCURACY_ALERT_THRESHOLD)

    reference, current, current_path = load_monitoring_data()
    common_columns = [column for column in reference.columns if column in current.columns and column != TARGET_COLUMN]
    numeric_columns = [column for column in common_columns if pd.api.types.is_numeric_dtype(reference[column])]
    categorical_columns = [column for column in common_columns if column not in numeric_columns]
    logging.info("Common monitoring columns: %s", len(common_columns))
    logging.info("Numeric columns: %s", numeric_columns)
    logging.info("Categorical columns: %s", categorical_columns)

    # Combine data quality, drift, and performance signals into one monitoring output.
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_data": str(PROJECT_ROOT / "train" / "train.csv"),
        "current_data": str(current_path),
        "data_quality": data_quality_report(current),
        "numeric_drift": numeric_drift(reference, current, numeric_columns),
        "categorical_drift": categorical_drift(reference, current, categorical_columns),
    }
    drifted_features = [
        column
        for group in ("numeric_drift", "categorical_drift")
        for column, result in metrics[group].items()
        if result["drift_detected"]
    ]
    alerts = []
    if drifted_features:
        alerts.append(f"Drift detected in: {', '.join(drifted_features)}")
        logging.warning("Drift detected in features: %s", drifted_features)
    if metrics["data_quality"]["columns_above_missing_threshold"]:
        alerts.append(
            "High missing rate in: "
            + ", ".join(metrics["data_quality"]["columns_above_missing_threshold"])
        )

    performance = performance_on_new_data(current)
    metrics["performance"] = performance
    if performance.get("alert"):
        alerts.append(f"New data accuracy below threshold: {performance['accuracy']:.4f}")

    metrics["summary"] = {
        "drifted_features": len(drifted_features),
        "monitoring_status": "alert" if alerts else "ok",
    }
    metrics["alerts"] = alerts

    save_json(metrics, ARTIFACTS_DIR / "metrics" / "monitoring_metrics.json")
    logging.info("Saved monitoring metrics to artifacts/metrics/monitoring_metrics.json")
    save_json(metrics, REPORTS_DIR / "drift_report.json")
    logging.info("Saved drift report to reports/drift_report.json")
    save_json(metrics, MONITORING_DIR / "reports" / "latest_monitoring_report.json")
    logging.info("Saved latest monitoring report to monitoring/reports/latest_monitoring_report.json")
    write_dashboard(metrics)
    write_retraining_flags(metrics)

    if alerts:
        alert_path = MONITORING_DIR / "alerts" / f"alerts-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.json"
        save_json({"alerts": alerts, "source": str(current_path)}, alert_path)
        logging.warning("Saved alert history to %s", alert_path)
    DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_LOG_PATH.write_text(json.dumps({"drifted_features": drifted_features}, indent=2), encoding="utf-8")
    logging.info("Saved drift detection log to logs/drift_detection.log")
    logging.info("Monitoring complete for %s with status %s", current_path, metrics["summary"]["monitoring_status"])
    print(json.dumps(metrics["summary"], indent=2))


if __name__ == "__main__":
    main()
