import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "NObeyesdad"
PARAMS = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text(encoding="utf-8"))
RANDOM_STATE = PARAMS["data"]["random_seed"]
TEST_SIZE = PARAMS["data"]["test_size"]
LOG_PATH = PROJECT_ROOT / "logs" / "monitoring.log"


def configure_logging():
    # Match the teacher example style: write detailed stage messages to logs/monitoring.log.
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - preprocessing - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


BINARY_MAPPINGS = {
    "Gender": {"Female": 0, "Male": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1},
    "SMOKE": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},
}

ORDINAL_MAPPING = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3,
}

ENGINEERED_FEATURES = ["Unhealthy_Score", "Activity_Habits", "Eating_Quality"]
ENGINEERED_SOURCE_FEATURES = ["FAVC", "SMOKE", "CALC", "FAF", "TUE", "FCVC", "NCP"]


def make_one_hot_encoder():
    # Support both old and new scikit-learn versions.
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


def make_preprocessor(columns):
    # Use different scalers for different obesity features, based on their ranges.
    standard = [col for col in ["Height"] if col in columns]
    robust = [col for col in ["Age", "Weight"] if col in columns]
    minmax = [
        col
        for col in [
            "CH2O",
            "FAF",
            "FCVC",
            "NCP",
            "TUE",
            "Unhealthy_Score",
            "Activity_Habits",
            "Eating_Quality",
            "CAEC",
            "CALC",
        ]
        if col in columns
    ]
    binary = [
        col
        for col in ["Gender", "family_history_with_overweight", "FAVC", "SMOKE", "SCC"]
        if col in columns
    ]
    nominal = [col for col in ["MTRANS"] if col in columns]

    transformers = []
    if standard:
        transformers.append(("standard", StandardScaler(), standard))
    if robust:
        transformers.append(("robust", RobustScaler(), robust))
    if minmax:
        transformers.append(("minmax", MinMaxScaler(), minmax))
    if binary:
        transformers.append(("binary", "passthrough", binary))
    if nominal:
        transformers.append(("nominal", make_one_hot_encoder(), nominal))

    logging.info("Configured preprocessing transformers: %s", [name for name, _, _ in transformers])
    return ColumnTransformer(transformers=transformers)


def prepare_dataframe(df):
    # Convert string categories into numeric values before ColumnTransformer scaling.
    missing_columns = [col for col in [*BINARY_MAPPINGS, "CAEC", "CALC"] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    prepared = df.copy()

    for col, mapping in BINARY_MAPPINGS.items():
        prepared[col] = prepared[col].map(mapping)

    prepared["CAEC"] = prepared["CAEC"].map(ORDINAL_MAPPING)
    prepared["CALC"] = prepared["CALC"].map(ORDINAL_MAPPING)

    # Engineered features from the previous portfolio work.
    prepared["Unhealthy_Score"] = prepared["FAVC"] + prepared["SMOKE"] + (prepared["CALC"] / 3)
    prepared["Activity_Habits"] = prepared["FAF"] / (prepared["TUE"] + 1)
    prepared["Eating_Quality"] = prepared["FCVC"] / (prepared["NCP"] + 1)

    if prepared.isna().any().any():
        missing = prepared.columns[prepared.isna().any()].tolist()
        raise ValueError(f"Preprocessing created missing values in columns: {missing}")

    return prepared


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_array(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def get_processed_feature_names(preprocessor, original_columns):
    try:
        return preprocessor.get_feature_names_out(original_columns).tolist()
    except Exception:
        return list(original_columns)


def preprocess_training_data(input_path):
    logging.info("=" * 70)
    logging.info("PREPROCESSING OBESITY TRAINING DATA")
    logging.info("=" * 70)
    logging.info("Loading raw obesity dataset from %s", input_path)

    raw_df = pd.read_csv(input_path)
    if TARGET_COLUMN not in raw_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found in {input_path}")
    logging.info("Loaded raw dataset with shape %s", raw_df.shape)

    # Fit preprocessing only on the obesity raw dataset.
    logging.info("Preparing categorical and ordinal obesity features")
    prepared_df = prepare_dataframe(raw_df)
    feature_columns = [
        col
        for col in prepared_df.columns
        if col != TARGET_COLUMN and col not in ENGINEERED_SOURCE_FEATURES
    ]
    logging.info(
        "Using %s model feature columns after replacing %s source columns with engineered features",
        len(feature_columns),
        len(ENGINEERED_SOURCE_FEATURES),
    )

    x = prepared_df[feature_columns]
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(prepared_df[TARGET_COLUMN])
    logging.info("Encoded target classes: %s", target_encoder.classes_.tolist())

    preprocessor = make_preprocessor(feature_columns)
    logging.info("Fitting preprocessing pipeline")
    x_processed = preprocessor.fit_transform(x)
    logging.info("Processed feature matrix shape: %s", x_processed.shape)

    # Keep class distribution stable in train/test with stratify.
    logging.info("Splitting data with test_size=%s and random_seed=%s", TEST_SIZE, RANDOM_STATE)
    x_train, x_test, y_train, y_test = train_test_split(
        x_processed,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df, test_df = train_test_split(
        raw_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=raw_df[TARGET_COLUMN],
    )

    train_dir = PROJECT_ROOT / "train"
    test_dir = PROJECT_ROOT / "test"
    artifact_data_dir = PROJECT_ROOT / "artifacts" / "data"
    preprocessing_dir = PROJECT_ROOT / "artifacts" / "preprocessing"

    # Save human-readable splits for monitoring and generated numpy arrays for model training.
    logging.info("Saving train/test CSV files")
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_dir / "train.csv", index=False)
    test_df.to_csv(test_dir / "test.csv", index=False)

    logging.info("Saving processed numpy arrays to artifacts/data")
    save_array(x_train, artifact_data_dir / "X_train.npy")
    save_array(x_test, artifact_data_dir / "X_test.npy")
    save_array(y_train, artifact_data_dir / "y_train.npy")
    save_array(y_test, artifact_data_dir / "y_test.npy")
    save_pickle(preprocessor, preprocessing_dir / "preprocessor.pkl")
    save_pickle(target_encoder, preprocessing_dir / "label_encoder.pkl")

    # Store preprocessing metadata so evaluate.py can map class ids back to names.
    logging.info("Saving preprocessing metadata and fitted encoders")
    save_json(
        {
            "raw_feature_columns": feature_columns,
            "engineered_features": ENGINEERED_FEATURES,
            "engineered_source_features_excluded": ENGINEERED_SOURCE_FEATURES,
            "processed_feature_columns": get_processed_feature_names(preprocessor, feature_columns),
            "target_column": TARGET_COLUMN,
            "target_classes": target_encoder.classes_.tolist(),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "source_rows": int(len(raw_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        preprocessing_dir / "feature_columns.json",
    )

    logging.info("Preprocessing complete")
    logging.info("Input rows: %s", len(raw_df))
    logging.info("Training rows: %s", len(train_df))
    logging.info("Test rows: %s", len(test_df))
    logging.info("Processed training shape: %s", x_train.shape)
    logging.info("Processed test shape: %s", x_test.shape)


def preprocess_new_data(input_path):
    logging.info("=" * 70)
    logging.info("PREPROCESSING OPTIONAL NEW DATA")
    logging.info("=" * 70)
    logging.info("Looking for new data at %s", input_path)

    if not input_path.exists():
        logging.info("No new data found; skipping X_new.npy generation")
        return

    # Reuse the fitted preprocessing artefacts from the training stage.
    logging.info("Loading fitted preprocessor from artifacts/preprocessing/preprocessor.pkl")
    with (PROJECT_ROOT / "artifacts" / "preprocessing" / "preprocessor.pkl").open("rb") as file:
        preprocessor = pickle.load(file)

    logging.info("Loading new data from %s", input_path)
    raw_df = pd.read_csv(input_path)
    logging.info("Loaded new data with shape %s", raw_df.shape)
    has_target = TARGET_COLUMN in raw_df.columns
    logging.info("New data includes target column: %s", has_target)
    feature_df = raw_df.drop(columns=[TARGET_COLUMN]) if has_target else raw_df
    prepared_df = prepare_dataframe(feature_df)

    logging.info("Transforming new data with fitted preprocessing pipeline")
    x_new = preprocessor.transform(prepared_df)
    artifact_data_dir = PROJECT_ROOT / "artifacts" / "data"
    save_array(x_new, artifact_data_dir / "X_new.npy")
    logging.info("Saved new feature array to artifacts/data/X_new.npy with shape %s", x_new.shape)

    # Labels are optional for new data. If labels exist, monitoring can calculate live accuracy.
    if has_target:
        logging.info("Encoding and saving new data labels")
        with (PROJECT_ROOT / "artifacts" / "preprocessing" / "label_encoder.pkl").open("rb") as file:
            target_encoder = pickle.load(file)
        y_new = target_encoder.transform(raw_df[TARGET_COLUMN])
        save_array(y_new, artifact_data_dir / "y_new.npy")
        logging.info("Saved new labels to artifacts/data/y_new.npy with shape %s", y_new.shape)

    logging.info("New data preprocessing complete")
    logging.info("New rows: %s", len(raw_df))
    logging.info("Processed new data shape: %s", x_new.shape)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Preprocess obesity dataset for Task 3 pipeline.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "ObesityDatasetRaw.csv"),
        help="Path to the raw obesity dataset CSV.",
    )
    parser.add_argument(
        "--new-data",
        default=str(PROJECT_ROOT / "data" / "new_data.csv"),
        help="Optional new data CSV to transform with the fitted preprocessing artefacts.",
    )
    args = parser.parse_args()
    preprocess_training_data(Path(args.input))
    preprocess_new_data(Path(args.new_data))


if __name__ == "__main__":
    main()
