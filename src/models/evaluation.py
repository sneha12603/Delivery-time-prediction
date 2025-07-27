import os
import pandas as pd
import joblib
import logging
import mlflow
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json
from yaml import safe_load

# initialize dagshub
import dagshub
dagshub.init(repo_owner='sneha12603', repo_name='Delivery-time-prediction', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/sneha12603/Delivery-time-prediction.mlflow")
mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

# logger setup
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("The file to load does not exist")
        return pd.DataFrame()

def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y

def load_model(model_path: Path):
    return joblib.load(model_path)

def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path, "w") as f:
        json.dump(info_dict, f, indent=4)

def load_params(params_path: Path):
    with open(params_path, "r") as file:
        return safe_load(file)

if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"
    model_path = root_path / "models" / "model.joblib"
    params_path = root_path / "params.yaml"

    # Load data
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    logger.info("Train and test data loaded successfully")

    X_train, y_train = make_X_and_y(train_data, TARGET)
    X_test, y_test = make_X_and_y(test_data, TARGET)

    # Load model
    model = load_model(model_path)
    logger.info("Model loaded successfully")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    mean_cv_score = -cv_scores.mean()

    metrics_dict = {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "mean_cv_score": mean_cv_score
    }

    with mlflow.start_run() as run:
        mlflow.set_tag("model", "Food Delivery Time Regressor")

        # Log model params
        mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metrics(metrics_dict)
        mlflow.log_metrics({f"CV_{i}": -score for i, score in enumerate(cv_scores)})

        # Inputs
        train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data, targets=TARGET)
        mlflow.log_input(train_data_input, context="training")
        mlflow.log_input(test_data_input, context="validation")

        # Model signature
        model_signature = mlflow.models.infer_signature(
            model_input=X_train.sample(20, random_state=42),
            model_output=model.predict(X_train.sample(20, random_state=42))
        )

        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="delivery_time_pred_model",
                signature=model_signature
            )
        except Exception as e:
            logger.warning(f"Could not log model due to: {e}")

        # Log artifacts
        for filename in ["stacking_regressor.joblib", "power_transformer.joblib", "preprocessor.joblib"]:
            path = root_path / "models" / filename
            if path.exists():
                mlflow.log_artifact(path)

        artifact_uri = mlflow.get_artifact_uri()

        # ✅ Register model in MLflow registry
        try:
            mlflow.register_model(
                model_uri=f"{artifact_uri}/delivery_time_pred_model",
                name="delivery_time_pred_model"
            )
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")

    # ✅ Save metadata
    save_model_info(
        save_json_path=root_path / "run_information.json",
        run_id=run.info.run_id,
        artifact_path=artifact_uri,
        model_name="delivery_time_pred_model"
    )

    # ✅ Save metrics to file
    with open(root_path / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    logger.info("Model evaluation, logging, and registration complete")
