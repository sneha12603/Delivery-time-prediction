import pandas as pd
import joblib
import logging
import mlflow
import dagshub
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import json
import tempfile

# initialize dagshub
dagshub.init(repo_owner='sneha12603', repo_name='Delivery-time-prediction', mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/sneha12603/Delivery-time-prediction.mlflow")

# set mlflow experiment name
mlflow.set_experiment("DVC Pipeline")

TARGET = "time_taken"

# create logger
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# create formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("The file to load does not exist")
        return None
    return df


def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model


def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name
    }
    with open(save_json_path, "w") as f:
        json.dump(info_dict, f, indent=4)


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent

    # train/test data paths
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path = root_path / "data" / "processed" / "test_trans.csv"

    # model path
    model_path = root_path / "models" / "model.joblib"

    # load data
    train_data = load_data(train_data_path)
    if train_data is None:
        exit(1)
    logger.info("Train data loaded successfully")

    test_data = load_data(test_data_path)
    if test_data is None:
        exit(1)
    logger.info("Test data loaded successfully")

    # split data
    X_train, y_train = make_X_and_y(train_data, TARGET)
    X_test, y_test = make_X_and_y(test_data, TARGET)
    logger.info("Data split completed")

    # load model
    model = load_model(model_path)
    logger.info("Model loaded successfully")

    # predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    logger.info("Prediction on data complete")

    # calculate errors and scores
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    mean_cv_score = -(cv_scores.mean())

    logger.info("Metrics calculated")

    # infer model signature
    model_signature = mlflow.models.infer_signature(
        model_input=X_train.sample(20, random_state=42),
        model_output=model.predict(X_train.sample(20, random_state=42))
    )

    # start mlflow run
    with mlflow.start_run() as run:
        mlflow.set_tag("model", "Food Delivery Time Regressor")
        mlflow.log_params(model.get_params())

        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("mean_cv_score", mean_cv_score)
        mlflow.log_metrics({f"CV_{i}": -score for i, score in enumerate(cv_scores)})

        # log input datasets
        train_data_input = mlflow.data.from_pandas(train_data, targets=TARGET)
        test_data_input = mlflow.data.from_pandas(test_data, targets=TARGET)
        mlflow.log_input(dataset=train_data_input, context="training")
        mlflow.log_input(dataset=test_data_input, context="validation")

        # Save model locally and log as artifacts to avoid registry errors
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_model_path = Path(tmp_dir) / "model"
            mlflow.sklearn.save_model(model, local_model_path, signature=model_signature)
            mlflow.log_artifacts(local_model_path, artifact_path="delivery_time_pred_model")

        # Log additional artifacts
        mlflow.log_artifact(root_path / "models" / "stacking_regressor.joblib")
        mlflow.log_artifact(root_path / "models" / "power_transformer.joblib")
        mlflow.log_artifact(root_path / "models" / "preprocessor.joblib")

        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Mlflow logging complete and model artifacts logged")

    # Save run info
    run_id = run.info.run_id
    model_name = "delivery_time_pred_model"
    save_json_path = root_path / "run_information.json"
    save_model_info(save_json_path=save_json_path,
                    run_id=run_id,
                    artifact_path=artifact_uri,
                    model_name=model_name)
    logger.info("Model information saved")
