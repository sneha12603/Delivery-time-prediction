import pandas as pd
import yaml
import joblib
import logging
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from pathlib import Path

TARGET = "time_taken"

# Logger setup
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("❌ The file to load does not exist")
        raise

def read_params(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def save_artifact(obj, save_dir: Path, filename: str):
    save_path = save_dir / filename
    joblib.dump(obj, save_path)
    logger.info(f"✅ Saved: {save_path}")

def make_X_and_y(data: pd.DataFrame, target_column: str):
    return data.drop(columns=[target_column]), data[target_column]

if __name__ == "__main__":
    root_path = Path(__file__).resolve().parents[2]
    data_path = root_path / "data" / "processed" / "train_trans.csv"
    params_file_path = root_path / "params.yaml"

    # Load training data
    training_data = load_data(data_path)
    X_train, y_train = make_X_and_y(training_data, TARGET)

    # Load model parameters
    model_params = read_params(params_file_path)["Train"]
    rf = RandomForestRegressor(**model_params["Random_Forest"])
    lgbm = LGBMRegressor(**model_params["LightGBM"])
    lr = LinearRegression()
    power_transform = PowerTransformer()

    # Build model
    stacking_reg = StackingRegressor(
        estimators=[("rf", rf), ("lgbm", lgbm)],
        final_estimator=lr,
        cv=5,
        n_jobs=-1
    )
    model = TransformedTargetRegressor(regressor=stacking_reg, transformer=power_transform)

    # Train model
    model.fit(X_train, y_train)
    logger.info("✅ Model training completed")

    # Save model and stacking regressor (optional)
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)

    save_artifact(model, model_save_dir, "model.joblib")
    save_artifact(stacking_reg, model_save_dir, "stacking_regressor.joblib")
    logger.info("✅ All artifacts saved successfully")
