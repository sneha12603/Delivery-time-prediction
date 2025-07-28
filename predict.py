import pandas as pd
import joblib
from src.features.data_utils_clean import perform_cleaning_for_prediction

# Load new data
data = pd.read_csv("data/raw/new_input_data.csv")

# Clean the data
cleaned_data = perform_cleaning_for_prediction(data)

# Load the preprocessor and full model
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/model.joblib")  # ✅ use this, not stacking_regressor.joblib

# Transform features
X_transformed = preprocessor.transform(cleaned_data)

# Predict
predictions = model.predict(X_transformed)

# Append and save predictions
data['predicted_delivery_time'] = predictions
print(data[['predicted_delivery_time']])

# Save predictions
data.to_csv("data/predicted_output.csv", index=False)
print("\nPredicted delivery times saved to data/predicted_output.csv")
