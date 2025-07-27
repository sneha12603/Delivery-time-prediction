from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional
from src.features.data_utils_clean import perform_cleaning_for_prediction

# Initialize FastAPI
app = FastAPI(title="Swiggy Delivery Time Prediction API")

# Load the trained model (which already includes the fitted PowerTransformer)
try:
    model = joblib.load("models/model.joblib")
except Exception as e:
    raise RuntimeError(f"❌ Could not load model: {e}")

# Define request schema
class DeliveryData(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

@app.get("/")
def home():
    return {"message": "✅ Swiggy Delivery Time Prediction API is running"}

@app.post("/predict")
def predict(data: DeliveryData):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Clean data using your custom cleaner
        cleaned_df = perform_cleaning_for_prediction(input_df)

        # Predict using the loaded model (includes power transformer inside)
        prediction = model.predict(cleaned_df)[0]

        return {
            "predicted_delivery_time_minutes": round(float(prediction), 2)
        }

    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}"
        }
