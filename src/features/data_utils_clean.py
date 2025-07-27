import numpy as np
import pandas as pd


def change_column_names(data: pd.DataFrame):
    return (
        data.rename(str.lower, axis=1)
        .rename({
            "delivery_person_id": "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken"
        }, axis=1)
    )


def time_of_day(ser):
    return pd.cut(ser, bins=[0, 6, 12, 17, 20, 24], right=True,
                  labels=["after_midnight", "morning", "afternoon", "evening", "night"])


def clean_lat_long(data: pd.DataFrame, threshold=1):
    location_columns = ['restaurant_latitude', 'restaurant_longitude', 'delivery_latitude', 'delivery_longitude']
    return data.assign(**{
        col: np.where(data[col] < threshold, np.nan, data[col].values) for col in location_columns
    })


def calculate_haversine_distance(df):
    lat1 = df['restaurant_latitude']
    lon1 = df['restaurant_longitude']
    lat2 = df['delivery_latitude']
    lon2 = df['delivery_longitude']

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return df.assign(distance=distance)


def create_distance_type(data: pd.DataFrame):
    return data.assign(distance_type=pd.cut(data["distance"], bins=[0, 5, 10, 15, 25],
                                            right=False, labels=["short", "medium", "long", "very_long"]))


def perform_cleaning_for_prediction(data: pd.DataFrame):
    data = change_column_names(data)

    if 'id' in data.columns:
        data = data.drop(columns="id")

    data = (
        data
        .replace("NaN ", np.nan)
        .assign(
            city_name=lambda x: x['rider_id'].str.split("RES").str.get(0),
            age=lambda x: x['age'].astype(float),
            ratings=lambda x: x['ratings'].astype(float),
            restaurant_latitude=lambda x: x['restaurant_latitude'].abs(),
            restaurant_longitude=lambda x: x['restaurant_longitude'].abs(),
            delivery_latitude=lambda x: x['delivery_latitude'].abs(),
            delivery_longitude=lambda x: x['delivery_longitude'].abs(),
            order_date=lambda x: pd.to_datetime(x['order_date'], dayfirst=True),
            order_day=lambda x: x['order_date'].dt.day,
            order_month=lambda x: x['order_date'].dt.month,
            order_day_of_week=lambda x: x['order_date'].dt.day_name().str.lower(),
            is_weekend=lambda x: x['order_date'].dt.day_name().isin(["Saturday", "Sunday"]).astype(int),
            order_time=lambda x: pd.to_datetime(x['order_time'], format='mixed', errors='coerce'),
            order_picked_time=lambda x: pd.to_datetime(x['order_picked_time'], format='mixed', errors='coerce'),
            pickup_time_minutes=lambda x: ((x['order_picked_time'] - x['order_time']).dt.total_seconds() / 60).fillna(0),
            order_time_hour=lambda x: x['order_time'].dt.hour,
            order_time_of_day=lambda x: x['order_time_hour'].pipe(time_of_day),
            weather=lambda x: x['weather'].str.replace("conditions ", "", regex=False).str.lower().replace("nan", np.nan),
            traffic=lambda x: x['traffic'].str.rstrip().str.lower(),
            type_of_order=lambda x: x['type_of_order'].str.rstrip().str.lower(),
            type_of_vehicle=lambda x: x['type_of_vehicle'].str.rstrip().str.lower(),
            festival=lambda x: x['festival'].str.rstrip().str.lower(),
            city_type=lambda x: x['city_type'].str.rstrip().str.lower(),
            multiple_deliveries=lambda x: x['multiple_deliveries'].astype(float)
        )
        .drop(columns=["order_time", "order_picked_time"])
    )

    data = clean_lat_long(data)
    data = calculate_haversine_distance(data)
    data = create_distance_type(data)

    # 🛠️ Fill missing values to prevent prediction errors
    data = data.fillna({
        col: data[col].mode()[0] if data[col].dtype == "O" else data[col].median()
        for col in data.columns
        if data[col].isna().any()
    })

    return data
