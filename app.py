import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used Car Price Prediction App")
st.markdown("Predict the resale value of your car instantly.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model("used_car_price_model.cbm")
    return model

model = load_model()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_car_data.csv")

df = load_data()

# ---------------- FEATURE ORDER (IMPORTANT) ----------------
feature_columns = [
    "location",
    "km_driven",
    "fuel_type",
    "transmission_type",
    "owner_type",
    "mileage",
    "engine",
    "max_power",
    "seats",
    "vehicle_age",
    "brand",
    "model",
    "seller_type"
]

# ---------------- DROPDOWN VALUES ----------------
brand_list = sorted(df["brand"].dropna().unique())
location_list = sorted(df["location"].dropna().unique())
fuel_list = sorted(df["fuel_type"].dropna().unique())
transmission_list = sorted(df["transmission_type"].dropna().unique())
owner_list = sorted(df["owner_type"].dropna().unique())
seller_list = sorted(df["seller_type"].dropna().unique())
seat_list = sorted(df["seats"].dropna().unique())

# ---------------- USER INPUT UI ----------------
col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", location_list)
    km_driven = st.number_input("Kilometers Driven", 0, 500000, step=1000)
    fuel_type = st.selectbox("Fuel Type", fuel_list)
    transmission_type = st.selectbox("Transmission Type", transmission_list)
    owner_type = st.selectbox("Owner Type", owner_list)
    seller_type = st.selectbox("Seller Type", seller_list)

with col2:
    mileage = st.number_input("Mileage (kmpl)", 5.0, 40.0, step=0.1)
    engine = st.number_input("Engine (CC)", 500, 5000)
    max_power = st.number_input("Max Power (bhp)", 20, 500)
    seats = st.selectbox("Seats", seat_list)
    vehicle_age = st.slider("Vehicle Age (Years)", 0, 25)

# Dynamic model selection
brand = st.selectbox("Brand", brand_list)
model_list = sorted(df[df["brand"] == brand]["model"].dropna().unique())
car_model = st.selectbox("Model", model_list)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Price"):

    input_dict = {
        "location": location,
        "km_driven": km_driven,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "owner_type": owner_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "vehicle_age": vehicle_age,
        "brand": brand,
        "model": car_model,
        "seller_type": seller_type
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure correct column order
    input_df = input_df[feature_columns]

    try:
        prediction = model.predict(input_df)[0]

        # If model was trained on log price
        if prediction < 1000:
            prediction = np.expm1(prediction)

        prediction = max(0, prediction)

        st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)