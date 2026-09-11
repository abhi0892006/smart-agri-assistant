"""
Streamlit frontend for the Smart Agriculture Assistant.

Lets you test all four trained models (crop recommendation, fertilizer
recommendation, irrigation prediction, disease detection) plus the IoT
simulator through a simple browser UI instead of curl/Swagger.

Usage:
    streamlit run app.py
"""
import streamlit as st
from PIL import Image
import tempfile
from pathlib import Path

import config

st.set_page_config(page_title="Smart Agriculture Assistant", page_icon="🌾", layout="wide")


@st.cache_resource
def load_crop_recommender():
    from src.crop_recommendation.predict import CropRecommender
    return CropRecommender()


@st.cache_resource
def load_fertilizer_recommender():
    from src.fertilizer_recommendation.predict import FertilizerRecommender
    return FertilizerRecommender()


@st.cache_resource
def load_irrigation_predictor():
    from src.irrigation_prediction.predict import IrrigationPredictor
    return IrrigationPredictor()


@st.cache_resource
def load_disease_detector():
    from src.disease_detection.predict import DiseaseDetector
    return DiseaseDetector()


st.title("🌾 Smart Agriculture Assistant")
st.caption(
    "Crop recommendation, fertilizer recommendation, irrigation prediction, and "
    "multi-crop disease detection — all backed by models trained on real public datasets."
)

tab_crop, tab_fert, tab_irrigation, tab_disease, tab_iot = st.tabs(
    ["🌱 Crop Recommendation", "🧪 Fertilizer", "💧 Irrigation", "🍃 Disease Detection", "📡 IoT Simulator"]
)

# --- Crop Recommendation ---------------------------------------------------
with tab_crop:
    st.subheader("Recommend a crop from soil + weather readings")
    st.caption("Model: MLP trained on the Kaggle Crop Recommendation Dataset (99.4% test accuracy)")

    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0)
        P = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0)
        K = st.number_input("Potassium (K)", 0.0, 250.0, 43.0)
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    with col2:
        temperature = st.number_input("Temperature (°C)", -10.0, 55.0, 20.8)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 82.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 3500.0, 202.9)

    if st.button("Recommend Crop", type="primary"):
        try:
            recommender = load_crop_recommender()
            result = recommender.recommend({
                "N": N, "P": P, "K": K, "temperature": temperature,
                "humidity": humidity, "ph": ph, "rainfall": rainfall,
            })
            st.success(f"**Top recommendation: {result['top_prediction']}**")
            st.write("Top candidates:")
            for crop, prob in result["top_k"]:
                st.write(f"- {crop}: {prob:.1%}")
            if result["warnings"]:
                st.warning("\n".join(result["warnings"]))
        except FileNotFoundError as e:
            st.error(f"Model not trained yet: {e}")

# --- Fertilizer Recommendation ---------------------------------------------
with tab_fert:
    st.subheader("Recommend a fertilizer from crop + soil nutrients")
    st.caption("Model: MLP trained on the Kaggle Fertilizer Prediction Dataset (95% test accuracy, small dataset)")

    col1, col2 = st.columns(2)
    with col1:
        f_temp = st.number_input("Temperature (°C)", 0.0, 55.0, 26.0, key="f_temp")
        f_humidity = st.number_input("Humidity (%)", 0.0, 100.0, 52.0, key="f_hum")
        f_moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 38.0, key="f_moist")
        soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
    with col2:
        crop_type = st.selectbox(
            "Crop Type",
            ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley",
             "Wheat", "Millets", "Oil seeds", "Ground Nuts", "Pulses"],
        )
        nitrogen = st.number_input("Nitrogen", 0.0, 150.0, 37.0, key="f_n")
        potassium = st.number_input("Potassium", 0.0, 150.0, 0.0, key="f_k")
        phosphorous = st.number_input("Phosphorous", 0.0, 150.0, 0.0, key="f_p")

    if st.button("Recommend Fertilizer", type="primary"):
        try:
            recommender = load_fertilizer_recommender()
            result = recommender.recommend({
                "Temparature": f_temp, "Humidity": f_humidity, "Moisture": f_moisture,
                "Soil Type": soil_type, "Crop Type": crop_type,
                "Nitrogen": nitrogen, "Potassium": potassium, "Phosphorous": phosphorous,
            })
            st.success(f"**Top recommendation: {result['top_prediction']}**")
            st.write("Top candidates:")
            for fert, prob in result["top_k"]:
                st.write(f"- {fert}: {prob:.1%}")
            if result["warnings"]:
                st.warning("\n".join(result["warnings"]))
        except FileNotFoundError as e:
            st.error(f"Model not trained yet: {e}")

# --- Irrigation Prediction ---------------------------------------------------
with tab_irrigation:
    st.subheader("Predict whether irrigation is needed")
    st.caption(
        "Model: MLP trained on the GCEK IoT Community Irrigation Dataset "
        "(83% accuracy, 80% recall on the minority 'irrigate' class; small imbalanced dataset)"
    )

    col1, col2 = st.columns(2)
    with col1:
        crop_label = st.selectbox("Crop", ["Paddy", "Ground Nuts"])
        crop_code = 1 if crop_label == "Paddy" else 2
        crop_days = st.number_input("Days since planting", 0, 365, 15)
        soil_moisture = st.number_input("Soil Moisture (raw sensor reading)", 0.0, 1000.0, 230.0)
    with col2:
        soil_temp = st.number_input("Soil Temperature (°C)", 0.0, 60.0, 25.0)
        i_temp = st.number_input("Air Temperature (°C)", 0.0, 55.0, 30.0, key="i_temp")
        i_humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, key="i_hum")

    if st.button("Predict Irrigation Need", type="primary"):
        try:
            predictor = load_irrigation_predictor()
            result = predictor.predict({
                "CropType": crop_code, "CropDays": crop_days,
                "Soil Moisture": soil_moisture, "Soil Temperature": soil_temp,
                "Temperature": i_temp, "Humidity": i_humidity,
            })
            if result["irrigate"]:
                st.success(f"**Irrigate now** (confidence: {result['confidence']:.1%})")
            else:
                st.info(f"**No irrigation needed** (confidence: {result['confidence']:.1%})")
            st.write(f"Probabilities: No={result['probabilities']['No']:.1%}, "
                     f"Yes={result['probabilities']['Yes']:.1%}")
            if result["warnings"]:
                st.warning("\n".join(result["warnings"]))
        except FileNotFoundError as e:
            st.error(f"Model not trained yet: {e}")

# --- Disease Detection -------------------------------------------------------
with tab_disease:
    st.subheader("Detect plant disease from a leaf photo")
    st.caption(
        "Model: CNN trained on an 8-class subset of PlantVillage "
        "(4 crops × disease/healthy, 93.9% test accuracy)"
    )
    st.info(
        "Recognizes: Apple (scab/healthy), Corn (common rust/healthy), "
        "Potato (early blight/healthy), Tomato (early blight/healthy)."
    )

    uploaded_file = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", width=300)

        if st.button("Detect Disease", type="primary"):
            try:
                detector = load_disease_detector()
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name)
                    result = detector.predict(tmp.name)
                Path(tmp.name).unlink(missing_ok=True)

                st.success(f"**Crop: {result['crop']} — Condition: {result['condition']}**")
                st.write("Top candidates:")
                for cls, prob in result["top_k"]:
                    st.write(f"- {cls}: {prob:.1%}")
            except FileNotFoundError as e:
                st.error(f"Model not trained yet: {e}")

# --- IoT Simulator ------------------------------------------------------------
with tab_iot:
    st.subheader("Simulate a live sensor reading and run the full pipeline")
    st.caption(
        "The IoT layer has no real deployment (per the source paper), so this "
        "generates a physically-plausible reading — diurnal temperature/light cycle, "
        "gradual soil moisture depletion — rather than using real data."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_soil_type = st.selectbox("Soil Type (for fertilizer)", ["Sandy", "Loamy", "Black", "Red", "Clayey"], key="sim_soil")
    with col2:
        sim_crop_type = st.selectbox(
            "Crop Type (for fertilizer)",
            ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy", "Barley",
             "Wheat", "Millets", "Oil seeds", "Ground Nuts", "Pulses"],
            key="sim_crop",
        )
    with col3:
        sim_crop_label = st.selectbox("Crop (for irrigation)", ["Paddy", "Ground Nuts"], key="sim_irr_crop")
        sim_crop_code = 1 if sim_crop_label == "Paddy" else 2

    if st.button("Simulate Reading & Run Pipeline", type="primary"):
        from src.iot.simulator import IoTSensorSimulator
        from src.iot.adapters import (
            to_crop_recommendation_input,
            to_fertilizer_recommendation_input,
            to_irrigation_prediction_input,
        )

        sim = IoTSensorSimulator()
        reading = sim.read()

        st.write("**Simulated sensor reading:**")
        st.json(reading.as_dict())

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Crop Recommendation**")
            try:
                rec = load_crop_recommender()
                result = rec.recommend(to_crop_recommendation_input(reading))
                st.success(result["top_prediction"])
            except FileNotFoundError:
                st.error("Not trained")
        with col2:
            st.write("**Fertilizer Recommendation**")
            try:
                rec = load_fertilizer_recommender()
                result = rec.recommend(
                    to_fertilizer_recommendation_input(reading, sim_soil_type, sim_crop_type)
                )
                st.success(result["top_prediction"])
            except FileNotFoundError:
                st.error("Not trained")
        with col3:
            st.write("**Irrigation Prediction**")
            try:
                pred = load_irrigation_predictor()
                result = pred.predict(
                    to_irrigation_prediction_input(reading, sim_crop_code, 15)
                )
                st.success("Irrigate" if result["irrigate"] else "No irrigation needed")
            except FileNotFoundError:
                st.error("Not trained")
