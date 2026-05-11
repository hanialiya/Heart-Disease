import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
log_model = joblib.load("logistic_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("❤️ Heart Disease Prediction App")

st.write("""
Aplikasi prediksi penyakit jantung menggunakan:
- Logistic Regression
- Random Forest
""")

# =========================
# PILIH MODEL
# =========================
model_choice = st.selectbox(
    "Pilih Model",
    ["Logistic Regression", "Random Forest"]
)

# =========================
# INPUT USER
# =========================
st.subheader("📋 Input Data Pasien")

age = st.slider("Age", 20, 80, 40)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

chest_pain = st.selectbox(
    "Chest Pain",
    ["Typical", "Atypical", "Non-anginal", "Asymptomatic"]
)

resting_bp = st.slider(
    "Resting Blood Pressure",
    80,
    200,
    120
)

cholesterol = st.slider(
    "Cholesterol",
    120,
    400,
    200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

max_hr = st.slider(
    "Max Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ["No", "Yes"]
)

smoking = st.selectbox(
    "Smoking",
    ["No", "Yes"]
)

bmi = st.slider(
    "BMI",
    15.0,
    45.0,
    25.0
)

family_history = st.selectbox(
    "Family History",
    ["No", "Yes"]
)

stress_level = st.slider(
    "Stress Level",
    1,
    10,
    5
)

physical_activity = st.selectbox(
    "Physical Activity",
    ["Rendah", "Sedang", "Tinggi"]
)

# =========================
# ENCODING
# =========================

# Gender
gender = 1 if gender == "Male" else 0

# Chest Pain
chest_pain_map = {
    "Typical": 0,
    "Atypical": 1,
    "Non-anginal": 2,
    "Asymptomatic": 3
}
chest_pain = chest_pain_map[chest_pain]

# Exercise Angina
exercise_angina = 1 if exercise_angina == "Yes" else 0

# Smoking
smoking = 1 if smoking == "Yes" else 0

# Family History
family_history = 1 if family_history == "Yes" else 0

# Physical Activity
physical_activity_map = {
    "Rendah": 0,
    "Sedang": 1,
    "Tinggi": 2
}
physical_activity = physical_activity_map[physical_activity]

# =========================
# DATAFRAME INPUT
# =========================
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "ChestPain": [chest_pain],
    "RestingBP": [resting_bp],
    "Cholesterol": [cholesterol],
    "FastingBS": [fasting_bs],
    "MaxHR": [max_hr],
    "ExerciseAngina": [exercise_angina],
    "Smoking": [smoking],
    "BMI": [bmi],
    "FamilyHistory": [family_history],
    "StressLevel": [stress_level],
    "PhysicalActivity": [physical_activity]
})

# =========================
# BUTTON PREDICT
# =========================
if st.button("🔍 Predict"):

    # =========================
    # LOGISTIC REGRESSION
    # =========================
    if model_choice == "Logistic Regression":

        input_scaled = scaler.transform(input_data)

        prediction = log_model.predict(input_scaled)[0]

        probability = log_model.predict_proba(
            input_scaled
        )[0][1]

    # =========================
    # RANDOM FOREST
    # =========================
    else:

        prediction = rf_model.predict(input_data)[0]

        probability = rf_model.predict_proba(
            input_data
        )[0][1]

    # =========================
    # HASIL PREDIKSI
    # =========================
    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:

        st.error(
            "⚠️ Pasien Berpotensi Mengalami Penyakit Jantung"
        )

    else:

        st.success(
            "✅ Pasien Tidak Berpotensi Penyakit Jantung"
        )

    st.write(
        f"### Probabilitas Penyakit Jantung: {probability:.2%}"
    )

    # =========================
    # REKOMENDASI KLINIS
    # =========================
    st.subheader("🩺 Rekomendasi Klinis")

    risk = probability

    if risk >= 0.75:

        st.error("Risiko tinggi penyakit jantung")

        st.write("""
        ### Disarankan:
        - Segera konsultasi ke dokter spesialis jantung
        - Lakukan pemeriksaan lanjutan
        - Kurangi makanan tinggi lemak dan kolesterol
        - Hindari rokok dan alkohol
        - Rutin memantau tekanan darah
        """)

    elif risk >= 0.50:

        st.warning("Risiko sedang penyakit jantung")

        st.write("""
        ### Disarankan:
        - Menjaga pola makan sehat
        - Rutin berolahraga
        - Mengurangi stres
        - Tidur cukup
        - Periksa kesehatan secara berkala
        """)

    else:

        st.success("Risiko rendah penyakit jantung")

        st.write("""
        ### Disarankan:
        - Tetap menjaga gaya hidup sehat
        - Pertahankan aktivitas fisik
        - Konsumsi makanan bergizi
        - Hindari kebiasaan merokok
        """)

    # =========================
    # TAMPILKAN DATA INPUT
    # =========================
    st.subheader("📋 Data Input Pasien")

    st.dataframe(input_data)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Heart Disease Prediction using Machine Learning")