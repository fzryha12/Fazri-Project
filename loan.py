# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# Load model dan preprocessing
interpreter = tf.lite.Interpreter(model_path="loan_prediction.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# CSS untuk tema futuristik
st.markdown("""
    <style>
        .stApp {
            background-color: #1a1a1a;
            color: white;
        }
        h1, h2, h3 {
            color: #3dfc89;
        }
        .stButton>button {
            background-color: #3dfc89;
            color: black;
            font-weight: bold;
            border-radius: 12px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Prediksi Risiko Pinjaman")

# Input pengguna
col1, col2 = st.columns(2)
with col1:
    married = st.selectbox("💍 Status Pernikahan", ['single', 'married'])
    house = st.selectbox("🏠 Kepemilikan Rumah", ['rented', 'owned', 'norent_noown'])
    car = st.selectbox("🚗 Kepemilikan Mobil", ['yes', 'no'])
    age = st.slider("📅 Usia", 18, 100, 30)
with col2:
    experience = st.number_input("🧠 Pengalaman Kerja (tahun)", min_value=0, max_value=50, value=5)
    job_years = st.number_input("🛠 Lama di Pekerjaan Saat Ini (tahun)", min_value=0, max_value=40, value=3)
    house_years = st.number_input("🏡 Lama Tinggal di Rumah (tahun)", min_value=0, max_value=40, value=5)

profession = st.text_input("👔 Profesi", "Software Engineer")
city = st.text_input("🌆 Kota", "Mumbai")
state = st.text_input("🌍 Provinsi", "Maharashtra")

cat_input = pd.DataFrame([{
    'Car_Ownership': car,
    'House_Ownership': house,
    'Married/Single': married,
    'Profession': profession,
    'CITY': city,
    'STATE': state
}])

# Susun ulang kolom sesuai urutan saat encoder.fit
expected_columns = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE']
cat_input = cat_input[expected_columns]

# Proses encoding
try:
    cat_encoded = encoder.transform(cat_input)
    if hasattr(cat_encoded, "toarray"):
        cat_encoded = cat_encoded.toarray()

    if hasattr(encoder, 'get_feature_names_out'):
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out())
    else:
        cat_encoded_df = pd.DataFrame(cat_encoded)
except Exception as e:
    st.error(f"Gagal encode input kategorikal: {e}")
    st.stop()

# Data numerik
income = st.number_input("💰 Penghasilan per Tahun", min_value=1000, max_value=1_000_000, value=50000)
numeric_input = pd.DataFrame([{
    'Id': 0,  
    'Income': income,
    'Age': age,
    'Experience': experience,
    'CURRENT_JOB_YRS': job_years,
    'CURRENT_HOUSE_YRS': house_years
}])

# Gabung dan scaling
full_input = pd.concat([numeric_input.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)

try:
    scaled_input = scaler.transform(full_input)
except Exception as e:
    st.error(f"Gagal scaling data input: {e}")
    st.stop()

# Prediksi saat tombol diklik
if st.button("🔮 Prediksi Risiko"):
    try:
        input_tensor = scaled_input.astype(np.float32)
        if len(input_tensor.shape) == 1:
            input_tensor = np.expand_dims(input_tensor, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output)
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        confidence = output[0][predicted_class]

        st.markdown(f"""
            <div style='
                background-color: #262626;
                border-left: 6px solid #3dfc89;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            '>
                <h3>💡 Hasil Prediksi</h3>
                <p><strong>Risiko:</strong> {predicted_label}</p>
                <p><strong>Probabilitas:</strong> {confidence:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
