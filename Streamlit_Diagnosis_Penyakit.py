import streamlit as st
import pandas as pd
import pickle

# Memuat model, scaler, dan label encoder dari file
diagnosis_model = pickle.load(open('diagnosis_penyakit.sav','rb'))

# Fungsi untuk memprediksi penyakit berdasarkan gejala
def predict_disease(symptoms):
    # Membuat dataframe dari gejala
    input_data = pd.DataFrame([symptoms])
    
    # Memprediksi penyakit
    prediction = diagnosis_model.predict(input_data_scaled)
    return prediction[0]

# Title
st.title('Aplikasi Diagnosa Penyakit Berbasis Gejala')

# Input gejala
fever = st.selectbox('Demam (fever):', [0, 1])
cough = st.selectbox('Batuk (cough):', [0, 1])
sore_throat = st.selectbox('Sakit Tenggorokan (sore throat):', [0, 1])
shortness_of_breath = st.selectbox('Sesak Napas (shortness of breath):', [0, 1])
headache = st.selectbox('Sakit Kepala (headache):', [0, 1])

# Tombol untuk prediksi
if st.button('Diagnosa'):
    symptoms = {
        'fever': fever,
        'cough': cough,
        'sore_throat': sore_throat,
        'shortness_of_breath': shortness_of_breath,
        'headache': headache,
    }
    diagnosis = predict_disease(symptoms)
    st.write(f'Hasil Diagnosa: {diagnosis}')

# Menjalankan aplikasi dengan streamlit run app.py
