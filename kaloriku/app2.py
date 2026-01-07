import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json
import time

# ======================================================
# Page Configuration
# ======================================================
st.set_page_config(
    page_title="Kaloriku - AI Calorie Estimation",
    layout="centered"
)

# ======================================================
# Custom CSS (Meniru Tailwind UI)
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

:root {
    --primary: #2563EB;
    --bg-light: #F3F4F6;
    --card: #FFFFFF;
    --border: #E5E7EB;
    --text-muted: #6B7280;
}

.stApp {
    background-color: #F3F4F6;
}

/* === CARD UTAMA === */
section[data-testid="stVerticalBlock"] {
    background: white;
    border-radius: 1.5rem;
    padding: 2.5rem;
    max-width: 900px;
    margin: 0 auto 2rem auto;
    box-shadow: 0 25px 50px -12px rgba(0,0,0,.15);
}

/* === FILE UPLOADER === */
div[data-testid="stFileUploader"] {
    border: 3px dashed #D1D5DB;
    border-radius: 1rem;
    padding: 2.5rem;
    background: #F9FAFB;
    margin-top: -1.5rem;          /* tarik ke atas */
    text-align: center;
    color: #6B7280;
    font-size: 0.85rem;
}

/* tambahkan ruang bawah di uploader */
div[data-testid="stFileUploader"] {
    padding-bottom: 3.5rem;
    position: relative;
}

/* hide ugly text */
div[data-testid="stFileUploader"] small {
    display: none;
}

/* center text */
div[data-testid="stFileUploader"] label {
    justify-content: center;
    font-weight: 600;
}

/* primary button */
.stButton > button {
    background-color: #2563EB;
    color: white;
    font-weight: 600;
    padding: 0.85rem 1.25rem;
    border-radius: 0.75rem;
    width: 100%;
}
            
/* === CARD PREVIEW MODE === */
.preview-mode section[data-testid="stVerticalBlock"] {
    max-width: 1100px;
    padding: 2rem;
}

/* image preview */
.preview-image img {
    border-radius: 0.75rem;
}

/* info bar bawah */
.file-info {
    margin-top: 1rem;
    background: #4B5563;
    color: white;
    border-radius: 0.75rem;
    padding: 0.75rem 1rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}

/* container kanan (tombol) */
.file-action {
    min-width: 140px;
}

/* pastikan tombol di dalam bar */
.file-action .stButton > button {
    background-color: #2563EB;
    color: white;
    font-weight: 600;
    padding: 0.55rem 1rem;
    border-radius: 0.6rem;
    width: 100%;
}

.file-meta {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    overflow: hidden;
}

.file-name {
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 600px;
}

.file-size {
    font-size: 0.75rem;
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# Load Model
# ======================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/mobilenetv2_finetune_v3.h5")

model = load_model()

# ======================================================
# Load Calorie Data
# ======================================================
@st.cache_data
def load_calorie_data():
    df = pd.read_csv("data/kalori_fatsecret.csv")
    return df.set_index("label").to_dict("index")

kalori_map = load_calorie_data()

# ======================================================
# Load Class Names
# ======================================================
with open("model/class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# ======================================================
# Session State
# ======================================================
if "step" not in st.session_state:
    st.session_state.step = "upload"

# ======================================================
# Header
# ======================================================
st.markdown("""
<div style="text-align:center; margin-bottom:2.5rem;">
    <h1 style="font-size:2.8rem; font-weight:700; line-height:1.2;">
        Cek Estimasi Kalori Makananmu<br>
        <span style="color:#2563EB;">Hanya dengan Satu Foto</span>
    </h1>
    <p style="color:#6B7280; max-width:640px; margin:1rem auto;">
        Model AI kami dirancang untuk mendeteksi berbagai jenis makanan populer di Indonesia secara akurat
        dan memberikan estimasi kalori per 100 gram berdasarkan referensi FatSecret.
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# Main Card Wrapper
# ======================================================

# ======================================================
# STEP 1 — UPLOAD
# ======================================================
if st.session_state.step == "upload":

    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Silahkan Upload Gambar Kamu Di Sini!"
    )

    if uploaded_file is not None:
        try:
            # reset pointer file
            uploaded_file.seek(0)

            # validasi gambar
            image = Image.open(uploaded_file)
            image.verify()  # cek struktur file

            # buka ulang setelah verify
            uploaded_file.seek(0)
            image = Image.open(uploaded_file).convert("RGB")

            # valid → lanjut
            st.session_state.uploaded_file = uploaded_file
            st.session_state.image = image
            st.session_state.step = "preview"
            st.rerun()

        except Exception:
            st.error(
                "❌ File tidak dapat diproses sebagai gambar.\n\n"
                "Pastikan file berformat JPG atau PNG dan tidak rusak."
            )

# ======================================================
# STEP 2 — PREVIEW
# ======================================================
elif st.session_state.step == "preview":

    st.markdown(
        "<style>section[data-testid='stVerticalBlock']{max-width:1100px}</style>",
        unsafe_allow_html=True
    )

    uploaded_file = st.session_state.uploaded_file
    image = st.session_state.image

    st.image(image, use_container_width=True)

    col_info, col_btn = st.columns([4, 1])

    with col_info:
        st.markdown(
            f"""
            <div class="file-info">
                <div class="file-meta">
                    <div class="file-name">{uploaded_file.name}</div>
                    <div class="file-size">{uploaded_file.size / 1024:.2f} KB</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_btn:
        st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
        if st.button("Prediksi"):
            st.session_state.step = "loading"
            st.rerun()


# ======================================================
# STEP 3 — LOADING
# ======================================================
elif st.session_state.step == "loading":

    st.markdown(
        "<style>section[data-testid='stVerticalBlock']{max-width:700px}</style>",
        unsafe_allow_html=True
    )

    with st.spinner("Menganalisis gambar makanan..."):
        time.sleep(1.5)
        st.session_state.step = "result"
        st.rerun()


# ======================================================
# STEP 4 — RESULT
# ======================================================
elif st.session_state.step == "result":

    uploaded_file = st.session_state.uploaded_file
    image = st.session_state.image

    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    label = CLASS_NAMES[class_index]

    kalori = kalori_map[label]
    min_kal = kalori["kalori_min_100g"]
    max_kal = kalori["kalori_max_100g"]

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("<h3>Hasil Analisis</h3>", unsafe_allow_html=True)
        st.markdown(f"**{label.replace('_',' ').title()}**")
        st.write(f"Estimasi Kalori: {min_kal} – {max_kal} kkal / 100g")
        st.write(f"Confidence: {confidence:.2%}")

        st.progress(confidence)

        if st.button("Bersihkan & Analisis Lagi"):
            st.session_state.step = "upload"
            st.rerun()
