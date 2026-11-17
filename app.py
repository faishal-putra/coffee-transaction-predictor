# ==============================================================
# Coffee Transaction Predictor
# Sistem Prediksi Total Transaksi Kedai Kopi Menggunakan Machine Learning
# ==============================================================
import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------
# Pengaturan halaman
# --------------------------------------------------------------
st.set_page_config(
    page_title="Coffee Transaction Predictor",
    page_icon="☕",
    layout="centered",
)

# Lokasi file model: satu folder dengan app.py
MODEL_PATH = Path(__file__).parent / "coffee_transaction_model.joblib"

OUTLET_OPTIONS = ["SHOP001", "SHOP002", "SHOP003"]
PAYMENT_OPTIONS = ["BCA", "Cash", "GoFood"]
NO_PRODUCT = "(Tidak ada)"

NAMA_HARI = {
    "Monday": "Senin", "Tuesday": "Selasa", "Wednesday": "Rabu",
    "Thursday": "Kamis", "Friday": "Jumat", "Saturday": "Sabtu",
    "Sunday": "Minggu",
}


# --------------------------------------------------------------
# Memuat model (hanya sekali, lalu disimpan di cache)
# --------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    data = joblib.load(MODEL_PATH)
    return (
        data["model"],
        data["mlb"],
        list(data["original_encoded_columns"]),
        list(data["final_feature_columns"]),
    )


# --------------------------------------------------------------
# Preprocessing: sama seperti saat training di Google Colab
# --------------------------------------------------------------
def build_features(outlet, products, payment, tanggal, waktu,
                   mlb, original_cols, final_cols):
    # 1) Fitur awal (6 kolom)
    raw = pd.DataFrame([{
        "Outlet": outlet,
        "TotalItem": len(products),
        "PaymentMethod": payment,
        "Hour": waktu.hour,
        "Month": tanggal.month,
        "DayOfWeek": tanggal.strftime("%A"),  # nama hari bahasa Inggris
    }])

    # 2) One-Hot Encoding -> 16 kolom, urutan mengikuti original_encoded_columns
    encoded = pd.get_dummies(raw, columns=["Outlet", "PaymentMethod", "DayOfWeek"])
    encoded = encoded.reindex(columns=original_cols, fill_value=0).astype(int)

    # 3) Fitur produk memakai mlb yang sudah tersimpan (tanpa fit ulang) -> 94 kolom
    prod_array = mlb.transform([list(products)])
    prod_df = pd.DataFrame(prod_array, columns=mlb.classes_, index=encoded.index)

    # 4) Gabungkan -> 110 kolom, urutan mengikuti final_feature_columns
    features = pd.concat([encoded, prod_df], axis=1)
    features = features.reindex(columns=final_cols, fill_value=0)
    return features


def format_rupiah(value):
    return "Rp" + f"{value:,.0f}".replace(",", ".")


# --------------------------------------------------------------
# Muat model, tampilkan pesan jelas jika gagal
# --------------------------------------------------------------
try:
    model, mlb, original_cols, final_cols = load_artifacts()
except FileNotFoundError:
    st.error(
        "File `coffee_transaction_model.joblib` tidak ditemukan. "
        "Pastikan file tersebut berada di folder yang sama dengan `app.py`."
    )
    st.stop()
except Exception as e:
    st.error(f"Model gagal dimuat: {e}")
    st.stop()

product_list = list(mlb.classes_)  # 94 produk, langsung dari model

# --------------------------------------------------------------
# Tampilan
# --------------------------------------------------------------
st.title("☕ Coffee Transaction Predictor")
st.caption("Sistem Prediksi Total Transaksi Kedai Kopi Menggunakan Machine Learning")
st.write(
    "Masukkan informasi transaksi di bawah ini, lalu tekan tombol prediksi "
    "untuk melihat perkiraan total transaksi."
)

st.subheader("Input Transaksi")

outlet = st.selectbox("Pilih Outlet", OUTLET_OPTIONS)
produk_1 = st.selectbox("Pilih Produk 1", product_list)
produk_2 = st.selectbox("Pilih Produk 2 (opsional)", [NO_PRODUCT] + product_list)
payment = st.selectbox("Metode Pembayaran", PAYMENT_OPTIONS)

col1, col2 = st.columns(2)
with col1:
    tanggal = st.date_input("Tanggal Transaksi", value=datetime.date.today())
with col2:
    waktu = st.time_input("Waktu Transaksi", value=datetime.time(10, 0))

if st.button("Prediksi Total Transaksi", type="primary", use_container_width=True):
    if produk_2 == produk_1:
        st.error("Produk 1 dan Produk 2 tidak boleh sama. Pilih produk yang berbeda atau kosongkan Produk 2.")
        st.stop()

    produk_terpilih = [produk_1]
    if produk_2 != NO_PRODUCT:
        produk_terpilih.append(produk_2)

    features = build_features(
        outlet, produk_terpilih, payment, tanggal, waktu,
        mlb, original_cols, final_cols,
    )

    # Pemeriksaan keamanan: harus 110 fitur dengan urutan yang sama persis
    assert features.shape == (1, 110), f"Jumlah fitur salah: {features.shape}"
    assert list(features.columns) == final_cols, "Urutan kolom tidak sesuai"

    prediksi = float(model.predict(features)[0])
    prediksi = max(prediksi, 0.0)  # total transaksi tidak mungkin negatif

    st.subheader("Hasil Prediksi")
    st.metric("Perkiraan Total Transaksi", format_rupiah(prediksi))

    st.markdown("**Ringkasan transaksi**")
    st.markdown(
        f"- **Outlet:** {outlet}\n"
        f"- **Produk:** {', '.join(produk_terpilih)}\n"
        f"- **Metode pembayaran:** {payment}\n"
        f"- **Tanggal:** {tanggal.strftime('%d-%m-%Y')} "
        f"({NAMA_HARI[tanggal.strftime('%A')]})\n"
        f"- **Waktu:** {waktu.strftime('%H:%M')}"
    )

    st.info(
        "Hasil ini adalah **estimasi** dari model Machine Learning, "
        "bukan nilai pembayaran yang dijamin."
    )

    # Bagian pengujian (untuk Langkah 5)
    with st.expander("Detail teknis (untuk pengujian)"):
        st.write(f"Jumlah produk dalam model (mlb.classes_): **{len(product_list)}**")
        st.write(f"Jumlah fitur yang dikirim ke model: **{features.shape[1]}**")
        st.write(
            "Urutan kolom sama dengan final_feature_columns: "
            f"**{list(features.columns) == final_cols}**"
        )
        st.write(f"Prediksi mentah dari model.predict(): **{model.predict(features)[0]:.2f}**")
        aktif = features.loc[:, (features != 0).any()].T
        aktif.columns = ["Nilai"]
        st.write("Fitur yang bernilai tidak nol:")
        st.dataframe(aktif)

st.divider()
st.caption(
    "Model: Linear Regression dengan fitur produk transaksi · "
    "MAE Rp1.855,62 · RMSE Rp4.404,70 · R² 0,8491"
)