# ==============================================================
# Coffee Transaction Predictor
# Sistem Prediksi Total Transaksi Kedai Kopi Menggunakan Machine Learning
# ==============================================================
import datetime
import html
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
    layout="wide",
)

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
# Gaya tampilan (CSS)
# --------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px;}
    #MainMenu, footer {visibility: hidden;}

    .hero {
        background: linear-gradient(135deg, #3b2314 0%, #6f4e37 55%, #a67c52 100%);
        border-radius: 20px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(59, 35, 20, 0.25);
    }
    .hero h1 {color: #ffffff; margin: 0; font-size: 2.3rem; font-weight: 800;}
    .hero p {color: #f3e5d8; margin: 0.4rem 0 0 0; font-size: 1.05rem;}
    .hero .badge {
        display: inline-block; margin-top: 1rem; padding: 0.25rem 0.8rem;
        background: rgba(255,255,255,0.18); color: #ffffff;
        border-radius: 999px; font-size: 0.8rem; letter-spacing: 0.03em;
    }

    .section-title {font-size: 1.15rem; font-weight: 700; margin-bottom: 0.2rem;}

    .result-card {
        background: linear-gradient(135deg, #3b2314 0%, #6f4e37 100%);
        border-radius: 18px; padding: 1.8rem; text-align: center;
        box-shadow: 0 8px 24px rgba(59, 35, 20, 0.25);
    }
    .result-card .label {color: #f3e5d8; font-size: 0.95rem; letter-spacing: 0.05em; text-transform: uppercase;}
    .result-card .amount {color: #ffffff; font-size: 3rem; font-weight: 800; margin: 0.3rem 0;}
    .result-card .sub {color: #e8d3bf; font-size: 0.85rem;}

    .placeholder-card {
        border: 2px dashed #a67c52; border-radius: 18px; padding: 2.5rem 1.5rem;
        text-align: center; color: #a67c52;
    }
    .placeholder-card .icon {font-size: 2.6rem;}

    .summary-row {
        display: flex; justify-content: space-between; gap: 1rem;
        padding: 0.55rem 0; border-bottom: 1px solid rgba(166,124,82,0.35);
    }
    .summary-row:last-child {border-bottom: none;}
    .summary-row .k {opacity: 0.7;}
    .summary-row .v {font-weight: 600; text-align: right;}

    div.stButton > button {
        border-radius: 12px; font-weight: 700; padding: 0.7rem 1rem;
        background: linear-gradient(135deg, #6f4e37, #3b2314); color: #ffffff; border: none;
    }
    div.stButton > button:hover {filter: brightness(1.15); color: #ffffff;}

    .stat-box {
        border: 1px solid rgba(166,124,82,0.45); border-radius: 14px;
        padding: 0.9rem; text-align: center;
    }
    .stat-box .n {font-size: 1.3rem; font-weight: 800; color: #a67c52;}
    .stat-box .t {font-size: 0.8rem; opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def summary_row(key, value):
    return (
        f'<div class="summary-row"><span class="k">{html.escape(key)}</span>'
        f'<span class="v">{html.escape(value)}</span></div>'
    )


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
# Header
# --------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>☕ Coffee Transaction Predictor</h1>
        <p>Sistem Prediksi Total Transaksi Kedai Kopi Menggunakan Machine Learning</p>
        <span class="badge">Linear Regression · 110 fitur · R² 0,8491</span>
    </div>
    """,
    unsafe_allow_html=True,
)

col_input, col_result = st.columns([1, 1], gap="large")

# --------------------------------------------------------------
# Kolom kiri: input transaksi
# --------------------------------------------------------------
with col_input:
    with st.container(border=True):
        st.markdown('<div class="section-title">📝 Input Transaksi</div>', unsafe_allow_html=True)
        st.caption("Isi informasi transaksi, lalu tekan tombol prediksi.")

        outlet = st.selectbox("Pilih Outlet", OUTLET_OPTIONS)
        produk_1 = st.selectbox("Pilih Produk 1", product_list)
        produk_2 = st.selectbox("Pilih Produk 2 (opsional)", [NO_PRODUCT] + product_list)
        payment = st.selectbox("Metode Pembayaran", PAYMENT_OPTIONS)

        c1, c2 = st.columns(2)
        with c1:
            tanggal = st.date_input("Tanggal Transaksi", value=datetime.date.today())
        with c2:
            waktu = st.time_input("Waktu Transaksi", value=datetime.time(10, 0))

        klik = st.button("Prediksi Total Transaksi", width="stretch")

# --------------------------------------------------------------
# Kolom kanan: hasil prediksi
# --------------------------------------------------------------
with col_result:
    st.markdown('<div class="section-title">📊 Hasil Prediksi</div>', unsafe_allow_html=True)

    if not klik:
        st.markdown(
            """
            <div class="placeholder-card">
                <div class="icon">🧾</div>
                <div><b>Belum ada prediksi</b></div>
                <div>Isi form di sebelah kiri lalu tekan<br>"Prediksi Total Transaksi".</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif produk_2 == produk_1:
        st.error("Produk 1 dan Produk 2 tidak boleh sama. Pilih produk yang berbeda atau kosongkan Produk 2.")
    else:
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

        prediksi_mentah = float(model.predict(features)[0])
        prediksi = max(prediksi_mentah, 0.0)  # total transaksi tidak mungkin negatif

        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Perkiraan Total Transaksi</div>
                <div class="amount">{format_rupiah(prediksi)}</div>
                <div class="sub">Estimasi dari model Machine Learning</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Ringkasan transaksi**")
            st.markdown(
                summary_row("Outlet", outlet)
                + summary_row("Produk", ", ".join(produk_terpilih))
                + summary_row("Metode pembayaran", payment)
                + summary_row(
                    "Tanggal",
                    f"{tanggal.strftime('%d-%m-%Y')} ({NAMA_HARI[tanggal.strftime('%A')]})",
                )
                + summary_row("Waktu", waktu.strftime("%H:%M")),
                unsafe_allow_html=True,
            )

        st.info(
            "Hasil ini adalah **estimasi** dari model Machine Learning, "
            "bukan nilai pembayaran yang dijamin."
        )

        # Bagian pengujian
        with st.expander("🔧 Detail teknis (untuk pengujian)"):
            t1, t2, t3 = st.columns(3)
            t1.markdown(
                f'<div class="stat-box"><div class="n">{len(product_list)}</div>'
                f'<div class="t">Produk di model</div></div>',
                unsafe_allow_html=True,
            )
            t2.markdown(
                f'<div class="stat-box"><div class="n">{features.shape[1]}</div>'
                f'<div class="t">Fitur ke model</div></div>',
                unsafe_allow_html=True,
            )
            urutan_ok = list(features.columns) == final_cols
            t3.markdown(
                f'<div class="stat-box"><div class="n">{urutan_ok}</div>'
                f'<div class="t">Urutan kolom sama</div></div>',
                unsafe_allow_html=True,
            )
            st.write(f"Prediksi mentah dari model.predict(): **{prediksi_mentah:.2f}**")
            aktif = features.loc[:, (features != 0).any()].T
            aktif.columns = ["Nilai"]
            st.write("Fitur yang bernilai tidak nol:")
            st.dataframe(aktif, width="stretch")

st.divider()
st.caption(
    "Model: Linear Regression dengan fitur produk transaksi · "
    "MAE Rp1.855,62 · RMSE Rp4.404,70 · R² 0,8491"
)
