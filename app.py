# =============================================
# STREAMLIT APP
# Klastering Zona Rawan Gempa Bumi di Indonesia
# KODE STABIL UNTUK PREDIKSI
# =============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Klaster Zona Rawan Gempa",
    layout="wide"
)

# =====================
# SESSION STATE (WAJIB DI SINI)
# =====================
if "history" not in st.session_state:
    st.session_state.history = []

# =====================
# LOAD MODEL, SCALER, DATASET
# =====================
@st.cache_resource
def load_artifacts():
    # load model & scaler
    kmeans = joblib.load("kmeans_gempa.pkl")
    scaler = joblib.load("scaler_gempa.pkl")

    # fitur (HARUS sama dengan training)
    feature_columns = [
        "magnitude",
        "depth",
        "phasecount",
        "azimuth_gap"
    ]

    # dataset
    df = pd.read_csv("indonesia_earthquake.csv")

    return kmeans, scaler, feature_columns, df


# =====================
# LOAD SEKALI SAJA
# =====================
kmeans, scaler, feature_cols, df_data = load_artifacts()

# median untuk input default
phasecount_median = df_data['phasecount'].median()
azimuth_gap_median = df_data['azimuth_gap'].median()

# =====================
# CSS
# =====================
# =====================
# STRONG EARTH-TONE UI
# =====================
st.markdown(
"""
<style>
:root {
    --bg-main: #e2d6c5;          /* background lebih gelap */
    --dark-green: #13261c;      /* hijau jauh lebih gelap */
    --green-2: #1f3d2b;
    --brown-main: #7a3e1d;      /* coklat lebih pekat */
    --brown-dark: #4b2412;
    --cream: #fff8f1;
    --accent: #e07a2f;
}

/* ===== APP ===== */
body, .stApp {
    background-color: var(--bg-main);
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ===== TITLE ===== */
h1 {
    color: var(--dark-green);
    font-weight: 900;
    letter-spacing: -1px;
}

/* ===== CARD TITLE ===== */
.card h2, .card h3, .card h4 {
    color: var(--dark-green);
    font-weight: 900;
}

/* ===== PARAGRAPH ===== */
.card p, .card li {
    font-size: 16px;
    line-height: 1.8;
    color: #1e1e1e;
}

/* ===== FEATURE CARD (HIJAU SOLID) ===== */
.feature-card {
    background: linear-gradient(
        135deg,
        #13261c,
        #1f3d2b
    );
    color: white;
    padding: 28px;
    border-radius: 22px;
    text-align: center;
    box-shadow: 0 16px 35px rgba(0,0,0,0.45);
    border: 3px solid #0e1b14;
    margin-bottom: 30px
}

.feature-card h4 {
    font-size: 15px;
    letter-spacing: 1px;
    font-weight: 800;
    color: #cfe5d6;
}

.feature-card p {
    font-size: 26px;
    font-weight: 900;
    margin: 0;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13261c, #1f3d2b);
}

section[data-testid="stSidebar"] * {
    color: #fff !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #e07a2f !important;
    font-weight: 900;
}

/* ===== RADIO MENU (NAV) ===== */
div[role="radiogroup"] label {
    background: #1f3d2b;
    color: white;
    padding: 14px 26px;
    border-radius: 16px;
    margin-right: 12px;
    font-weight: 900;
    border: 3px solid #0e1b14;
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}

div[role="radiogroup"] label:hover {
    background: var(--brown-main);
    color: white;
    transform: translateY(-2px);
}

/* ===== BUTTON ===== */
.stButton > button {
    background: linear-gradient(
        135deg,
        var(--brown-main),
        var(--brown-dark)
    );
    color: white;
    border-radius: 16px;
    padding: 14px 32px;
    font-size: 16px;
    font-weight: 900;
    border: none;
    box-shadow: 0 10px 25px rgba(0,0,0,0.45);
}

.stButton > button:hover {
    background: var(--accent);
    color: black;
}

/* ===== FIX RADIO MENU TEXT COLOR ===== */

/* teks default */
div[role="radiogroup"] label {
    color: white !important;
}

/* teks saat dipilih (checked) */
div[role="radiogroup"] label[data-checked="true"] {
    color: white !important;
    background: linear-gradient(135deg, #13261c, #1f3d2b);
    border-color: #e07a2f;
}

/* teks saat hover */
div[role="radiogroup"] label:hover {
    color: white !important;
}

/* icon bullet radio */
div[role="radiogroup"] svg {
    fill: white !important;
}

</style>
""",
unsafe_allow_html=True
)

# =====================
# GLOBAL HEADER
# =====================
st.markdown("""
<h1 style="text-align:center; margin-bottom:5px;">
Klastering Zona Rawan Gempa Bumi di Indonesia
</h1>
<p style="text-align:center; color:#4e2a1a; font-weight:600;">
Analisis Kerawanan Gempa Berbasis Machine Learning
</p>
<hr style="border:2px solid #1f2d24; margin:20px 0;">
""", unsafe_allow_html=True)


# =====================
# SIDEBAR NAVIGATION
# =====================
with st.sidebar:
    st.markdown("## Menu Navigasi")
    st.markdown("---")

    menu = st.radio(
        "Pilih Halaman",
        [
            "Beranda",
            "Dataset",
            "Prediksi Klaster",
            "Riwayat Prediksi",
            "Peta Persebaran Klaster"
        ]
    )

    st.markdown("---")
    st.caption("Dashboard Klaster Gempa Indonesia")


# =====================
# BERANDA
# =====================
if menu == "Beranda":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Tentang Aplikasi")
    st.write(
        "Aplikasi ini dirancang sebagai **dashboard analisis kebencanaan** "
        "yang bertujuan untuk mengelompokkan wilayah di Indonesia berdasarkan "
        "**tingkat kerawanan gempa bumi**, menggunakan pendekatan "
        "*unsupervised machine learning*."
    )

    st.subheader("Pendekatan Ilmiah")
    st.write(
        "Metode **K-Means Clustering** digunakan untuk mengidentifikasi pola "
        "kemiripan karakteristik seismik antar wilayah, tanpa memerlukan "
        "label risiko sebelumnya."
    )

    st.markdown(
        """
        **Alasan pemilihan K-Means:**
        - Data bersifat **numerik & kontinu**
        - Cocok untuk **analisis eksploratif kebencanaan**
        - Komputasi efisien untuk dataset besar
        - Hasil klaster mudah **diinterpretasikan & dipetakan**
        """
    )

    st.subheader("Output Sistem")
    st.write(
        "Hasil akhir berupa **klaster zona rawan gempa** yang divisualisasikan "
        "dalam bentuk **peta interaktif**, guna mendukung analisis dan "
        "pengambilan keputusan mitigasi risiko."
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ===== FEATURE CARD =====
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>METODE</h4>
            <p>K-Means</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="feature-card">
            <h4>JUMLAH KLASTER</h4>
            <p>{kmeans.n_clusters}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>JENIS ANALISIS</h4>
            <p>Unsupervised</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Tujuan Pengembangan Sistem</h3>
        <ul>
            <li>Mengelompokkan wilayah rawan gempa secara objektif</li>
            <li>Mendukung mitigasi & perencanaan kebencanaan</li>
            <li>Menyediakan visualisasi klaster berbasis peta</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# =====================
# DATASET
# =====================
elif menu == "Dataset":
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Judul utama card
    st.markdown("<h2>Dataset Gempa Bumi</h2>", unsafe_allow_html=True)

    # Tampilkan dataframe
    st.dataframe(df_data, use_container_width=True)
    st.write("Jumlah data:", df_data.shape[0])

    # Penjelasan tentang dataset & variabel (SEMUA DALAM SATU CARD)
    st.markdown(
        """
        <hr>
        <h3>Tentang Dataset</h3>
        <p>Dataset ini merupakan hasil olahan dari data gempa bumi di Indonesia, awalnya dikumpulkan oleh <b>@Agapitus Keyka Vigiliant</b>.
        Dataset ini memberikan informasi mendetail setiap kejadian gempa, termasuk lokasi, waktu, magnitudo, serta parameter teknis seismik seperti azimuth gap dan phase count.
        Dataset ini dapat digunakan untuk analisis spasial, identifikasi zona rawan gempa, prediksi risiko, dan penelitian seismologi di Indonesia.</p>

        <h3>Penjelasan Variabel</h3>
        <ul>
            <li><b>eventID</b> : ID unik untuk setiap kejadian gempa, berfungsi sebagai pengenal yang membedakan satu gempa dengan yang lain dalam dataset.</li>
            <li><b>date</b> : Tanggal kejadian gempa yang dicatat secara resmi, dalam format dd/mm/yyyy, membantu menentukan urutan kronologis dan analisis tren gempa.</li>
            <li><b>time</b> : Waktu terjadinya gempa, dicatat dengan presisi jam:menit:detik, berguna untuk studi distribusi waktu dan pola aktivitas seismik.</li>
            <li><b>latitude</b> : Koordinat lintang lokasi gempa, menunjukkan posisi utara atau selatan ekuator, penting untuk pemetaan spasial kejadian gempa.</li>
            <li><b>longitude</b> : Koordinat bujur lokasi gempa, menunjukkan posisi timur atau barat Greenwich, digabungkan dengan latitude untuk menentukan titik geografi tepat gempa.</li>
            <li><b>magnitude</b> : Besaran magnitudo gempa yang mencerminkan kekuatan getaran, biasanya diukur dalam skala Richter atau metode lain yang relevan.</li>
            <li><b>mag_type</b> : Tipe magnitudo yang digunakan, contohnya:
                <ul>
                    <li><b>Mw</b> : Magnitudo ditentukan menggunakan momen seismik.</li>
                    <li><b>Mw(mB)</b> : Magnitudo dihitung dari integrasi ganda momen seismik dan gelombang broadband.</li>
                    <li><b>MLv</b> : Magnitudo dihitung dari gelombang lokal dengan komponen vertikal.</li>
                    <li><b>ML</b> : Magnitudo dihitung dari gelombang lokal dengan komponen horizontal.</li>
                    <li><b>mB</b> : Magnitudo menggunakan gelombang periode panjang atau broadband.</li>
                    <li><b>Mb</b> : Magnitudo menggunakan gelombang periode pendek.</li>
                </ul>
            </li>
            <li><b>depth</b> : Kedalaman gempa dalam kilometer, yang berpengaruh terhadap tingkat kerusakan dan persebaran energi gempa di permukaan.</li>
            <li><b>phasecount</b> : Jumlah fase seismik yang tercatat oleh stasiun, memberikan indikasi kualitas data dan jumlah stasiun yang mendeteksi gempa tersebut.</li>
            <li><b>azimuth_gap</b> : Selisih sudut azimuth antara dua stasiun seismik paling ekstrem dalam jaringan. Nilai ini bisa menjadi salah satu parameter untuk menilai potensi gempa, meskipun tidak dapat memprediksi secara keseluruhan karena banyak faktor lain yang mempengaruhi.</li>
            <li><b>location</b> : Nama lokasi atau wilayah di Indonesia tempat gempa terjadi, berguna untuk analisis spasial dan informasi publik terkait risiko gempa di daerah tersebut.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================
# PREDIKSI KLASTER
# =====================
elif menu == "Prediksi Klaster":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Prediksi Zona Rawan Gempa")

    # ===== INPUT SEISMIK (UNTUK MODEL) =====
    mag = st.number_input("Magnitudo", min_value=0.0, max_value=10.0, value=5.0)
    depth = st.number_input("Kedalaman Gempa (km)", min_value=0.0, max_value=700.0, value=10.0)

    phasecount = st.number_input(
        "Phase Count",
        min_value=0,
        max_value=200,
        value=int(phasecount_median)
    )

    azimuth_gap = st.number_input(
        "Azimuth Gap (°)",
        min_value=0.0,
        max_value=360.0,
        value=float(azimuth_gap_median)
    )

    # ===== INPUT METADATA (BUKAN UNTUK MODEL) =====
    lokasi = st.text_input(
        "Nama Daerah / Wilayah",
        placeholder="Contoh: Kabupaten Bantul, DIY"
    )

    if st.button("Prediksi Klaster"):
        # ===== PREDIKSI (HANYA 4 FITUR) =====
        X_input = pd.DataFrame(
            [[mag, depth, phasecount, azimuth_gap]],
            columns=feature_cols
        )

        X_scaled = scaler.transform(X_input)
        cluster = int(kmeans.predict(X_scaled)[0])

        # ===== LABEL MAKNA KLASTER (OPSIONAL TAPI BAGUS) =====
        cluster_label = {
            0: "Risiko Rendah",
            1: "Risiko Sedang",
            2: "Risiko Tinggi"
        }

        st.success(
            f"Gempa yang terjadi di wilayah **{lokasi if lokasi else 'Tidak diisi'}** "
            f"termasuk ke dalam **Klaster {cluster} ({cluster_label.get(cluster, 'Tidak diketahui')})**"
        )

        # ===== SIMPAN HASIL PREDIKSI (PENDATAAN) =====
        import os

        # SIMPAN KE SESSION (UNTUK RIWAYAT)
        st.session_state.history.append({
            "lokasi": lokasi,
            "magnitude": mag,
            "depth": depth,
            "phasecount": phasecount,
            "azimuth_gap": azimuth_gap,
            "cluster": cluster
        })

        log_data = pd.DataFrame([{
            "lokasi": lokasi,
            "magnitude": mag,
            "depth": depth,
            "phasecount": phasecount,
            "azimuth_gap": azimuth_gap,
            "cluster": cluster
        }])

        log_data.to_csv(
            "log_prediksi_gempa.csv",
            mode="a",
            header=not os.path.exists("log_prediksi_gempa.csv"),
            index=False
        )

        # tampilkan ringkasan
        st.markdown("###Ringkasan Prediksi")
        st.dataframe(log_data, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================
# RIWAYAT PREDIKSI
# =====================
elif menu == "Riwayat Prediksi":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("Riwayat Prediksi Gempa")

    # kalau belum ada data di session
    if "history" not in st.session_state or len(st.session_state.history) == 0:
        st.info("Belum ada data prediksi pada session ini.")
    else:
        # ambil data dari session
        history_df = pd.DataFrame(st.session_state.history)

        # bersihkan spasi lokasi
        history_df["lokasi"] = history_df["lokasi"].astype(str).str.strip()

        # ===== FILTER LOKASI =====
        lokasi_list = sorted(history_df["lokasi"].dropna().unique())

        selected_lokasi = st.selectbox(
            "Pilih Lokasi / Wilayah",
            options=["Semua Lokasi"] + lokasi_list
        )

        if selected_lokasi != "Semua Lokasi":
            filtered_df = history_df[history_df["lokasi"] == selected_lokasi]
        else:
            filtered_df = history_df

        # ===== INFO RINGKAS =====
        st.info(f"Menampilkan **{len(filtered_df)}** data prediksi")

        # ===== TABEL DATA =====
        st.dataframe(
            filtered_df.sort_values(by="cluster"),
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("⬇Unduh Riwayat Prediksi")

        # ===== DOWNLOAD CSV =====
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="riwayat_prediksi_session.csv",
            mime="text/csv"
        )

        # ===== DOWNLOAD EXCEL =====
        from io import BytesIO

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            filtered_df.to_excel(writer, index=False, sheet_name="Riwayat Prediksi")

        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name="riwayat_prediksi_session.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    st.markdown('</div>', unsafe_allow_html=True)


# =====================
# PETA PERSEBARAN
# =====================
elif menu == "Peta Persebaran Klaster":
    st.title("Peta Persebaran Klaster Gempa")

    m = folium.Map(location=[-2.5, 118], zoom_start=5)
    cluster = MarkerCluster().add_to(m)

    for _, row in df_data.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            popup=f"""
            Lokasi: {row.get('location', '-') }<br>
            Magnitudo: {row['magnitude']}<br>
            Kedalaman: {row['depth']} km
            """,
            color="red",
            fill=True,
            fill_opacity=0.7
        ).add_to(cluster)

    st_folium(m, height=650, use_container_width=True)
