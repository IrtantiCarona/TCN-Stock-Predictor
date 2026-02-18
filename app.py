import math
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tcn import TCN
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Huber
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Harga Saham Menggunakan Model TCN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0a0e1a; }
[data-testid="stAppViewContainer"] { background: #0a0e1a; }
[data-testid="stSidebar"] { background: #0d1220 !important; border-right: 1px solid #1e2d4a; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.metric-card {
    background: linear-gradient(135deg, #0d1220 0%, #111827 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a7aad;
    margin-bottom: 0.3rem;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #e8f4ff;
    font-family: 'Space Mono', monospace;
}

.pred-box {
    background: linear-gradient(135deg, #0a1628 0%, #0e2040 100%);
    border: 2px solid #2563eb;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.pred-box::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at center, rgba(37,99,235,0.08) 0%, transparent 60%);
}
.pred-label {
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3b82f6;
    font-family: 'Space Mono', monospace;
}
.pred-value {
    font-size: 3rem;
    font-weight: 700;
    color: #60a5fa;
    font-family: 'Space Mono', monospace;
    margin: 0.5rem 0;
}
.pred-ticker { font-size: 0.85rem; color: #64748b; }

.stButton > button {
    background: linear-gradient(90deg, #1d4ed8, #2563eb);
    color: white; border: none; border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem; letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem; width: 100%; transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    transform: translateY(-1px);
}
hr { border-color: #1e2d4a !important; }

.modal-step {
    display: flex;
    gap: 0.9rem;
    margin-bottom: 1rem;
    align-items: flex-start;
}
.modal-step-num {
    background: rgba(37,99,235,0.2);
    border: 1px solid rgba(37,99,235,0.4);
    color: #60a5fa;
    font-family: "Space Mono", monospace;
    font-size: 0.75rem;
    font-weight: 700;
    border-radius: 50%;
    width: 26px; height: 26px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.modal-step-text {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
}
.modal-step-text strong { color: #e2e8f0; }
.modal-warning {
    background: rgba(234,179,8,0.08);
    border: 1px solid rgba(234,179,8,0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #fbbf24;
    margin-top: 1.2rem;
    line-height: 1.6;
}
.modal-close-hint {
    text-align: center;
    margin-top: 1.4rem;
    font-size: 0.72rem;
    color: #334155;
    font-family: "Space Mono", monospace;
}

/* Styling st.dialog agar sesuai tema dark */
div[data-testid="stDialog"] > div > div {
    background: linear-gradient(160deg, #0d1628 0%, #0a1220 100%) !important;
    border: 1px solid #2563eb !important;
    border-radius: 16px !important;
}
div[data-testid="stDialog"] h1 {
    color: #e8f4ff !important;
    font-size: 1.1rem !important;
}
div[data-testid="stDialog"] button[data-testid="stBaseButton-headerNoPadding"] {
    color: #60a5fa !important;
}
.title-bar {
    display: flex; align-items: center; gap: 1rem;
    padding: 1.5rem 0 2rem 0;
    border-bottom: 1px solid #1e2d4a;
    margin-bottom: 2rem;
}
.title-text h1 { color: #e8f4ff; margin: 0; font-size: 1.6rem; line-height: 1.2; }
.title-text p  { color: #4a7aad; margin: 0; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── FIXED PARAMETERS (sesuai penelitian) ─────────────────────────────────────
WINDOW_SIZE  = 60
E_SLOVIN     = 0.05
HORIZON      = 1
NB_FILTERS   = 64
KERNEL_SIZE  = 3
NB_STACKS    = 1
DILATIONS    = [1, 2, 4, 8, 16, 32]
DROPOUT_RATE = 0.2
EPOCHS       = 80
BATCH_SIZE   = 32
PATIENCE     = 10

MODEL_PATH  = "tcn_model.keras"
SCALER_PATH = "scaler.pkl"


# ── LOAD MODEL & SCALER ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, f"File model '{MODEL_PATH}' tidak ditemukan."
    if not os.path.exists(SCALER_PATH):
        return None, None, f"File scaler '{SCALER_PATH}' tidak ditemukan."
    try:
        model = load_model(MODEL_PATH, custom_objects={"TCN": TCN, "Huber": Huber})
        scaler = joblib.load(SCALER_PATH)
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def create_windows(X, y, window_size):
    X_w, y_w = [], []
    for i in range(len(X) - window_size + 1):
        X_w.append(X[i: i + window_size])
        y_w.append(y[i + window_size - 1])
    return np.array(X_w), np.array(y_w)


@st.cache_data(show_spinner=False)
def load_data(ticker, start, end):
    # Tambah 1 hari ke end agar tanggal yang dipilih user ikut terambil
    # (yfinance menggunakan end secara eksklusif)
    end_inclusive = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(ticker, start=start, end=str(end_inclusive.date()), progress=False)
    return df[["Close"]]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi")
    st.markdown("---")

    st.markdown("**📊 Data Saham**")
    ticker = st.text_input("Kode Saham (Yahoo Finance)", value="BBRI.JK")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Mulai", value=pd.to_datetime("2018-01-01"))
    with col2:
        end_date = st.date_input("Selesai", value=pd.to_datetime("2025-12-31"))

    st.markdown("---")
    st.markdown("**🔧 Preprocessing**")
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
                padding:0.8rem 1rem;font-size:0.8rem;color:#94a3b8;line-height:1.9;">
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Window Size</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{WINDOW_SIZE}</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Margin Error (e)</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{E_SLOVIN}</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Horizon</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{HORIZON} hari</strong>
    </div>
    <p style="font-size:0.68rem;color:#334155;margin-top:0.4rem;font-style:italic;">
        🔒 Parameter dikunci sesuai penelitian
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🧠 Arsitektur TCN**")
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
                padding:0.8rem 1rem;font-size:0.8rem;color:#94a3b8;line-height:1.9;">
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Nb Filters</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{NB_FILTERS}</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Kernel Size</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{KERNEL_SIZE}</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Nb Stacks</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{NB_STACKS}</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Dilations</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{DILATIONS}</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Dropout Rate</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{DROPOUT_RATE}</strong>
    </div>
    <p style="font-size:0.68rem;color:#334155;margin-top:0.4rem;font-style:italic;">
        🔒 Parameter dikunci sesuai penelitian
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🏋️ Training**")
    st.markdown(f"""
    <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:8px;
                padding:0.8rem 1rem;font-size:0.8rem;color:#94a3b8;line-height:1.9;">
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Max Epochs</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{EPOCHS}</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Batch Size</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{BATCH_SIZE}</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Patience</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">{PATIENCE}</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Optimizer</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">Adam</strong><br>
        <span style="color:#4a7aad;font-family:'Space Mono',monospace;">Loss</span>
        &nbsp;&nbsp;→ <strong style="color:#60a5fa;">Huber (δ=1.0)</strong>
    </div>
    <p style="font-size:0.68rem;color:#334155;margin-top:0.4rem;font-style:italic;">
        🔒 Parameter dikunci sesuai penelitian
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("🔮  Jalankan Prediksi")


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-bar">
  <div><div style="font-size:2.8rem;line-height:1">📈</div></div>
  <div class="title-text">
    <h1>TCN Stock Price Predictor</h1>
    <p>Temporal Convolutional Network · Prediksi Harga Saham</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── POPUP PANDUAN (st.dialog) ────────────────────────────────────────────────
@st.dialog("📖 Panduan Penggunaan")
def show_guide_dialog():
    st.markdown("""
    <div class="modal-step">
      <div class="modal-step-num">1</div>
      <div class="modal-step-text">
        Masukkan <strong>kode saham</strong> di sidebar kiri (format Yahoo Finance,
        contoh: <code style="color:#7dd3fc;">BBRI.JK</code>, <code style="color:#7dd3fc;">TLKM.JK</code>).
      </div>
    </div>
    <div class="modal-step">
      <div class="modal-step-num">2</div>
      <div class="modal-step-text">
        Tentukan <strong>rentang tanggal</strong> data historis yang ingin digunakan.
        Disarankan minimal <strong>3 tahun</strong> agar hasil prediksi lebih akurat.
      </div>
    </div>
    <div class="modal-step">
      <div class="modal-step-num">3</div>
      <div class="modal-step-text">
        Tekan tombol <strong>🔮 Jalankan Prediksi</strong> di bagian bawah sidebar.
      </div>
    </div>
    <div class="modal-step">
      <div class="modal-step-num">4</div>
      <div class="modal-step-text">
        Lihat hasil prediksi harga saham untuk <strong>hari perdagangan berikutnya</strong>,
        beserta grafik perbandingan harga aktual vs prediksi dan evaluasi model (MAE, RMSE, MAPE).
      </div>
    </div>
    <div class="modal-warning" style="margin-top:1.2rem;">
      ⚠️ <strong>Perhatian:</strong> Sistem prediksi harga saham ini dikembangkan untuk keperluan akademik. Model yang digunakan masih bersifat eksperimental. Segala bentuk keputusan investasi yang diambil berdasarkan informasi dari sistem ini sepenuhnya menjadi tanggung jawab pengguna. Gunakan hasil prediksi secara bijak sebelum mengambil keputusan investasi
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✅  Mengerti, Mulai Gunakan", use_container_width=True, key="close_guide_btn"):
        st.rerun()

if "show_guide" not in st.session_state:
    st.session_state.show_guide = True

if st.session_state.show_guide:
    st.session_state.show_guide = False
    show_guide_dialog()

# Cek ketersediaan file model & scaler
model, scaler, err = load_artifacts()
if err:
    st.error(f"⚠️ {err}")
    st.markdown("""
    <div style="background:#1a0a0a;border:1px solid #5f1e1e;border-radius:10px;
                padding:1.5rem;margin-top:1rem;font-size:0.85rem;color:#94a3b8;line-height:1.8;">
        <strong style="color:#f87171;">Cara menyiapkan file model:</strong><br><br>
        <strong>1.</strong> Jalankan kode ini di <strong>Google Colab</strong> setelah training selesai:<br>
        <code style="background:#0d1220;padding:0.5rem 0.8rem;border-radius:6px;
                     display:block;margin:0.6rem 0;color:#7dd3fc;font-size:0.8rem;">
        model.save("tcn_model.keras")<br>
        import joblib<br>
        joblib.dump(scaler, "scaler.pkl")<br>
        from google.colab import files<br>
        files.download("tcn_model.keras")<br>
        files.download("scaler.pkl")
        </code>
        <strong>2.</strong> Letakkan kedua file tersebut di folder yang sama dengan <code>app.py</code>:<br>
        <code style="background:#0d1220;padding:0.5rem 0.8rem;border-radius:6px;
                     display:block;margin:0.6rem 0;color:#7dd3fc;font-size:0.8rem;">
        📁 project/<br>
        &nbsp;&nbsp;├── app.py<br>
        &nbsp;&nbsp;├── tcn_model.keras<br>
        &nbsp;&nbsp;└── scaler.pkl
        </code>
        <strong>3.</strong> Jalankan ulang aplikasi.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.success("✅ Model TCN berhasil dimuat. Masukkan kode saham dan tekan **🔮 Jalankan Prediksi**.")

if not run_btn:
    st.markdown("""
    <div style="background:#0d1220;border:1px dashed #1e3a5f;border-radius:14px;
                padding:3rem;text-align:center;margin-top:2rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">🔬</div>
        <p style="color:#4a7aad;font-family:'Space Mono',monospace;font-size:0.9rem;">
            Pilih kode saham & rentang tanggal di sidebar, lalu tekan<br>
            <strong style="color:#60a5fa;">🔮 Jalankan Prediksi</strong> untuk memulai.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── RUN PIPELINE ──────────────────────────────────────────────────────────────
with st.status("⏳ Memuat data saham...", expanded=True) as status:
    try:
        df_tcn = load_data(ticker.upper(), str(start_date), str(end_date))
        if df_tcn.empty or len(df_tcn) < WINDOW_SIZE + 10:
            st.error("Data tidak cukup. Coba perluas rentang tanggal atau ganti kode saham.")
            st.stop()
        st.write(f"✅ Data berhasil dimuat: **{len(df_tcn):,}** baris")
    except Exception as ex:
        st.error(f"Gagal memuat data: {ex}")
        st.stop()

    # Split (Slovin)
    N = len(df_tcn)
    n_slovin   = N / (1 + N * (E_SLOVIN ** 2))
    train_size = math.ceil(n_slovin)
    test_size  = N - train_size
    df_train_raw = df_tcn.iloc[:train_size]
    df_test_raw  = df_tcn.iloc[train_size:]
    st.write(f"📂 Train: **{train_size:,}** | Test: **{test_size:,}**")

    # Fit scaler ulang dari data train baru agar menyesuaikan rentang tanggal input
    from sklearn.preprocessing import MinMaxScaler
    scaler_baru = MinMaxScaler()
    train_scaled = scaler_baru.fit_transform(df_train_raw)
    test_scaled  = scaler_baru.transform(df_test_raw)
    scaler = scaler_baru  # pakai scaler baru untuk inverse_transform

    # Windowing untuk data test
    X_te = test_scaled[:-HORIZON]
    y_te = test_scaled[HORIZON:]
    X_test, y_test = create_windows(X_te, y_te, WINDOW_SIZE)
    st.write(f"🪟 X_test: {X_test.shape}")

    # Prediksi dengan model yang sudah di-load
    status.update(label="🔮 Memprediksi...", state="running")
    y_pred        = model.predict(X_test, verbose=0)
    y_pred_actual = scaler.inverse_transform(y_pred)
    y_test_actual = scaler.inverse_transform(y_test)

    # Prediksi hari berikutnya
    last_window     = test_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    next_day_scaled = model.predict(last_window, verbose=0)
    next_day_price  = scaler.inverse_transform(next_day_scaled)[0][0]

    # Metrics
    mae  = mean_absolute_error(y_test_actual, y_pred_actual)
    mse  = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

    status.update(label="✅ Selesai!", state="complete")


# ── NEXT-DAY PREDICTION ───────────────────────────────────────────────────────
st.markdown("### 🎯 Prediksi Harga Besok")
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    last_close      = df_tcn["Close"].iloc[-1]
    last_date       = df_tcn.index[-1]  # tanggal data perdagangan terakhir
    next_trade_date = pd.bdate_range(start=last_date, periods=2)[1]  # hari bursa berikutnya
    delta_pct       = (next_day_price - float(last_close)) / float(last_close) * 100
    arrow           = "▲" if delta_pct >= 0 else "▼"
    arrow_color     = "#22c55e" if delta_pct >= 0 else "#ef4444"
    st.markdown(f"""
    <div class="pred-box">
        <div class="pred-label">Prediksi Harga Close</div>
        <div class="pred-value">Rp{next_day_price:,.0f}</div>
        <div style="font-size:1rem;color:{arrow_color};font-family:'Space Mono',monospace;font-weight:700;">
            {arrow} {abs(delta_pct):.2f}% dari hari terakhir
        </div>
        <div class="pred-ticker" style="margin-top:0.8rem;">
            {ticker.upper()} · Harga terakhir: Rp{float(last_close):,.0f}
        </div>
        <div style="margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid rgba(37,99,235,0.25);
                    font-size:0.75rem;color:#64748b;font-family:'Space Mono',monospace;">
            📅 Data terakhir: <strong style="color:#94a3b8;">{last_date.strftime('%d %B %Y')}</strong>
            &nbsp;·&nbsp;
            Prediksi untuk: <strong style="color:#60a5fa;">{next_trade_date.strftime('%d %B %Y')}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── METRICS ───────────────────────────────────────────────────────────────────
st.markdown("### 📊 Evaluasi Model")
m1, m2, m3 = st.columns(3)
for col, label, value, unit in zip(
    [m1, m2, m3],
    ["MAE", "RMSE", "MAPE"],
    [mae, rmse, mape],
    ["Rp", "Rp", "%"]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{unit}{value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── PREDICTION CHART ──────────────────────────────────────────────────────────
st.markdown("### 📉 Grafik Prediksi vs Aktual")

tanggal_test  = df_test_raw.index[WINDOW_SIZE:]
n_pts         = min(len(tanggal_test), len(y_test_actual.flatten()))
tanggal_plot  = tanggal_test[:n_pts]
y_actual_plot = y_test_actual.flatten()[:n_pts]
y_pred_plot   = y_pred_actual.flatten()[:n_pts]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=tanggal_plot, y=y_actual_plot,
    name="Aktual", line=dict(color="#60a5fa", width=1.5),
    hovertemplate="<b>Aktual</b><br>%{x|%d %b %Y}<br>Rp%{y:,.0f}<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=tanggal_plot, y=y_pred_plot,
    name="Prediksi", line=dict(color="#f97316", width=1.5, dash="dot"),
    hovertemplate="<b>Prediksi</b><br>%{x|%d %b %Y}<br>Rp%{y:,.0f}<extra></extra>"
))
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1220",
    font=dict(family="DM Sans", color="#94a3b8"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
    xaxis=dict(gridcolor="#1e2d4a", showgrid=True, tickformat="%b %Y"),
    yaxis=dict(gridcolor="#1e2d4a", showgrid=True, tickprefix="Rp"),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=20, b=10),
    height=420,
)
st.plotly_chart(fig, use_container_width=True)

# ── DATA SUMMARY ──────────────────────────────────────────────────────────────
with st.expander("🗂️ Ringkasan Data"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sample Data Pertama**")
        st.dataframe(df_tcn.head(10), use_container_width=True)
    with c2:
        st.markdown("**Sample Data Terakhir**")
        st.dataframe(df_tcn.tail(10), use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center;color:#334155;font-size:0.75rem;font-family:'Space Mono',monospace;">
    TCN Stock Predictor · Temporal Convolutional Network · Powered by keras-tcn & yfinance
</p>
""", unsafe_allow_html=True)
