"""
📈 Stock Price Prediction using LSTM / RNN / GRU
--------------------------------------------------
An animated, 3D-visualized Streamlit app for sequence-model based
stock price forecasting. Deployable for free on Streamlit Community Cloud.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout
from tensorflow.keras.callbacks import Callback

# ============================================================
# PAGE CONFIG + STYLE
# ============================================================
st.set_page_config(
    page_title="Stock Price Prediction | LSTM · RNN · GRU",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

/* ---------- animated aurora background ---------- */
.stApp {
    background: radial-gradient(circle at 15% 20%, #1b2a4a 0%, transparent 45%),
                radial-gradient(circle at 85% 10%, #3a1b4a 0%, transparent 40%),
                radial-gradient(circle at 50% 90%, #0f2f3a 0%, transparent 45%),
                #05060a;
    background-size: 200% 200%;
    animation: auroraShift 18s ease-in-out infinite;
}
@keyframes auroraShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ---------- hero banner ---------- */
.hero {
    padding: 34px 38px;
    border-radius: 22px;
    margin-bottom: 22px;
    background: linear-gradient(120deg, rgba(76,201,240,0.14), rgba(247,37,133,0.14) 50%, rgba(123,223,141,0.14));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
    animation: sheen 6s linear infinite;
}
@keyframes sheen { 0% { transform: translateX(-100%);} 100% { transform: translateX(100%);} }
.hero-title {
    font-size: 40px; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #4cc9f0, #f72585 55%, #7bdf8d);
    background-size: 300% auto;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: gradientMove 6s ease infinite;
}
@keyframes gradientMove { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.hero-sub { color: #b8c0d4; font-size: 15.5px; margin-top: 8px; }
.badge-row { margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }
.badge {
    font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
    padding: 6px 12px; border-radius: 999px;
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    color: #d8dee9;
}

/* ---------- headings ---------- */
h1, h2, h3 { color: #f5f5f5 !important; letter-spacing: -0.01em; }

/* ---------- glass metric cards ---------- */
.metric-card {
    background: rgba(255,255,255,0.045);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 20px 22px;
    border: 1px solid rgba(255,255,255,0.09);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}
.metric-card:hover {
    transform: translateY(-4px) scale(1.01);
    border-color: rgba(255,255,255,0.25);
    box-shadow: 0 12px 30px rgba(0,0,0,0.45);
}

/* ---------- buttons ---------- */
.stButton > button {
    background: linear-gradient(90deg, #4cc9f0, #f72585);
    background-size: 200% auto;
    color: white; border: none; font-weight: 700;
    border-radius: 12px; padding: 10px 18px;
    transition: background-position 0.5s ease, transform 0.15s ease;
}
.stButton > button:hover { background-position: right center; transform: translateY(-1px); }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-size: 15px; padding: 10px 18px; border-radius: 10px 10px 0 0;
    background: rgba(255,255,255,0.03);
}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0d14, #0e1220);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* ---------- dataframe / containers ---------- */
[data-testid="stDataFrame"], .stPlotlyChart {
    border-radius: 16px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">📈 Stock Price Prediction</div>
    <div class="hero-sub">LSTM · RNN · GRU sequence models trained per-stock on real NSE daily data (2016–2017) —
    visualized with animated 3D trajectories.</div>
    <div class="badge-row">
        <span class="badge">🔵 LSTM → Open Price</span>
        <span class="badge">🔴 RNN → Close Price</span>
        <span class="badge">🟢 GRU → Close Price</span>
        <span class="badge">⚡ Live per-epoch training</span>
    </div>
</div>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "Dataset", "stock_data.csv")

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        df = None
    return df

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload your own CSV (optional)", type=["csv"])

df_raw = pd.read_csv(uploaded) if uploaded is not None else load_data()

if df_raw is None:
    st.warning("⚠️ No bundled dataset found and no file uploaded. "
               "Please upload a CSV with columns: SYMBOL, TIMESTAMP, OPEN, CLOSE.")
    st.stop()

required_cols = {"SYMBOL", "TIMESTAMP", "OPEN", "CLOSE"}
missing = required_cols - set(df_raw.columns)
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    symbol_counts = df_raw.groupby("SYMBOL").size().sort_values(ascending=False)
    default_symbols = symbol_counts[symbol_counts >= 100].index.tolist()
    symbol = st.selectbox(
        "📌 Select Stock Symbol",
        options=default_symbols if default_symbols else symbol_counts.index.tolist(),
        index=0,
    )
    time_step = st.slider("Sequence length (days)", min_value=10, max_value=60, value=30, step=5)
    epochs = st.slider("Training epochs", min_value=5, max_value=50, value=15, step=5)
    train_btn = st.button("🚀 Train Models", type="primary", use_container_width=True)

    st.divider()
    with st.expander("ℹ️ How this works"):
        st.markdown("""
**LSTM (Long Short-Term Memory)** — remembers long-term patterns using gated
memory cells; used here to predict the **Open** price.

**RNN (Simple Recurrent Neural Network)** — passes a hidden state step to
step; simplest sequence model, prone to forgetting long patterns; predicts
**Close** price.

**GRU (Gated Recurrent Unit)** — a lighter, faster alternative to LSTM with
fewer gates; also predicts **Close** price for comparison.

Each model looks back over the last **N days** (sequence length) of scaled
prices to predict the next day's price.
        """)

# ============================================================
# DATA PREP FOR SELECTED SYMBOL
# ============================================================
stock_df = df_raw[df_raw["SYMBOL"] == symbol].copy()
stock_df["TIMESTAMP"] = pd.to_datetime(stock_df["TIMESTAMP"])
stock_df = stock_df.sort_values("TIMESTAMP").reset_index(drop=True)
stock_df = stock_df[["TIMESTAMP", "OPEN", "CLOSE"]].dropna()

st.subheader(f"📊 {symbol} — {len(stock_df)} trading days")
c1, c2 = st.columns([3, 1])
with c1:
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=stock_df["TIMESTAMP"], y=stock_df["OPEN"],
                                    name="Open", line=dict(color="#4cc9f0")))
    price_fig.add_trace(go.Scatter(x=stock_df["TIMESTAMP"], y=stock_df["CLOSE"],
                                    name="Close", line=dict(color="#f72585")))
    price_fig.update_layout(template="plotly_dark", height=320,
                             margin=dict(l=10, r=10, t=30, b=10),
                             legend=dict(orientation="h", y=1.1))
    st.plotly_chart(price_fig, use_container_width=True)
with c2:
    st.dataframe(stock_df.tail(8), height=280, use_container_width=True)

MIN_ROWS = time_step + 20
if len(stock_df) < MIN_ROWS:
    st.error(f"Not enough history for '{symbol}' with sequence length {time_step}. "
              f"Pick a symbol with more rows or reduce the sequence length.")
    st.stop()

# ============================================================
# SEQUENCE CREATION
# ============================================================
def create_sequences(data, step):
    X, y = [], []
    for i in range(step, len(data)):
        X.append(data[i - step:i])
        y.append(data[i])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(stock_df[["OPEN", "CLOSE"]])

X, y = create_sequences(scaled, time_step)
split = int(len(X) * 0.85)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

y_train_open, y_test_open = y_train[:, 0], y_test[:, 0]
y_train_close, y_test_close = y_train[:, 1], y_test[:, 1]

test_dates = stock_df["TIMESTAMP"].iloc[time_step + split:].reset_index(drop=True)

# ============================================================
# MODEL BUILDERS
# ============================================================
def build_model(cell):
    m = Sequential([
        cell(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        cell(64),
        Dropout(0.2),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

class ProgressCallback(Callback):
    def __init__(self, bar, label, total_epochs):
        self.bar = bar
        self.label = label
        self.total = total_epochs
    def on_epoch_end(self, epoch, logs=None):
        pct = (epoch + 1) / self.total
        self.bar.progress(pct, text=f"{self.label} — epoch {epoch+1}/{self.total} (loss={logs['loss']:.5f})")

@st.cache_resource(show_spinner=False)
def train_all(symbol, time_step, epochs, _X_train, _y_train_open, _y_train_close):
    results = {}
    for name, cell, target in [
        ("LSTM", LSTM, _y_train_open),
        ("RNN", SimpleRNN, _y_train_close),
        ("GRU", GRU, _y_train_close),
    ]:
        model = build_model(cell)
        history = model.fit(_X_train, target, epochs=epochs, batch_size=32, verbose=0)
        results[name] = {"model": model, "loss_curve": history.history["loss"]}
    return results

# ============================================================
# TRAIN + PREDICT
# ============================================================
if train_btn:
    st.session_state["trained"] = True
    st.session_state.pop("models_cache_key", None)

if st.session_state.get("trained"):
    cache_key = (symbol, time_step, epochs)
    if st.session_state.get("models_cache_key") != cache_key:
        prog_area = st.empty()
        bars = {name: prog_area.progress(0, text=f"{name} — waiting…") for name in ["LSTM", "RNN", "GRU"]}
        # Train sequentially with live progress bars
        trained = {}
        for name, cell, target in [
            ("LSTM", LSTM, y_train_open),
            ("RNN", SimpleRNN, y_train_close),
            ("GRU", GRU, y_train_close),
        ]:
            model = build_model(cell)
            bar = st.progress(0, text=f"{name} — starting…")
            model.fit(X_train, target, epochs=epochs, batch_size=32, verbose=0,
                      callbacks=[ProgressCallback(bar, name, epochs)])
            bar.empty()
            trained[name] = model
        prog_area.empty()

        st.session_state["models"] = trained
        st.session_state["models_cache_key"] = cache_key
        st.success("✅ Models trained successfully!")

    models = st.session_state["models"]

    pred_lstm = models["LSTM"].predict(X_test, verbose=0).reshape(-1)
    pred_rnn = models["RNN"].predict(X_test, verbose=0).reshape(-1)
    pred_gru = models["GRU"].predict(X_test, verbose=0).reshape(-1)

    lstm_rmse = np.sqrt(mean_squared_error(y_test_open, pred_lstm))
    rnn_rmse = np.sqrt(mean_squared_error(y_test_close, pred_rnn))
    gru_rmse = np.sqrt(mean_squared_error(y_test_close, pred_gru))

    # ---------------- Metrics ----------------
    st.subheader("📉 Model Performance (lower RMSE = better)")
    m1, m2, m3 = st.columns(3)
    for col, name, rmse, color in [
        (m1, "🔵 LSTM (Open)", lstm_rmse, "#4cc9f0"),
        (m2, "🔴 RNN (Close)", rnn_rmse, "#f72585"),
        (m3, "🟢 GRU (Close)", gru_rmse, "#7bdf8d"),
    ]:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div style="color:{color};font-size:14px;font-weight:600;">{name}</div>
                <div style="font-size:32px;font-weight:800;margin-top:4px;
                            background:linear-gradient(90deg,{color},#ffffff);
                            -webkit-background-clip:text;background-clip:text;color:transparent;">{rmse:.4f}</div>
                <div style="color:#8a93a6;font-size:12px;margin-top:2px;">RMSE (scaled) · lower is better</div>
                <div style="height:4px;border-radius:4px;margin-top:10px;background:rgba(255,255,255,0.08);overflow:hidden;">
                    <div style="height:100%;width:{min(100, rmse*4000):.0f}%;background:{color};border-radius:4px;"></div>
                </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ============================================================
    # ANIMATED 3D VISUALIZATION
    # ============================================================
    st.subheader("🌀 Animated 3D Prediction Trajectory")
    st.caption("Axes: Time → , Actual Close price ↑, Predicted Close price ↗. "
               "Press ▶ Play to watch predictions unfold day-by-day. "
               "A perfect model traces a diagonal-flat path (actual ≈ predicted).")

    n = len(pred_gru)
    steps = list(range(n))
    actual_close_series = y_test_close
    pred_close_series = pred_gru  # GRU close predictions for the 3D story

    frames = []
    frame_step = max(1, n // 60)  # cap ~60 frames for smooth playback
    frame_indices = list(range(frame_step, n + 1, frame_step))
    if frame_indices[-1] != n:
        frame_indices.append(n)

    for idx in frame_indices:
        frames.append(go.Frame(
            data=[
                go.Scatter3d(
                    x=steps[:idx], y=actual_close_series[:idx], z=pred_close_series[:idx],
                    mode="lines", line=dict(color="#4cc9f0", width=5), name="Trajectory"
                ),
                go.Scatter3d(
                    x=[steps[idx - 1]], y=[actual_close_series[idx - 1]], z=[pred_close_series[idx - 1]],
                    mode="markers", marker=dict(size=6, color="#f72585"), name="Current"
                ),
            ],
            name=str(idx)
        ))

    fig3d = go.Figure(
        data=[
            go.Scatter3d(x=[steps[0]], y=[actual_close_series[0]], z=[pred_close_series[0]],
                         mode="lines", line=dict(color="#4cc9f0", width=5), name="Trajectory"),
            go.Scatter3d(x=[steps[0]], y=[actual_close_series[0]], z=[pred_close_series[0]],
                         mode="markers", marker=dict(size=6, color="#f72585"), name="Current"),
        ],
        frames=frames
    )
    fig3d.update_layout(
        template="plotly_dark",
        height=560,
        scene=dict(
            xaxis_title="Time step (test days)",
            yaxis_title="Actual (scaled)",
            zaxis_title="Predicted (scaled)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1, x=0, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 60, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
            ]
        )]
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # ---------------- Static comparison line chart ----------------
    st.subheader("📈 Actual vs Predicted — All Models")
    tabs = st.tabs(["Open (LSTM)", "Close (RNN)", "Close (GRU)"])
    with tabs[0]:
        f = go.Figure()
        f.add_trace(go.Scatter(x=test_dates, y=y_test_open, name="Actual Open", line=dict(color="#4cc9f0")))
        f.add_trace(go.Scatter(x=test_dates, y=pred_lstm, name="Predicted Open (LSTM)", line=dict(color="#ffd166", dash="dot")))
        f.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(f, use_container_width=True)
    with tabs[1]:
        f = go.Figure()
        f.add_trace(go.Scatter(x=test_dates, y=y_test_close, name="Actual Close", line=dict(color="#f72585")))
        f.add_trace(go.Scatter(x=test_dates, y=pred_rnn, name="Predicted Close (RNN)", line=dict(color="#ffd166", dash="dot")))
        f.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(f, use_container_width=True)
    with tabs[2]:
        f = go.Figure()
        f.add_trace(go.Scatter(x=test_dates, y=y_test_close, name="Actual Close", line=dict(color="#7bdf8d")))
        f.add_trace(go.Scatter(x=test_dates, y=pred_gru, name="Predicted Close (GRU)", line=dict(color="#ffd166", dash="dot")))
        f.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(f, use_container_width=True)

    # ---------------- 3D bar comparison ----------------
    st.subheader("🏆 Model Comparison")
    bar_fig = go.Figure(data=[go.Bar(
        x=["LSTM (Open)", "RNN (Close)", "GRU (Close)"],
        y=[lstm_rmse, rnn_rmse, gru_rmse],
        marker_color=["#4cc9f0", "#f72585", "#7bdf8d"],
        text=[f"{v:.4f}" for v in [lstm_rmse, rnn_rmse, gru_rmse]],
        textposition="outside",
    )])
    bar_fig.update_layout(template="plotly_dark", height=360, yaxis_title="RMSE (lower is better)",
                           margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar_fig, use_container_width=True)

else:
    st.markdown("""
    <div class="metric-card" style="text-align:center;padding:40px 20px;">
        <div style="font-size:40px;">👈✨</div>
        <div style="font-size:18px;font-weight:700;margin-top:8px;">
            Pick a stock symbol in the sidebar and hit <span style="color:#f72585;">🚀 Train Models</span>
        </div>
        <div style="color:#8a93a6;margin-top:6px;">The animated 3D trajectory will appear right here.</div>
    </div>
    """, unsafe_allow_html=True)
