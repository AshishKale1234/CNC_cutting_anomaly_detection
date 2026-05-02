# ── app.py ───────────────────────────────────────────────────────────────────
# CNC Cutting Anomaly Detection — Final Batch App

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import tempfile
import os
from datetime import timedelta

from config import (
    ANOMALY_THRESHOLD, ZONE_STABLE, ZONE_ABNORMAL,
    TFLITE_MODEL_PATH, SAMP_RATE, N_FFT, HOP_LENGTH, N_MELS
)
from model import ChatterModel  # legacy class name retained internally

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cutting Anomaly Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .stButton > button {
        background-color: #238636; color: white;
        border: none; border-radius: 6px;
        padding: 0.6rem 1.8rem; font-size: 1rem; font-weight: 600;
    }
    .stButton > button:hover { background-color: #2ea043; }
    .stButton > button:disabled { background-color: #30363d; color: #8b949e; }
    [data-testid="stFileUploader"] {
        background: #161b22; border: 2px dashed #30363d;
        border-radius: 10px; padding: 1rem;
    }
    [data-testid="stFileUploader"]:hover { border-color: #2ea8a0; }
    [data-testid="stMetricValue"] { color: #2ea8a0 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return ChatterModel(TFLITE_MODEL_PATH)


def get_status(index: float):
    if index < ZONE_STABLE:
        return "STABLE", "#3fb950", "✅"
    elif index < ZONE_ABNORMAL:
        return "ABNORMAL", "#f7a325", "⚠️"
    else:
        return "CRITICAL", "#ff4444", "❌"


def build_gauge(anomaly_index: float):
    status, color, emoji = get_status(anomaly_index)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=anomaly_index,
        title={'text': f"Cutting Anomaly Index  {emoji}  {status}",
               'font': {'color': '#e6edf3', 'size': 20}},
        number={'suffix': '%', 'font': {'color': color, 'size': 52}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#8b949e', 'tickfont': {'color': '#8b949e'}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#161b22',
            'borderwidth': 0,
            'steps': [
                {'range': [0, ZONE_STABLE],             'color': 'rgba(63,185,80,0.25)'},
                {'range': [ZONE_STABLE, ZONE_ABNORMAL], 'color': 'rgba(247,163,37,0.25)'},
                {'range': [ZONE_ABNORMAL, 100],         'color': 'rgba(255,68,68,0.25)'},
            ],
            'threshold': {'line': {'color': color, 'width': 5}, 'thickness': 0.9, 'value': anomaly_index}
        }
    ))
    fig.update_layout(
        height=320, paper_bgcolor='#0d1117', font=dict(color='#e6edf3'),
        margin=dict(t=60, b=10, l=30, r=30)
    )
    return fig


def build_amplitude_overlay(audio, sr, probs, anomaly_flags, threshold):
    """Top: raw amplitude envelope; overlay anomaly regions."""
    window = sr  # 1 second
    n = len(audio) // window
    rms = np.array([
        np.sqrt(np.mean(audio[i*window:(i+1)*window]**2))
        for i in range(n)
    ])
    x = list(range(n))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Raw Amplitude (RMS) with Anomaly Regions Overlaid", "Anomaly Probability Over Time"],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15
    )

    fig.add_trace(go.Scatter(
        x=x, y=rms.tolist(), mode='lines',
        name='RMS Amplitude', line=dict(color='#f7a325', width=1.5)
    ), row=1, col=1)

    amp_max = float(rms.max()) if len(rms) > 0 else 1.0
    anomaly_x, anomaly_y = [], []
    for i, flag in enumerate(anomaly_flags[:n]):
        if flag:
            anomaly_x += [i, i+1, i+1, i, None]
            anomaly_y += [0, 0, amp_max, amp_max, None]
    if anomaly_x:
        fig.add_trace(go.Scatter(
            x=anomaly_x, y=anomaly_y,
            fill='toself', fillcolor='rgba(255,68,68,0.3)',
            line=dict(width=0), name='Anomaly Flagged',
            hoverinfo='skip'
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(range(len(probs))), y=probs.tolist(),
        mode='lines', name='Anomaly Probability',
        line=dict(color='#2ea8a0', width=1.8)
    ), row=2, col=1)

    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#ff4444", line_width=1.5,
        annotation_text=f"Threshold = {threshold:.2f}",
        annotation_font_color="#ff4444",
        row=2, col=1
    )

    fig.update_layout(
        height=600, paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
        margin=dict(t=60, b=40, l=30, r=30)
    )
    fig.update_xaxes(gridcolor='#30363d', zerolinecolor='#30363d',
                     title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(gridcolor='#30363d', zerolinecolor='#30363d',
                     title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(gridcolor='#30363d', zerolinecolor='#30363d',
                     range=[0, 1], title_text="Probability", row=2, col=1)
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(color='#e6edf3', size=13)
    return fig


def build_spectrogram(audio, sr):
    """Mel spectrogram of the full audio — downsampled for display."""
    M = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=N_FFT, window='hann', n_mels=N_MELS
    )
    S_db = librosa.power_to_db(2 * abs(M) / N_FFT, ref=1)

    # Downsample time axis for display if too large
    n_frames = S_db.shape[1]
    MAX_FRAMES = 4000   # keeps payload under Streamlit's 200MB limit
    if n_frames > MAX_FRAMES:
        step = n_frames // MAX_FRAMES
        S_db = S_db[:, ::step]
        n_frames = S_db.shape[1]

    time_axis = np.linspace(0, len(audio) / sr, n_frames)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=sr/2)

    fig = go.Figure(data=go.Heatmap(
        z=S_db,
        x=time_axis,
        y=mel_freqs,
        colorscale='Inferno',
        zmin=-80, zmax=0,
        colorbar=dict(title='dB', tickfont=dict(color='#8b949e'))
    ))
    fig.update_layout(
        height=400,
        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        font=dict(color='#e6edf3'),
        title=dict(text='Mel Spectrogram — What the CNN Analyses',
                   font=dict(color='#e6edf3', size=15)),
        xaxis=dict(title='Time (seconds)', gridcolor='#30363d'),
        yaxis=dict(title='Frequency (Hz)', gridcolor='#30363d'),
        margin=dict(t=60, b=40, l=30, r=30)
    )
    return fig

def extract_anomaly_events(anomaly_flags):
    events, in_evt, evt_start = [], False, 0
    for i, flag in enumerate(anomaly_flags):
        if flag and not in_evt:
            evt_start = i; in_evt = True
        elif not flag and in_evt:
            events.append((evt_start, i - 1)); in_evt = False
    if in_evt:
        events.append((evt_start, len(anomaly_flags) - 1))
    return events


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚙️ CNC Cutting Anomaly Detection Dashboard")
st.markdown(
    "Upload a machining audio recording to detect acoustic anomalies and compute a "
    "**Cutting Anomaly Index (CAI)**. The system flags deviations from stable cutting — "
    "which may correspond to chatter, adverse cutting parameters, tool wear, or built-up edge."
)
st.markdown("---")

# ── File uploader ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload .wav file from Maijker",
    type=['wav'],
    help="Download the audio file from Maijker for your machining session, then upload here."
)

if uploaded_file is None:
    st.info("⬆️  Upload a `.wav` file to begin analysis.")
    st.stop()

file_size_mb = uploaded_file.size / (1024 * 1024)
st.success(f"📁 **{uploaded_file.name}** — {file_size_mb:.2f} MB")

# ── Threshold slider + Analyse button ────────────────────────────────────────
col_t1, col_t2, col_t3 = st.columns([1, 2, 1])
with col_t1:
    threshold = st.slider(
        "Threshold",
        min_value=0.1, max_value=0.95, value=float(ANOMALY_THRESHOLD), step=0.05,
        help="Lower = more sensitive. Higher = more conservative."
    )
with col_t3:
    st.markdown("")
    st.markdown("")
    analyse_clicked = st.button("🚀 Analyse", use_container_width=True)

if not analyse_clicked:
    st.stop()

# ── Save to temp and load ────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
    tmp.write(uploaded_file.getbuffer())
    tmp_path = tmp.name

status_box = st.empty()
progress   = st.progress(0)

try:
    status_box.info("Loading audio...")
    progress.progress(15)
    audio, sr = librosa.load(tmp_path, sr=SAMP_RATE)
    duration = len(audio) / sr

    status_box.info(f"Loaded {duration:.1f}s of audio. Loading model...")
    progress.progress(30)
    model = load_model()

    status_box.info("Running CNN inference on Mel spectrograms...")
    progress.progress(50)
    probs, _, _, n_windows = model.predict_audio(audio)

    anomaly_flags = probs >= threshold
    anomaly_index = float(np.mean(anomaly_flags) * 100)
    progress.progress(95)

    status_box.success("✅ Analysis complete")
    progress.progress(100)
finally:
    try:
        os.remove(tmp_path)
    except Exception:
        pass

# ── Metrics row ──────────────────────────────────────────────────────────────
st.markdown("")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Duration",              f"{duration:.0f} s")
m2.metric("Windows Analysed",      f"{n_windows}")
m3.metric("Anomaly Windows",       f"{int(np.sum(anomaly_flags))}")
m4.metric("Cutting Anomaly Index", f"{anomaly_index:.1f}%")

# ── Gauge ────────────────────────────────────────────────────────────────────
st.plotly_chart(build_gauge(anomaly_index), use_container_width=True)

# ── Status banner ────────────────────────────────────────────────────────────
status_label, color, emoji = get_status(anomaly_index)
st.markdown(
    f'<div style="background:{color}22; border:1px solid {color}; '
    f'border-radius:10px; padding:1.2rem; text-align:center; '
    f'font-size:1.4rem; font-weight:bold; color:{color}">'
    f'{emoji}  Machining Status: {status_label}</div>',
    unsafe_allow_html=True
)
st.markdown("")

# ── Anomaly events ───────────────────────────────────────────────────────────
events = extract_anomaly_events(anomaly_flags)
if events:
    st.markdown(f"### 🔴 Cutting Anomalies Detected — {len(events)} event(s)")
    st.caption(
        st.caption(
    "Persistent horizontal bright bands at specific frequencies indicate sustained acoustic anomalies. "
    "Broadband vertical bursts often correspond to transient high-energy cutting events."
)
    )
    event_cols = st.columns(min(3, len(events)))
    for j, (s, e) in enumerate(events):
        s_time = str(timedelta(seconds=s))
        e_time = str(timedelta(seconds=e))
        dur    = e - s + 1
        with event_cols[j % len(event_cols)]:
            st.markdown(
                f'<div style="background:#161b22; border:1px solid #30363d; '
                f'border-left:4px solid #ff4444; border-radius:6px; '
                f'padding:0.8rem; margin-bottom:0.5rem">'
                f'<b style="color:#ff4444">Event {j+1}</b><br>'
                f'<span style="color:#8b949e; font-size:0.9rem">'
                f'{s_time} → {e_time}<br>Duration: {dur}s</span>'
                f'</div>',
                unsafe_allow_html=True
            )
else:
    st.success("✅ No cutting anomalies detected — machining was stable throughout.")

st.markdown("")

# ── Amplitude + probability overlay ──────────────────────────────────────────
st.markdown("### 📊 Amplitude vs Model Predictions")
st.caption(
    "Red shaded regions = anomalies flagged by the model. "
    "Amplitude spikes that aren't shaded are loud events the CNN determined were not anomalous."
)
st.plotly_chart(build_amplitude_overlay(audio, sr, probs, anomaly_flags, threshold),
                use_container_width=True)

# ── Spectrogram ──────────────────────────────────────────────────────────────
st.markdown("### 🎵 Frequency Analysis")
st.caption(
    "True chatter typically appears as persistent horizontal bright bands at specific frequencies. "
    "Broadband vertical bursts often indicate adverse cutting or normal high-energy events."
)
st.plotly_chart(build_spectrogram(audio, sr), use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Purdue ME 597 | MMRL | CNN on Mel Spectrograms | TFLite Inference")