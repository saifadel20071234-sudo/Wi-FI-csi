"""
Wi-Fi CSI Human Tracking System - Live Hardware Dashboard
Pure Hardware Stream (No Simulation / No Demo)
Connects directly to ESP32 Receiver over Serial and tracks human thermal signature in a large dark room canvas.
"""

import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import joblib
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt
from scipy.fft import fft
from scipy.stats import skew, kurtosis, iqr
from sklearn.decomposition import PCA

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None

from heatmap_engine import SmoothThermalTracker

# --- Page Setup ---
st.set_page_config(
    page_title="Wi-Fi CSI Live Human Tracker",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Dark Style
st.markdown("""
<style>
    .stApp {
        background-color: #030407;
        color: #E2E8F0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    header, footer { visibility: hidden; }
    
    .hud-card {
        background: rgba(14, 23, 42, 0.75);
        border: 1px solid rgba(0, 180, 216, 0.25);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }
    .hud-title {
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .hud-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
WINDOW_SIZE = 30
NUM_SUB = 64
FILES = ["csi_model.pkl", "scaler.pkl", "feature_cols.pkl"]

STATES = {
    0: ("🏠 EMPTY ROOM", "#64748B", "rgba(100, 116, 139, 0.2)"),
    1: ("🧍 PERSON IN CENTER", "#00E5FF", "rgba(0, 229, 255, 0.2)"),
    2: ("🧍 PERSON ON RIGHT", "#F77F00", "rgba(247, 127, 0, 0.2)"),
    3: ("🧍 PERSON ON LEFT", "#38BDF8", "rgba(56, 189, 248, 0.2)"),
    4: ("🏃 PERSON MOVING", "#FF007F", "rgba(255, 0, 127, 0.2)"),
}

# --- Load Model Artifacts ---
@st.cache_resource(show_spinner="Loading Unified AI Model...")
def load_model_artifacts():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        [os.path.join(cur_dir, f) for f in FILES],
        [os.path.join(os.path.dirname(cur_dir), "ml", f) for f in FILES],
        FILES
    ]
    for paths in candidates:
        if all(os.path.exists(p) for p in paths):
            try:
                m, s, c = (joblib.load(p) for p in paths)
                return {"ok": True, "model": m, "scaler": s, "cols": list(c)}
            except Exception as e:
                return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "Model files not found. Run ml/ml_model.py first."}

artifacts = load_model_artifacts()

# --- DSP Feature Extraction (35 Features Aligned) ---
def lowpass_window(win, cutoff=0.15, order=4):
    try:
        cleaned = medfilt(win, kernel_size=(3, 1))
        b, a = butter(order, cutoff, btype="low")
        return filtfilt(b, a, cleaned, axis=0)
    except Exception:
        return win

def extract_live_features(win, rssi):
    mean_s, std_s = np.mean(win, axis=0), np.std(win, axis=0)
    flat = win.flatten()
    tvar = np.var(win, axis=0)
    
    diff1 = np.diff(win, axis=0)
    diff2 = np.diff(diff1, axis=0)
    
    sig_mean = np.mean(win, axis=1)
    fv = np.abs(fft(sig_mean))[: len(win) // 2]
    p = fv / (np.sum(fv) + 1e-6)
    freqs = np.linspace(0, 1, len(fv))
    spectral_centroid = float(np.sum(freqs * p))
    spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * p)))

    try:
        corr_m = np.corrcoef(win.T)
        corr_m = np.nan_to_num(corr_m, nan=0.0)
        triu = np.triu_indices(NUM_SUB, k=1)
        corr_mean = float(np.mean(corr_m[triu]))
        corr_std  = float(np.std(corr_m[triu]))
    except Exception:
        corr_mean, corr_std = 0.0, 0.0

    try:
        pca = PCA(n_components=3)
        pca.fit(win)
        evr = pca.explained_variance_ratio_
        pca_1, pca_2, pca_3 = float(evr[0]), float(evr[1]), float(evr[2])
    except Exception:
        pca_1, pca_2, pca_3 = 0.0, 0.0, 0.0

    max_v = float(np.max(win))
    min_v = float(np.min(win))
    rms_v = float(np.sqrt(np.mean(win ** 2)))
    low_b = float(np.mean(mean_s[:21]))
    mid_b = float(np.mean(mean_s[21:43]))
    high_b = float(np.mean(mean_s[43:]))

    f = {
        "mean_amp": float(np.mean(win)),
        "std_amp": float(np.std(win)),
        "max_amp": max_v,
        "min_amp": min_v,
        "range_amp": max_v - min_v,
        "variance_amp": float(np.var(win)),
        "rms_amp": rms_v,
        "crest_factor": float(max_v / (rms_v + 1e-6)),
        "energy_amp": float(np.sum(win ** 2)),
        "skewness": float(skew(flat)),
        "kurtosis": float(kurtosis(flat)),
        "iqr_amp": float(iqr(flat)),
        "low_band": low_b,
        "mid_band": mid_b,
        "high_band": high_b,
        "band_ratio_lh": float(low_b / (high_b + 1e-6)),
        "band_ratio_mh": float(mid_b / (high_b + 1e-6)),
        "subcarrier_corr_mean": corr_mean,
        "subcarrier_corr_std": corr_std,
        "temporal_var_mean": float(np.mean(tvar)),
        "temporal_var_max": float(np.max(tvar)),
        "velocity_mean": float(np.mean(np.abs(diff1))),
        "velocity_max": float(np.max(np.abs(diff1))),
        "accel_mean": float(np.mean(np.abs(diff2))),
        "fft_peak": float(np.max(fv[1:]) if len(fv) > 1 else fv[0]),
        "fft_energy": float(np.sum(fv ** 2)),
        "fft_entropy": float(-np.sum(p * np.log(p + 1e-10))),
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "pca_evr_1": pca_1,
        "pca_evr_2": pca_2,
        "pca_evr_3": pca_3,
        "rssi_mean": float(np.mean(rssi)),
        "rssi_std": float(np.std(rssi)),
        "rssi_range": float(np.max(rssi) - np.min(rssi)),
    }
    return f

def parse_csi_serial_line(line: str):
    line = line.strip()
    if not (line.startswith("CSI_START") and "CSI_END" in line):
        return None
    try:
        parts = line.replace(",CSI_END", "").split(",")
        num_sub = int(parts[3])
        if num_sub != NUM_SUB:
            return None
        amplitudes = [float(x) for x in parts[4:4 + num_sub]]
        return int(parts[1]), int(parts[2]), amplitudes
    except Exception:
        return None

# --- Session State ---
if "tracker" not in st.session_state:
    st.session_state.tracker = SmoothThermalTracker(room_width=5.0, room_height=5.0, resolution=75)
if "live_running" not in st.session_state:
    st.session_state.live_running = False

# --- Sidebar: Hardware Controls Only ---
with st.sidebar:
    st.title("📡 Hardware Control")
    st.markdown("**Live ESP32 Receiver Link**")
    st.divider()

    # Detect COM Ports
    available_ports = []
    if serial:
        available_ports = [p.device for p in list_ports.comports()]
    
    default_env_port = os.environ.get("CSI_DEFAULT_PORT", "")
    if not available_ports:
        st.warning("⚠️ No COM ports found! Connect ESP32 Receiver via USB.")
        selected_port = st.text_input("Manual Port Entry", value=default_env_port or "COM9")
    else:
        # COM9 is the Receiver board
        default_idx = 0
        if "COM9" in available_ports:
            default_idx = available_ports.index("COM9")
        elif default_env_port in available_ports:
            default_idx = available_ports.index(default_env_port)
        elif len(available_ports) > 1:
            default_idx = len(available_ports) - 1
            
        selected_port = st.selectbox("Select Receiver COM Port", available_ports, index=default_idx)
        if selected_port == "COM8":
            st.warning("⚠️ Note: COM8 is usually the Transmitter! Make sure you select the Receiver (COM9).")

    baud_rate = st.selectbox("Baud Rate", [115200, 460800, 921600], index=0)
    
    st.divider()
    if artifacts["ok"]:
        st.success("✅ AI Model Loaded (35 Features)")
    else:
        st.error(f"❌ Model Error: {artifacts['error']}")

    st.markdown("""
    **Hardware Setup Reminder:**
    - **ESP32 #1 (TX):** On USB power broadcasting `CSI_TX_NODE`.
    - **ESP32 #2 (RX):** Connected to this laptop via USB.
    """)

# --- Main Layout ---
col_radar, col_telemetry = st.columns([2.2, 1.0])

with col_radar:
    st.markdown("### 🗺️ Live Spatial Thermal Radar")
    radar_placeholder = st.empty()

with col_telemetry:
    st.markdown("### 📊 Live RF Telemetry")
    status_card = st.empty()
    rssi_card = st.empty()
    confidence_card = st.empty()
    packets_card = st.empty()
    st.divider()
    st.markdown("##### 📶 64 Subcarrier Spectrum")
    spectrum_placeholder = st.empty()

# --- Hardware Start / Stop Toggle ---
col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    start_btn = st.button("🚀 CONNECT & START LIVE TRACKING", type="primary", use_container_width=True)
with col_btn2:
    stop_btn = st.button("🛑 DISCONNECT & STOP", use_container_width=True)

if stop_btn:
    st.session_state.live_running = False

if start_btn:
    st.session_state.live_running = True

# --- Pure Hardware Live Tracking Loop ---
if st.session_state.live_running:
    if not artifacts["ok"]:
        st.error("Cannot start tracking: Model artifacts are missing!")
        st.stop()

    try:
        ser = serial.Serial(selected_port, baud_rate, timeout=0.1)
        ser.reset_input_buffer()
    except Exception as e:
        st.error(f"Failed to open port {selected_port}: {e}")
        st.session_state.live_running = False
        st.stop()

    window_buffer = deque(maxlen=WINDOW_SIZE)
    rssi_buffer = deque(maxlen=WINDOW_SIZE)
    packet_counter = 0
    current_label = 0
    current_conf = 0.0
    latest_rssi = -60
    latest_amps = [0.0] * NUM_SUB

    tracker = st.session_state.tracker
    
    # High-fluidity display rate limiter (12.5 FPS target for buttery UI)
    last_ui_render = 0.0
    last_spectrum_render = 0.0
    UI_INTERVAL = 0.075       # ~13.3 FPS
    SPECTRUM_INTERVAL = 0.18  # ~5.5 FPS

    try:
        while st.session_state.live_running:
            raw_line = ser.readline().decode("utf-8", errors="ignore")
            parsed = parse_csi_serial_line(raw_line)

            if parsed:
                ts, rssi_val, amps = parsed
                window_buffer.append(amps)
                rssi_buffer.append(rssi_val)
                packet_counter += 1
                latest_rssi = rssi_val
                latest_amps = amps

                # Run AI Model Inference once window has enough data
                if len(window_buffer) == WINDOW_SIZE:
                    win_arr = np.array(window_buffer)
                    rssi_arr = np.array(rssi_buffer)

                    # Filter & Extract 35 features
                    filtered_win = lowpass_window(win_arr)
                    feats = extract_live_features(filtered_win, rssi_arr)

                    # Dynamic column alignment
                    vec = [feats[c] for c in artifacts["cols"]]
                    X_scaled = artifacts["scaler"].transform([vec])
                    probs = artifacts["model"].predict_proba(X_scaled)[0]
                    idx = int(np.argmax(probs))
                    current_label = int(artifacts["model"].classes_[idx])
                    current_conf = float(probs[idx]) * 100

                now = time.time()
                # 1. Update Spatial Thermal Radar & Telemetry HUD (Rate-limited for max smoothness)
                if now - last_ui_render >= UI_INTERVAL:
                    last_ui_render = now
                    
                    fig = tracker.render_thermal_frame(current_label, latest_rssi)
                    radar_placeholder.pyplot(fig)
                    plt.close(fig)

                    name, text_color, bg_color = STATES.get(current_label, ("UNKNOWN", "#FFF", "rgba(0,0,0,0.5)"))
                    coord_str = f"({tracker.curr_x:.2f}m, {tracker.curr_y:.2f}m)" if tracker.curr_intensity > 0.1 else "N/A (Empty Room)"
                    speed_str = f"{tracker.speed:.2f} m/s" if tracker.curr_intensity > 0.1 else "0.00 m/s"

                    status_card.markdown(f"""
                    <div class="hud-card">
                        <div class="hud-title">Detected Activity / Zone</div>
                        <span class="status-badge" style="background:{bg_color}; color:{text_color}; border:1px solid {text_color};">
                            {name}
                        </span>
                        <div style="font-size:0.8rem; color:#94A3B8; margin-top:4px;">
                            Position: <b style="color:#FFF;">{coord_str}</b> &nbsp;|&nbsp; Velocity: <b style="color:#00E5FF;">{speed_str}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    rssi_card.markdown(f"""
                    <div class="hud-card">
                        <div class="hud-title">Live RSSI (Signal Strength)</div>
                        <div class="hud-value" style="color: #38BDF8;">{latest_rssi} <span style="font-size:0.9rem; color:#64748B;">dBm</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    confidence_card.markdown(f"""
                    <div class="hud-card">
                        <div class="hud-title">AI Detection Confidence</div>
                        <div class="hud-value" style="color: #4ADE80;">{current_conf:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    packets_card.markdown(f"""
                    <div class="hud-card">
                        <div class="hud-title">Raw CSI Packets Ingested</div>
                        <div class="hud-value" style="color: #A78BFA;">{packet_counter:,}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 2. Update Subcarrier Spectrum (Slightly slower rate to conserve frontend bandwidth)
                if now - last_spectrum_render >= SPECTRUM_INTERVAL:
                    last_spectrum_render = now
                    fig_sub, ax_sub = plt.subplots(figsize=(4, 1.8), facecolor="#0E172A")
                    ax_sub.set_facecolor("#0E172A")
                    ax_sub.bar(range(NUM_SUB), latest_amps, color="#00E5FF", width=0.85, alpha=0.9)
                    ax_sub.set_xlim(-1, 64)
                    ax_sub.set_ylim(0, max(45, max(latest_amps) + 5))
                    ax_sub.set_xlabel("OFDM Subcarrier Index (0-63)", color="#64748B", fontsize=7)
                    ax_sub.set_ylabel("Amplitude", color="#64748B", fontsize=7)
                    ax_sub.tick_params(colors="#64748B", labelsize=6)
                    for spine in ax_sub.spines.values():
                        spine.set_color("#1E293B")
                    spectrum_placeholder.pyplot(fig_sub)
                    plt.close(fig_sub)

    except Exception as e:
        st.error(f"Hardware Streaming Interrupted: {e}")
    finally:
        ser.close()

else:
    # Idle State: Render initial dark room canvas
    fig = st.session_state.tracker.render_thermal_frame(0)
    radar_placeholder.pyplot(fig)
    plt.close(fig)

    status_card.markdown("""
    <div class="hud-card">
        <div class="hud-title">Target State</div>
        <div class="hud-value" style="color: #64748B;">STANDBY (AWAITING HARDWARE)</div>
    </div>
    """, unsafe_allow_html=True)

    rssi_card.markdown("""
    <div class="hud-card">
        <div class="hud-title">Live RSSI</div>
        <div class="hud-value" style="color: #64748B;">-- dBm</div>
    </div>
    """, unsafe_allow_html=True)
