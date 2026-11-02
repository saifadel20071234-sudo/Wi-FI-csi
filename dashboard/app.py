"""Live Wi-Fi CSI 3-Node Multi-Static Human Sensing & Thermal Radar Dashboard.

Real-time human presence detection and continuous thermal radar tracking off
a live Wi-Fi CSI link spanning 3 nodes:
1. Node 1: TX (ESP32 #1 - SSID: CSI_TX_NODE)
2. Node 2: EXT (CSI_NODE_EXT Access Point)
3. Node 3: RX (ESP32 #2 on Laptop COM port)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

from heatmap_engine import SmoothThermalTracker
from runtime import FingerprintLocalizer, LiveCSIState, parse_csi_serial_line


st.set_page_config(
    page_title="Wi-Fi CSI Biometric Radar Tracker",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("csi_model.pkl", "scaler.pkl", "feature_cols.pkl")
STATES = {
    0: "Empty Room (Clean)",
    1: "Person in Centre (Zone 1)",
    2: "Person on Right (Zone 2 - EXT)",
    3: "Person on Left (Zone 3 - TX)",
    4: "Person Moving (Active Stride)",
}


@st.cache_resource(show_spinner="Loading CSI model and calibration...")
def load_model_artifacts() -> dict:
    """Load internally consistent model bundle and its zone calibration."""
    for directory in (os.path.dirname(os.path.abspath(__file__)), os.path.join(ROOT_DIR, "ml")):
        paths = [os.path.join(directory, filename) for filename in FILES]
        if not all(os.path.exists(path) for path in paths):
            continue
        try:
            model, scaler, columns = (joblib.load(path) for path in paths)
            columns = list(columns)
            if model.n_features_in_ != len(columns) or scaler.n_features_in_ != len(columns):
                raise ValueError("Model, scaler, and feature-column counts do not match.")
            calibration_path = os.path.join(ROOT_DIR, "data", "csi_features.csv")
            localizer = None
            calibration_error = None
            if os.path.exists(calibration_path):
                localizer = FingerprintLocalizer.from_dataframe(pd.read_csv(calibration_path), scaler, columns)
            else:
                calibration_error = "Calibration CSV is missing; only fixed zone anchors are available."
            return {
                "ok": True,
                "model": model,
                "scaler": scaler,
                "cols": columns,
                "localizer": localizer,
                "calibration_error": calibration_error,
            }
        except Exception as exc:
            return {"ok": False, "error": f"Could not load model bundle: {exc}"}
    return {"ok": False, "error": "Model files are missing. Run ml/ml_model.py first."}


@dataclass
class SerialManager:
    """A process-level singleton serial connection shared safely across Streamlit sessions."""

    connection: object | None = None
    port: str | None = None
    baud_rate: int | None = None
    _buffer: str = ""

    def ensure(self, port: str, baud_rate: int):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install requirements.txt first.")
        if (
            self.connection is not None
            and getattr(self.connection, "is_open", False)
            and self.port == port
            and self.baud_rate == baud_rate
        ):
            try:
                _ = self.connection.in_waiting
                return self.connection
            except Exception:
                self.close()
        self.close()
        last_exc = None
        for attempt in range(3):
            try:
                connection = serial.Serial(port=None, baudrate=baud_rate, timeout=0.05)
                connection.dtr = False
                connection.rts = False
                connection.port = port
                connection.open()
                time.sleep(0.12)
                connection.reset_input_buffer()
                self.connection, self.port, self.baud_rate = connection, port, baud_rate
                self._buffer = ""
                return connection
            except Exception as exc:
                last_exc = exc
                self.close()
                time.sleep(0.3)
        raise last_exc

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
        self.connection = self.port = self.baud_rate = None
        self._buffer = ""

    def read_available(self, maximum_frames: int = 40) -> list[tuple[int, int, list[float]]]:
        if self.connection is None or not getattr(self.connection, "is_open", False):
            return []
        frames = []
        try:
            bytes_to_read = self.connection.in_waiting
            if bytes_to_read > 0:
                chunk = self.connection.read(bytes_to_read).decode("utf-8", errors="ignore")
                self._buffer += chunk
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    frame = parse_csi_serial_line(line)
                    if frame is not None:
                        frames.append(frame)
                        if len(frames) >= maximum_frames:
                            break
                if len(self._buffer) > 4096:
                    self._buffer = self._buffer[-1024:]
        except Exception:
            self.close()
        return frames


@st.cache_resource
def get_shared_serial_manager() -> SerialManager:
    return SerialManager()


def initialize_session() -> None:
    # Auto-start live tracking immediately on page load!
    st.session_state.setdefault("live_running", True)
    st.session_state.setdefault("tracker", SmoothThermalTracker(room_width=5.0, room_height=5.0, resolution=80))
    st.session_state.setdefault("live_state", LiveCSIState())
    st.session_state["serial_manager"] = get_shared_serial_manager()


def start_tracking() -> None:
    st.session_state.live_running = True
    st.session_state.live_state.reset()
    st.session_state.tracker = SmoothThermalTracker(room_width=5.0, room_height=5.0, resolution=80)


def stop_tracking() -> None:
    st.session_state.live_running = False
    get_shared_serial_manager().close()


def available_ports() -> list[str]:
    return [] if list_ports is None else [port.device for port in list_ports.comports()]


initialize_session()
artifacts = load_model_artifacts()

# App Header
st.title("📡 Wi-Fi CSI 3-Node Biometric Thermal Radar Tracker")
st.caption("Multi-Static Human Sensing with ESP32 #1 (TX) + CSI_NODE_EXT (Extender) + ESP32 #2 (RX)")

with st.sidebar:
    st.header("⚡ Hardware Link Setup")
    ports = available_ports()
    configured_port = os.environ.get("CSI_DEFAULT_PORT", "COM8")
    if ports:
        default_index = ports.index(configured_port) if configured_port in ports else 0
        selected_port = st.selectbox("Receiver COM port", ports, index=default_index, key="receiver_port")
    else:
        st.warning("No serial ports found. Connect the receiver or enter its port manually.")
        selected_port = st.text_input("Receiver COM port", value=configured_port, key="receiver_port")
    baud_rate = st.selectbox("Baud rate", (115200, 460800, 921600), index=0, key="receiver_baud")
    
    st.divider()
    st.markdown("### 🌐 Active 3-Node Topology")
    st.markdown(
        """
        - 📡 **Node 1: TX (ESP32 #1)**
          * Channel 6 SoftAP (`CSI_TX_NODE`)
        - 📶 **Node 2: EXT (`CSI_NODE_EXT`)**
          * Range Extender AP
        - 📥 **Node 3: RX (ESP32 #2)**
          * Dual-AP 20Hz Unicast Ping Client
        """
    )
    st.divider()
    if artifacts["ok"]:
        st.success(f"AI Model Ready ({len(artifacts['cols'])} Features)")
        if artifacts["calibration_error"]:
            st.warning(artifacts["calibration_error"])
        else:
            st.info("Calibrated Spatial Centroids: Left / Centre / Right")
    else:
        st.error(artifacts["error"])

with st.container(horizontal=True):
    if st.session_state.live_running:
        st.button("⏸ Pause Live Tracking", type="secondary", on_click=stop_tracking)
    else:
        st.button("▶ Resume Live Tracking", type="primary", on_click=start_tracking, disabled=not artifacts["ok"])
    st.button("🔄 Reset Canvas / Recalibrate", on_click=start_tracking)


@st.fragment(run_every=0.20 if st.session_state.live_running else None)
def live_panel() -> None:
    """Poll frames and render responsive live radar frame."""
    state: LiveCSIState = st.session_state.live_state
    tracker: SmoothThermalTracker = st.session_state.tracker
    connection_error = None
    
    if st.session_state.live_running:
        try:
            manager = get_shared_serial_manager()
            manager.ensure(selected_port.strip(), int(baud_rate))
            for frame in manager.read_available():
                state.ingest(frame, artifacts, artifacts["localizer"])
        except Exception as exc:
            connection_error = str(exc)

    label = state.label if state.packet_count else 0
    figure = tracker.render_thermal_frame(label, float(state.latest_rssi or -55), state.estimated_position)
    
    radar_col, telemetry_col = st.columns((2.4, 1.0))
    
    with radar_col:
        st.subheader("🖥 Real-Time FLIR Thermal Spatial Radar (5.0m × 5.0m)")
        st.pyplot(figure)
        plt.close(figure)
        
        if state.estimated_position is not None:
            x, y = state.estimated_position
            st.info(
                f"🎯 **Biometric Target Detected:** Position: `({x:.2f}m, {y:.2f}m)` | "
                f"Zone Confidence: `{state.zone_confidence:.0f}%` | Motion Energy: `{state.motion_energy:.2f}`"
            )
        else:
            st.caption("Room is currently empty / ambient RF baseline.")

    with telemetry_col:
        st.subheader("📊 Live Telemetry HUD")
        if connection_error:
            st.error(f"Receiver Error: {connection_error}")
        elif not st.session_state.live_running:
            st.warning("Tracking Paused.")
        elif state.packet_count == 0:
            st.warning("Listening for CSI frames from ESP32...")
        else:
            st.success("🟢 3-Node Link Synchronized (20 Hz)")

        st.metric("Detected Biometric State", STATES.get(label, "Unknown"), border=True)
        st.metric("AI Model Confidence", f"{state.confidence:.1f}%", border=True)
        st.metric("Signal RSSI", "--" if state.latest_rssi is None else f"{state.latest_rssi} dBm", border=True)
        st.metric("Streaming Speed", f"{tracker.speed:.2f} m/s", border=True)
        st.metric("Packets Processed", state.packet_count, border=True)
        
        if state.packet_count:
            st.caption("64-Subcarrier CSI Amplitude Spectrum")
            st.bar_chart(pd.DataFrame({"amplitude": state.latest_amplitudes}), y="amplitude", height=140)
            
        if state.zone_probabilities:
            st.caption("3-Zone Spatial Probability Distribution")
            st.bar_chart(
                pd.DataFrame({
                    "Zone": ["Center", "Right (EXT)", "Left (TX)"],
                    "Probability": [
                        state.zone_probabilities.get(1, 0.0),
                        state.zone_probabilities.get(2, 0.0),
                        state.zone_probabilities.get(3, 0.0),
                    ],
                }),
                x="Zone",
                y="Probability",
                height=140,
            )


live_panel()

with st.expander("ℹ️ How the 3-Node Multi-Static Wi-Fi CSI System Works"):
    st.markdown(
        """
        - **Node 1 (ESP32 #1 - TX):** Transmits high-frequency 802.11n beacons on Channel 6.
        - **Node 2 (CSI_NODE_EXT AP):** Extends RF coverage and creates a secondary multipath reflection boundary.
        - **Node 3 (ESP32 #2 - RX):** Captures 64-subcarrier raw Channel State Information at 20 packets/sec via active ICMP ping echoes.
        - **Biometric Heatmap Engine:** Fuses subcarrier variance across low, mid, and high bands with calibrated spatial centroids to continuously track the human's walking path and render an infrared FLIR thermal radiation silhouette.
        """
    )
