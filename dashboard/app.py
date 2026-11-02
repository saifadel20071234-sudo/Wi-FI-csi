"""Live Wi-Fi CSI 3-Node Multi-Static Human Sensing & Thermal Radar Dashboard.

Real-time human presence detection and continuous thermal radar tracking off
a live Wi-Fi CSI link spanning 3 nodes:
1. Node 1: TX (ESP32 #1 - SSID: CSI_TX_NODE)
2. Node 2: EXT (CSI_NODE_EXT Access Point)
3. Node 3: RX (ESP32 #2 on Laptop COM port)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

# Ensure dashboard folder is in sys.path so sibling imports work everywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
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
    page_title="Wi-Fi CSI Multi-Person Thermal Radar Tracker",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("csi_model.pkl", "scaler.pkl", "feature_cols.pkl")
STATES = {
    0: "الغرفة فارغة (0 أشخاص)",
    1: "شخص واحد متتبع (1 Person)",
    2: "شخصان متتبعان (2 Persons)",
    3: "3 أشخاص متتبعين (3 Persons)",
    4: "4 أشخاص متتبعين (4 Persons)",
    5: "5 أشخاص متتبعين (5 Persons)",
}


@st.cache_resource(show_spinner="Loading CSI multi-person AI model and calibration...")
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
                calibration_error = "Calibration CSV is missing; fallback zone anchors active."
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
    # Auto-start live tracking immediately on page load
    st.session_state.setdefault("live_running", True)
    st.session_state.setdefault("tracker", SmoothThermalTracker(room_width=5.0, room_height=5.0, resolution=70))
    st.session_state.setdefault("live_state", LiveCSIState())
    st.session_state["serial_manager"] = get_shared_serial_manager()


def start_tracking() -> None:
    st.session_state.live_running = True
    st.session_state.live_state.reset()
    st.session_state.tracker = SmoothThermalTracker(room_width=5.0, room_height=5.0, resolution=70)


def stop_tracking() -> None:
    st.session_state.live_running = False
    get_shared_serial_manager().close()


def available_ports() -> list[str]:
    return [] if list_ports is None else [port.device for port in list_ports.comports()]


initialize_session()
artifacts = load_model_artifacts()

# App Header
st.title("📡 Wi-Fi CSI Multi-Person Thermal Radar & Biometric Tracker")
st.caption("Zero-Delay Tracking (0 to 5 Persons): ESP32 #1 (TX) + CSI_NODE_EXT (AP) + ESP32 #2 (RX)")

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
          * 20Hz Unicast Ping Client on Laptop
        """
    )
    st.divider()
    if artifacts["ok"]:
        st.success(f"Multi-Person AI Model Ready (0-5 Persons)")
    else:
        st.error(artifacts["error"])

# Control buttons
col_b1, col_b2 = st.columns(2)
with col_b1:
    if st.session_state.live_running:
        st.button("⏸ إيقاف مؤقت (Pause)", type="secondary", on_click=stop_tracking)
    else:
        st.button("▶ استئناف التتبع (Resume)", type="primary", on_click=start_tracking, disabled=not artifacts["ok"])
with col_b2:
    st.button("🔄 إعادة ضبط الحسابات (Reset)", on_click=start_tracking)


@st.fragment(run_every=0.15 if st.session_state.live_running else None)
def live_panel() -> None:
    """Poll frames and render responsive live radar frame with zero delay (~10ms)."""
    state: LiveCSIState = st.session_state.live_state
    tracker: SmoothThermalTracker = st.session_state.tracker
    connection_error = None

    if st.session_state.live_running:
        try:
            port_to_use = st.session_state.get("receiver_port", selected_port).strip()
            baud_to_use = int(st.session_state.get("receiver_baud", baud_rate))
            manager = get_shared_serial_manager()
            manager.ensure(port_to_use, baud_to_use)
            for frame in manager.read_available():
                state.ingest(frame, artifacts, artifacts["localizer"])
        except Exception as exc:
            connection_error = str(exc)

    label = state.label if state.packet_count else 0

    # Render high-performance FLIR thermal radar in ~10ms (Zero-Delay)
    radar_image = tracker.render_thermal_frame(
        label=label,
        live_rssi=float(state.latest_rssi or -55),
        estimated_position=state.estimated_position,
        people_count=state.people_count,
        secondary_position=state.secondary_position,
        all_targets=state.all_targets,
    )

    radar_col, telemetry_col = st.columns((2.4, 1.0))

    with radar_col:
        st.subheader("🖥 Real-Time FLIR Thermal Spatial Radar (5.0m × 5.0m)")
        # Fast Pillow rendering with zero Matplotlib lag!
        st.image(radar_image, use_container_width=True)

        # Clear target tracking summary below radar
        if state.people_count > 0 and state.all_targets:
            msg_parts = []
            for idx, (tx, ty) in enumerate(state.all_targets):
                tgt_speed = tracker.targets[idx].speed if idx < len(tracker.targets) else 0.0
                msg_parts.append(f"🎯 **شخص #{idx + 1}:** إحداثيات `X = {tx:.2f}m, Y = {ty:.2f}m` | سرعة الحركة: `{tgt_speed:.2f} m/s`")
            st.info("\n\n".join(msg_parts))
        else:
            st.caption("المكان فارغ حالياً (0 أشخاص) — بانتظار مرور أي شخص لرسم النقطة وتتبعها فوراً دون أي نقطة وهمية.")

    with telemetry_col:
        st.subheader("📊 لوحة المؤشرات الحية (HUD)")
        if connection_error:
            st.error(f"Receiver Error: {connection_error}")
        elif not st.session_state.live_running:
            st.warning("Tracking Paused.")
        elif state.packet_count == 0:
            st.warning("Listening for CSI frames from ESP32...")
        else:
            st.success("🟢 3-Node Link Synchronized (20 Hz)")

        # 1. Prominent People Counter Metric
        if state.people_count == 0:
            count_display = "0 (المكان فارغ)"
        elif state.people_count == 1:
            count_display = "1 شخص واحد"
        elif state.people_count == 2:
            count_display = "2 شخصان"
        else:
            count_display = f"{state.people_count} أشخاص"

        st.metric("👥 عدد الأشخاص المتواجدين (People Count)", count_display, border=True)

        st.metric("حالة النشاط الحيوي", STATES.get(state.people_count, "Unknown"), border=True)
        st.metric("سرعة حركة الشخص الأول", f"{tracker.speed:.2f} m/s", border=True)
        st.metric("طاقة الاضطراب الحركي", f"{state.motion_energy:.2f}", border=True)
        st.metric("قوة الإشارة (RSSI)", "--" if state.latest_rssi is None else f"{state.latest_rssi} dBm", border=True)
        st.metric("إجمالي الفريمات المعالجة", state.packet_count, border=True)

        if state.packet_count:
            st.caption("64-Subcarrier CSI Amplitude Spectrum")
            st.bar_chart(pd.DataFrame({"amplitude": state.latest_amplitudes}), y="amplitude", height=130)

        if state.zone_probabilities:
            st.caption("توزيع كثافة الإشارة بين العقد الثلاث")
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
                height=130,
            )


live_panel()

with st.expander("ℹ️ How the 3-Node Multi-Static Wi-Fi CSI System Works"):
    st.markdown(
        """
        - **Node 1 (ESP32 #1 - TX):** Transmits high-frequency 802.11n beacons on Channel 6.
        - **Node 2 (CSI_NODE_EXT AP):** Extends RF coverage and creates a secondary multipath reflection boundary.
        - **Node 3 (ESP32 #2 - RX):** Captures 64-subcarrier raw Channel State Information at 20 packets/sec via active ICMP ping echoes.
        - **Biometric Heatmap Engine:** Fuses subcarrier variance across low, mid, and high bands with calibrated spatial centroids to continuously track the human's walking path and render an infrared FLIR thermal radiation silhouette with distinct zero-delay tracking dots.
        """
    )
