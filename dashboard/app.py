"""Live Wi-Fi CSI dashboard for a two-ESP32 sensing link.

The dashboard is explicit about its measurement limit: one CSI link provides
a calibrated Left/Centre/Right zone estimate, not camera-grade 2-D location.
The target moves only when incoming CSI data changes that estimate.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

from heatmap_engine import SmoothThermalTracker
from runtime import FingerprintLocalizer, LiveCSIState, parse_csi_serial_line


st.set_page_config(
    page_title="Wi-Fi CSI live tracker",
    page_icon=":material/radar:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("csi_model.pkl", "scaler.pkl", "feature_cols.pkl")
STATES = {0: "Empty room", 1: "Person in centre", 2: "Person on right", 3: "Person on left", 4: "Person moving"}


@st.cache_resource(show_spinner="Loading CSI model and calibration...")
def load_model_artifacts() -> dict:
    """Load one internally consistent model bundle and its zone calibration."""
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
            return {"ok": True, "model": model, "scaler": scaler, "cols": columns,
                    "localizer": localizer, "calibration_error": calibration_error}
        except Exception as exc:
            return {"ok": False, "error": f"Could not load model bundle: {exc}"}
    return {"ok": False, "error": "Model files are missing. Run ml/ml_model.py first."}


@dataclass
class SerialManager:
    """A process-level singleton serial connection shared safely across Streamlit sessions."""

    connection: object | None = None
    port: str | None = None
    baud_rate: int | None = None

    def ensure(self, port: str, baud_rate: int):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Install requirements.txt first.")
        if (self.connection is not None and getattr(self.connection, "is_open", False)
                and self.port == port and self.baud_rate == baud_rate):
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
                # Set control levels before opening: this reduces reset pulses on common ESP32 boards.
                connection.dtr = False
                connection.rts = False
                connection.port = port
                connection.open()
                time.sleep(0.15)
                connection.reset_input_buffer()
                self.connection, self.port, self.baud_rate = connection, port, baud_rate
                return connection
            except Exception as exc:
                last_exc = exc
                self.close()
                time.sleep(0.4)
        raise last_exc

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:
                pass
        self.connection = self.port = self.baud_rate = None

    def read_available(self, maximum_frames: int = 40) -> list[tuple[int, int, list[float]]]:
        if self.connection is None or not getattr(self.connection, "is_open", False):
            return []
        frames = []
        try:
            for _ in range(maximum_frames):
                if not self.connection.in_waiting:
                    break
                raw = self.connection.readline().decode("utf-8", errors="ignore")
                frame = parse_csi_serial_line(raw)
                if frame is not None:
                    frames.append(frame)
        except Exception:
            self.close()
        return frames


@st.cache_resource
def get_shared_serial_manager() -> SerialManager:
    return SerialManager()


def initialize_session() -> None:
    st.session_state.setdefault("live_running", False)
    st.session_state.setdefault("tracker", SmoothThermalTracker(room_width=5.0, room_height=5.0, resolution=75))
    st.session_state.setdefault("live_state", LiveCSIState())
    st.session_state["serial_manager"] = get_shared_serial_manager()


def start_tracking() -> None:
    st.session_state.live_running = True
    st.session_state.live_state.reset()
    st.session_state.tracker = SmoothThermalTracker(room_width=5.0, room_height=5.0, resolution=75)


def stop_tracking() -> None:
    st.session_state.live_running = False
    get_shared_serial_manager().close()


def available_ports() -> list[str]:
    return [] if list_ports is None else [port.device for port in list_ports.comports()]


initialize_session()
artifacts = load_model_artifacts()

st.title(":material/radar: Wi-Fi CSI live person tracker")
st.caption("The heatmap follows measured CSI changes; it does not generate a synthetic walking path.")

with st.sidebar:
    st.header("Receiver connection")
    ports = available_ports()
    configured_port = os.environ.get("CSI_DEFAULT_PORT", "COM9")
    if ports:
        default_index = ports.index(configured_port) if configured_port in ports else 0
        selected_port = st.selectbox("Receiver COM port", ports, index=default_index, key="receiver_port")
    else:
        st.warning("No serial ports found. Connect the receiver or enter its port manually.")
        selected_port = st.text_input("Receiver COM port", value=configured_port, key="receiver_port")
    baud_rate = st.selectbox("Baud rate", (115200, 460800, 921600), key="receiver_baud")
    st.divider()
    if artifacts["ok"]:
        st.success(f"Model loaded: {len(artifacts['cols'])} features")
        if artifacts["calibration_error"]:
            st.warning(artifacts["calibration_error"])
        else:
            st.info("CSI fingerprint calibration: Left / Centre / Right")
    else:
        st.error(artifacts["error"])
    st.divider()
    st.markdown("**Hardware setup**\n\n1. Power ESP32 transmitter.\n2. Connect ESP32 receiver by USB.\n3. Start tracking below.")

with st.container(horizontal=True):
    st.button("Start live tracking", icon=":material/play_arrow:", type="primary", on_click=start_tracking,
              disabled=not artifacts["ok"])
    st.button("Stop and disconnect", icon=":material/stop:", on_click=stop_tracking)


@st.fragment(run_every=0.25 if st.session_state.live_running else None)
def live_panel() -> None:
    """Poll a bounded number of frames, then render one responsive UI tick."""
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
    radar_column, telemetry_column = st.columns((2.2, 1.0))
    with radar_column:
        st.subheader("Live spatial heatmap")
        st.pyplot(figure)
        plt.close(figure)
        if state.estimated_position is not None:
            x, y = state.estimated_position
            st.caption(f"CSI zone estimate: x={x:.2f} m, y={y:.2f} m — zone confidence {state.zone_confidence:.0f}%")
        else:
            st.caption("No person position estimated yet.")
    with telemetry_column:
        st.subheader("Live RF telemetry")
        if connection_error:
            st.error(f"Receiver connection failed: {connection_error}")
        elif not st.session_state.live_running:
            st.info("Standby — select the receiver port and press Start live tracking.")
        elif state.packet_count == 0:
            st.warning("Listening for CSI frames. Confirm that the transmitter is powered and associated.")
        else:
            st.success("Receiving CSI frames")
        st.metric("Detected state", STATES.get(label, "Unknown"), border=True)
        st.metric("Model confidence", f"{state.confidence:.1f}%", border=True)
        st.metric("RSSI", "--" if state.latest_rssi is None else f"{state.latest_rssi} dBm", border=True)
        st.metric("CSI packets", state.packet_count, border=True)
        st.metric("Estimated speed", f"{tracker.speed:.2f} m/s", border=True)
        if state.packet_count:
            st.caption("64-subcarrier amplitude")
            st.bar_chart(pd.DataFrame({"amplitude": state.latest_amplitudes}), y="amplitude")
        if state.zone_probabilities:
            st.caption("Current calibrated zone evidence")
            st.bar_chart(pd.DataFrame({"zone": ["Centre", "Right", "Left"], "probability": [
                state.zone_probabilities.get(1, 0.0), state.zone_probabilities.get(2, 0.0),
                state.zone_probabilities.get(3, 0.0)]}), x="zone", y="probability")


live_panel()

with st.expander("How to interpret this map"):
    st.markdown(
        "With one transmitter-receiver CSI link, the system can show a calibrated **Left / Centre / Right** "
        "zone estimate and smooth transitions between them. It cannot uniquely measure exact two-dimensional "
        "coordinates from two ESP32 boards alone. For true 2-D localisation, add calibrated links around the room "
        "and train with known (x, y) reference points."
    )
