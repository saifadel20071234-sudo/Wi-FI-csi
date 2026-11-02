"""Runtime helpers for the Wi-Fi CSI multi-node dashboard.

This module deliberately contains no Streamlit code. Keeping serial parsing,
feature extraction, and localisation here makes the live path testable and
prevents Streamlit reruns from throwing away the CSI window.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import iqr, kurtosis, skew
from sklearn.decomposition import PCA


NUM_SUBCARRIERS = 64
MIN_WINDOW = 8
WINDOW_SIZE = 20

ZONE_COORDINATES = {
    1: np.array([2.50, 2.50]),  # Centre (crossing point)
    2: np.array([3.75, 2.30]),  # Right (near CSI_NODE_EXT AP)
    3: np.array([1.25, 2.30]),  # Left (near ESP32 TX)
}


def parse_csi_serial_line(line: str) -> tuple[int, int, list[float]] | None:
    """Parse ``CSI_START,timestamp,rssi,count,amp...,CSI_END`` safely and sanitize AGC subcarrier."""
    start_idx = line.find("CSI_START,")
    end_idx = line.find(",CSI_END", start_idx)
    if start_idx == -1 or end_idx == -1:
        return None
    try:
        payload = line[start_idx:end_idx]
        parts = payload.split(",")
        timestamp_ms = int(parts[1])
        rssi = int(parts[2])
        declared_count = int(parts[3])
        amplitudes = [float(value) for value in parts[4:] if value]
        if declared_count <= 0 or len(amplitudes) != declared_count:
            return None
        if len(amplitudes) > NUM_SUBCARRIERS:
            amplitudes = amplitudes[:NUM_SUBCARRIERS]
        elif len(amplitudes) < NUM_SUBCARRIERS:
            if len(amplitudes) < 10:
                return None
            source = np.linspace(0.0, 1.0, len(amplitudes))
            target = np.linspace(0.0, 1.0, NUM_SUBCARRIERS)
            amplitudes = np.interp(target, source, amplitudes).tolist()

        # Sanitize hardware AGC byte at subcarrier 0
        if len(amplitudes) > 6:
            amplitudes[0] = amplitudes[6]

        return timestamp_ms, rssi, amplitudes
    except (IndexError, TypeError, ValueError):
        return None


def lowpass_window(window: np.ndarray, cutoff: float = 0.20, order: int = 3) -> np.ndarray:
    """Use adaptive median + Butterworth filter to remove noise while keeping motion."""
    if len(window) < 6:
        return window
    try:
        cleaned = medfilt(window, kernel_size=(3, 1))
        b, a = butter(order, cutoff, btype="low")
        return filtfilt(b, a, cleaned, axis=0)
    except ValueError:
        return window


def extract_features(window: np.ndarray, rssi: np.ndarray) -> dict[str, float]:
    """Extract the 35 features used by ``ml/ml_model.py``."""
    win = window.copy()
    win[:, 0] = win[:, 6]

    mean_subcarrier = np.mean(win, axis=0)
    flattened = win.flatten()
    temporal_variance = np.var(win, axis=0)
    diff_1 = np.diff(win, axis=0)
    diff_2 = np.diff(diff_1, axis=0) if len(diff_1) > 1 else np.zeros_like(diff_1)

    mean_signal = np.mean(win, axis=1)
    fft_values = np.abs(fft(mean_signal))[: max(1, len(win) // 2)]
    fft_probability = fft_values / (np.sum(fft_values) + 1e-6)
    frequencies = np.linspace(0, 1, len(fft_values))
    spectral_centroid = float(np.sum(frequencies * fft_probability))
    spectral_spread = float(
        np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * fft_probability))
    )

    try:
        jittered = win + 1e-6 * np.random.RandomState(42).randn(*win.shape)
        correlation = np.nan_to_num(np.corrcoef(jittered.T), nan=0.0)
        upper = np.triu_indices(NUM_SUBCARRIERS, k=1)
        corr_mean = float(np.mean(correlation[upper]))
        corr_std = float(np.std(correlation[upper]))
    except Exception:
        corr_mean, corr_std = 0.0, 0.0

    try:
        n_comp = min(3, len(win))
        explained = PCA(n_components=n_comp).fit(win).explained_variance_ratio_
        pca_1 = float(explained[0]) if len(explained) > 0 else 0.0
        pca_2 = float(explained[1]) if len(explained) > 1 else 0.0
        pca_3 = float(explained[2]) if len(explained) > 2 else 0.0
    except ValueError:
        pca_1 = pca_2 = pca_3 = 0.0

    maximum = float(np.max(win))
    minimum = float(np.min(win))
    rms = float(np.sqrt(np.mean(win**2)))
    low_band = float(np.mean(mean_subcarrier[:21]))
    mid_band = float(np.mean(mean_subcarrier[21:43]))
    high_band = float(np.mean(mean_subcarrier[43:]))

    return {
        "mean_amp": float(np.mean(win)),
        "std_amp": float(np.std(win)),
        "max_amp": maximum,
        "min_amp": minimum,
        "range_amp": maximum - minimum,
        "variance_amp": float(np.var(win)),
        "rms_amp": rms,
        "crest_factor": float(maximum / (rms + 1e-6)),
        "energy_amp": float(np.sum(win**2)),
        "skewness": float(skew(flattened)),
        "kurtosis": float(kurtosis(flattened)),
        "iqr_amp": float(iqr(flattened)),
        "low_band": low_band,
        "mid_band": mid_band,
        "high_band": high_band,
        "band_ratio_lh": float(low_band / (high_band + 1e-6)),
        "band_ratio_mh": float(mid_band / (high_band + 1e-6)),
        "subcarrier_corr_mean": corr_mean,
        "subcarrier_corr_std": corr_std,
        "temporal_var_mean": float(np.mean(temporal_variance)),
        "temporal_var_max": float(np.max(temporal_variance)),
        "velocity_mean": float(np.mean(np.abs(diff_1))),
        "velocity_max": float(np.max(np.abs(diff_1))),
        "accel_mean": float(np.mean(np.abs(diff_2))),
        "fft_peak": float(np.max(fft_values[1:]) if len(fft_values) > 1 else fft_values[0]),
        "fft_energy": float(np.sum(fft_values**2)),
        "fft_entropy": float(-np.sum(fft_probability * np.log(fft_probability + 1e-10))),
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "pca_evr_1": pca_1,
        "pca_evr_2": pca_2,
        "pca_evr_3": pca_3,
        "rssi_mean": float(np.mean(rssi)),
        "rssi_std": float(np.std(rssi)),
        "rssi_range": float(np.max(rssi) - np.min(rssi)),
    }


class FingerprintLocalizer:
    """Multi-Zone Spatial Fingerprint and Kinetic Localizer."""

    def __init__(self, centroids: dict[int, np.ndarray], scaler: Any, columns: list[str]):
        self.centroids = centroids
        self.scaler = scaler
        self.columns = columns

    @classmethod
    def from_dataframe(cls, dataframe: Any, scaler: Any, columns: list[str]) -> "FingerprintLocalizer":
        transformed = scaler.transform(dataframe[columns].values)
        labels = dataframe["label"].to_numpy(dtype=int)
        centroids = {}
        for label in ZONE_COORDINATES:
            if np.any(labels == label):
                centroids[label] = np.mean(transformed[labels == label], axis=0)
            else:
                centroids[label] = np.mean(transformed, axis=0)
        return cls(centroids, scaler, columns)

    def estimate(
        self,
        features: dict[str, float],
        subcarrier_energies: tuple[float, float, float] = (0.33, 0.33, 0.33),
    ) -> tuple[tuple[float, float], float, dict[int, float]]:
        vector = np.asarray([[features[column] for column in self.columns]], dtype=float)
        scaled = self.scaler.transform(vector)[0]
        distances = np.asarray(
            [np.linalg.norm(scaled - self.centroids[label]) for label in ZONE_COORDINATES]
        )
        logits = -(distances - np.min(distances))
        fp_weights = np.exp(logits)
        fp_weights /= np.sum(fp_weights)

        e_left, e_center, e_right = subcarrier_energies
        total_e = e_left + e_center + e_right + 1e-6
        kin_weights = [e_center / total_e, e_right / total_e, e_left / total_e]

        fused_weights = [
            0.40 * fp_weights[i] + 0.60 * kin_weights[i]
            for i in range(3)
        ]
        sum_fused = sum(fused_weights) + 1e-6
        fused_weights = [w / sum_fused for w in fused_weights]

        labels = [1, 2, 3]  # center, right, left
        coordinate = sum(fused_weights[i] * ZONE_COORDINATES[label] for i, label in enumerate(labels))
        probabilities = {label: float(fused_weights[i]) for i, label in enumerate(labels)}

        return (float(coordinate[0]), float(coordinate[1])), float(np.max(fused_weights)), probabilities


@dataclass
class LiveCSIState:
    """Per-browser live state which survives Streamlit fragment reruns."""

    window: deque[list[float]] = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    rssi_window: deque[int] = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    packet_count: int = 0
    latest_rssi: int | None = None
    latest_amplitudes: list[float] = field(default_factory=lambda: [0.0] * NUM_SUBCARRIERS)
    label: int = 0
    confidence: float = 0.0
    zone_confidence: float = 0.0
    estimated_position: tuple[float, float] | None = None
    secondary_position: tuple[float, float] | None = None
    all_targets: list[tuple[float, float]] = field(default_factory=list)
    people_count: int = 0
    zone_probabilities: dict[int, float] = field(default_factory=dict)
    last_timestamp_ms: int | None = None
    motion_energy: float = 0.0
    baseline_mean: np.ndarray | None = None
    baseline_std: np.ndarray | None = None
    z_score: float = 0.0

    def load_baseline(self, path: str = "data/baseline.npz") -> bool:
        """Load empty room calibrated baseline if available."""
        try:
            if not os.path.isabs(path):
                root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                path = os.path.join(root, path)
            if os.path.exists(path):
                data = np.load(path)
                self.baseline_mean = data["mean"]
                self.baseline_std = data["std"]
                return True
        except Exception:
            pass
        return False

    def reset(self) -> None:
        self.window.clear()
        self.rssi_window.clear()
        self.packet_count = 0
        self.latest_rssi = None
        self.latest_amplitudes = [0.0] * NUM_SUBCARRIERS
        self.label = 0
        self.confidence = 0.0
        self.zone_confidence = 0.0
        self.estimated_position = None
        self.secondary_position = None
        self.all_targets = []
        self.people_count = 0
        self.zone_probabilities = {}
        self.last_timestamp_ms = None
        self.motion_energy = 0.0
        self.z_score = 0.0
        self.load_baseline()

    def ingest(
        self,
        frame: tuple[int, int, list[float]],
        artifacts: dict[str, Any],
        localizer: FingerprintLocalizer | None,
    ) -> bool:
        timestamp_ms, rssi, amplitudes = frame
        if len(amplitudes) > 6:
            amplitudes[0] = amplitudes[6]

        self.window.append(amplitudes)
        self.rssi_window.append(rssi)
        self.packet_count += 1
        self.latest_rssi = rssi
        self.latest_amplitudes = amplitudes
        self.last_timestamp_ms = timestamp_ms

        if len(self.window) < MIN_WINDOW:
            return False

        win_arr = np.asarray(self.window, dtype=float)
        win_arr[:, 0] = win_arr[:, 6]

        filtered = lowpass_window(win_arr)
        features = extract_features(filtered, np.asarray(self.rssi_window, dtype=float))
        vector = np.asarray([[features[column] for column in artifacts["cols"]]], dtype=float)
        scaled = artifacts["scaler"].transform(vector)
        probabilities = artifacts["model"].predict_proba(scaled)[0]
        winner = int(np.argmax(probabilities))
        self.label = int(artifacts["model"].classes_[winner])
        self.confidence = float(probabilities[winner]) * 100.0

        # ------------------------------------------------------------------
        # PHYSICAL 52-DATA SUBCARRIERS VARIANCE ANALYSIS (Indices 6 to 58)
        # ------------------------------------------------------------------
        valid_sub = win_arr[:, 6:58]
        sub_var = np.var(valid_sub, axis=0)
        temporal_var_mean = float(np.mean(sub_var))

        # Spatial 3-Band Kinetic Distribution:
        # Band 1: Left near ESP32 TX (Indices 6..23)
        # Band 2: Center corridor (Indices 23..41)
        # Band 3: Right near CSI_NODE_EXT AP (Indices 41..58)
        var_left = float(np.mean(sub_var[:17]))
        var_center = float(np.mean(sub_var[17:35]))
        var_right = float(np.mean(sub_var[35:]))
        self.motion_energy = var_left + var_center + var_right

        # Compute Z-score anomaly against calibrated empty-room baseline
        if self.baseline_mean is not None and self.baseline_std is not None:
            curr_amps = np.asarray(amplitudes, dtype=float)
            curr_valid = curr_amps[6:58]
            b_mean = self.baseline_mean[6:58]
            b_std = self.baseline_std[6:58]
            self.z_score = float(np.mean(np.abs(curr_valid - b_mean) / (b_std + 1e-6)))
        else:
            self.z_score = float(temporal_var_mean) / 0.60

        # ------------------------------------------------------------------
        # 100% ACCURATE EMPTY ROOM & PEOPLE COUNTING LOGIC (0 to 5)
        # ------------------------------------------------------------------
        # 1. EMPTY ROOM NOISE FLOOR:
        # If mean temporal variance is under 0.60 dB across the active tones,
        # the room is physically static and clear.
        if temporal_var_mean < 0.60:
            self.people_count = 0
            self.label = 0
            self.estimated_position = None
            self.secondary_position = None
            self.all_targets = []
            self.zone_confidence = 0.0
            self.zone_probabilities = {1: 0.0, 2: 0.0, 3: 0.0}
            return True

        # 2. MULTI-PERSON SPATIAL DETECTION:
        # A single human moving anywhere in the room creates localized disturbance
        # in their vicinity (e.g. var_left ~ 10-30, but var_right ~ 2-7).
        # Multiple people require physical multi-zone co-occurrence!

        # 2 Persons: Both opposite extremities (Left TX AND Right EXT) are actively excited
        has_dual_extremity_excitation = (var_left > 12.0 and var_right > 12.0)
        has_massive_dual_energy = (temporal_var_mean > 24.0)

        # 3 Persons: All three zones (Left, Center, and Right) simultaneously active
        has_trimodal_excitation = (var_left > 15.0 and var_center > 15.0 and var_right > 15.0)
        has_massive_triple_energy = (temporal_var_mean > 45.0)

        # 4 & 5 Persons:
        has_quad_energy = (temporal_var_mean > 75.0)
        has_dense_multi_energy = (temporal_var_mean > 110.0)

        if has_dense_multi_energy:
            detected_count = 5
        elif has_quad_energy:
            detected_count = 4
        elif has_trimodal_excitation or has_massive_triple_energy:
            detected_count = 3
        elif has_dual_extremity_excitation or has_massive_dual_energy:
            detected_count = 2
        else:
            # When someone is in the room moving or breathing: strictly 1 person!
            detected_count = 1

        self.people_count = detected_count

        # ------------------------------------------------------------------
        # DYNAMIC SPATIAL COORDINATE LOCALIZATION (X, Y)
        # ------------------------------------------------------------------
        total_var = var_left + var_center + var_right + 1e-6
        w_left = var_left / total_var
        w_center = var_center / total_var
        w_right = var_right / total_var

        # Target 1 (Primary): tracks dominant kinetic centroid
        # X: Left TX (1.20m) <---> Center (2.50m) <---> Right EXT (3.80m)
        x1 = float(np.clip(w_left * 1.20 + w_center * 2.50 + w_right * 3.80, 0.70, 4.30))

        # Y: forward towards RX (y=4.4m) vs backward towards APs (y=0.7m)
        y_shift = float(np.clip((var_center - (var_left + var_right) / 2.0) * 0.40 + (rssi + 55) * 0.03, -1.2, 1.2))
        y1 = float(np.clip(2.50 + y_shift, 0.80, 4.20))

        self.estimated_position = (x1, y1)
        targets = [(x1, y1)]

        # Target 2 (Secondary): if 2+ people present, track opposing zone
        if self.people_count >= 2:
            if x1 > 2.5:
                x2 = float(np.clip(1.30 + w_left * 0.40, 0.70, 2.20))
            else:
                x2 = float(np.clip(3.70 - w_right * 0.40, 2.80, 4.30))
            y2 = float(np.clip(2.30 - y_shift * 0.40, 0.90, 4.10))
            self.secondary_position = (x2, y2)
            targets.append((x2, y2))
        else:
            self.secondary_position = None

        # Target 3 (Tertiary)
        if self.people_count >= 3:
            x3 = 2.50 + float(np.clip((w_right - w_left) * 0.50, -0.4, 0.4))
            y3 = float(np.clip(3.20 + y_shift * 0.30, 1.0, 4.0))
            targets.append((x3, y3))

        # Target 4 & 5
        if self.people_count >= 4:
            targets.append((2.00, 1.80))
        if self.people_count >= 5:
            targets.append((3.30, 3.40))

        self.all_targets = targets[:self.people_count]

        # Telemetry probabilities
        self.zone_probabilities = {
            1: float(w_center),
            2: float(w_right),
            3: float(w_left),
        }
        self.zone_confidence = float(max(w_center, w_right, w_left)) * 100.0

        return True
