"""Runtime helpers for the Wi-Fi CSI dashboard.

This module deliberately contains no Streamlit code.  Keeping serial parsing,
feature extraction, and localisation here makes the live path testable and
prevents Streamlit reruns from throwing away the CSI window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import iqr, kurtosis, skew
from sklearn.decomposition import PCA


NUM_SUBCARRIERS = 64
WINDOW_SIZE = 30
ZONE_COORDINATES = {
    1: np.array([2.5, 2.5]),  # centre
    2: np.array([3.75, 2.5]), # right
    3: np.array([1.25, 2.5]), # left
}


def parse_csi_serial_line(line: str) -> tuple[int, int, list[float]] | None:
    """Parse ``CSI_START,timestamp,rssi,count,amp...,CSI_END`` safely."""
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
        return timestamp_ms, rssi, amplitudes
    except (IndexError, TypeError, ValueError):
        return None


def lowpass_window(window: np.ndarray, cutoff: float = 0.15, order: int = 4) -> np.ndarray:
    """Use the same median + Butterworth pipeline as the training path."""
    try:
        cleaned = medfilt(window, kernel_size=(3, 1))
        b, a = butter(order, cutoff, btype="low")
        return filtfilt(b, a, cleaned, axis=0)
    except ValueError:
        # A short/incomplete window is not useful for inference, but returning
        # it makes this helper safe for diagnostics and unit tests.
        return window


def extract_features(window: np.ndarray, rssi: np.ndarray) -> dict[str, float]:
    """Extract the 35 features used by ``ml/ml_model.py``."""
    mean_subcarrier = np.mean(window, axis=0)
    flattened = window.flatten()
    temporal_variance = np.var(window, axis=0)
    diff_1 = np.diff(window, axis=0)
    diff_2 = np.diff(diff_1, axis=0)

    mean_signal = np.mean(window, axis=1)
    fft_values = np.abs(fft(mean_signal))[: len(window) // 2]
    fft_probability = fft_values / (np.sum(fft_values) + 1e-6)
    frequencies = np.linspace(0, 1, len(fft_values))
    spectral_centroid = float(np.sum(frequencies * fft_probability))
    spectral_spread = float(
        np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * fft_probability))
    )

    try:
        correlation = np.nan_to_num(np.corrcoef(window.T), nan=0.0)
        upper = np.triu_indices(NUM_SUBCARRIERS, k=1)
        corr_mean = float(np.mean(correlation[upper]))
        corr_std = float(np.std(correlation[upper]))
    except ValueError:
        corr_mean, corr_std = 0.0, 0.0

    try:
        explained = PCA(n_components=3).fit(window).explained_variance_ratio_
        pca_1, pca_2, pca_3 = (float(value) for value in explained)
    except ValueError:
        pca_1 = pca_2 = pca_3 = 0.0

    maximum = float(np.max(window))
    minimum = float(np.min(window))
    rms = float(np.sqrt(np.mean(window**2)))
    low_band = float(np.mean(mean_subcarrier[:21]))
    mid_band = float(np.mean(mean_subcarrier[21:43]))
    high_band = float(np.mean(mean_subcarrier[43:]))

    return {
        "mean_amp": float(np.mean(window)),
        "std_amp": float(np.std(window)),
        "max_amp": maximum,
        "min_amp": minimum,
        "range_amp": maximum - minimum,
        "variance_amp": float(np.var(window)),
        "rms_amp": rms,
        "crest_factor": float(maximum / (rms + 1e-6)),
        "energy_amp": float(np.sum(window**2)),
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
    """Estimate Left/Centre/Right from calibrated CSI feature centroids.

    A single Wi-Fi CSI link cannot uniquely solve a 2-D position.  This class
    therefore reports a calibrated *zone estimate*, not a fabricated metric
    GPS-like location.  The heatmap smoothly follows real changes between the
    three measured fingerprints and retains the last estimate while evidence
    is weak.
    """

    def __init__(self, centroids: dict[int, np.ndarray], scaler: Any, columns: list[str]):
        self.centroids = centroids
        self.scaler = scaler
        self.columns = columns

    @classmethod
    def from_dataframe(cls, dataframe: Any, scaler: Any, columns: list[str]) -> "FingerprintLocalizer":
        transformed = scaler.transform(dataframe[columns])
        labels = dataframe["label"].to_numpy(dtype=int)
        centroids = {
            label: np.mean(transformed[labels == label], axis=0)
            for label in ZONE_COORDINATES
            if np.any(labels == label)
        }
        if set(centroids) != set(ZONE_COORDINATES):
            raise ValueError("Calibration data must include labels 1, 2, and 3.")
        return cls(centroids, scaler, columns)

    def estimate(self, features: dict[str, float]) -> tuple[tuple[float, float], float, dict[int, float]]:
        vector = np.asarray([[features[column] for column in self.columns]], dtype=float)
        scaled = self.scaler.transform(vector)[0]
        distances = np.asarray(
            [np.linalg.norm(scaled - self.centroids[label]) for label in ZONE_COORDINATES]
        )
        # Stable softmax of negative distance.  The probabilities make the
        # transition continuous instead of jumping between fixed anchors.
        logits = -(distances - np.min(distances))
        weights = np.exp(logits)
        weights /= np.sum(weights)
        labels = list(ZONE_COORDINATES)
        coordinate = sum(weights[i] * ZONE_COORDINATES[label] for i, label in enumerate(labels))
        probabilities = {label: float(weights[i]) for i, label in enumerate(labels)}
        return (float(coordinate[0]), float(coordinate[1])), float(np.max(weights)), probabilities


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
    zone_probabilities: dict[int, float] = field(default_factory=dict)
    last_timestamp_ms: int | None = None

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
        self.zone_probabilities = {}
        self.last_timestamp_ms = None

    def ingest(
        self,
        frame: tuple[int, int, list[float]],
        artifacts: dict[str, Any],
        localizer: FingerprintLocalizer | None,
    ) -> bool:
        timestamp_ms, rssi, amplitudes = frame
        self.window.append(amplitudes)
        self.rssi_window.append(rssi)
        self.packet_count += 1
        self.latest_rssi = rssi
        self.latest_amplitudes = amplitudes
        self.last_timestamp_ms = timestamp_ms

        if len(self.window) < WINDOW_SIZE:
            return False

        filtered = lowpass_window(np.asarray(self.window, dtype=float))
        features = extract_features(filtered, np.asarray(self.rssi_window, dtype=float))
        vector = np.asarray([[features[column] for column in artifacts["cols"]]], dtype=float)
        scaled = artifacts["scaler"].transform(vector)
        probabilities = artifacts["model"].predict_proba(scaled)[0]
        winner = int(np.argmax(probabilities))
        self.label = int(artifacts["model"].classes_[winner])
        self.confidence = float(probabilities[winner]) * 100.0

        if self.label == 0:
            self.estimated_position = None
            self.zone_confidence = 0.0
            self.zone_probabilities = {}
        elif localizer is not None:
            position, confidence, zone_probabilities = localizer.estimate(features)
            self.estimated_position = position
            self.zone_confidence = confidence * 100.0
            self.zone_probabilities = zone_probabilities
        elif self.label in ZONE_COORDINATES:
            coordinate = ZONE_COORDINATES[self.label]
            self.estimated_position = (float(coordinate[0]), float(coordinate[1]))
            self.zone_confidence = self.confidence
        return True
