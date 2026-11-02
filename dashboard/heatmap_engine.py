"""
================================================================================
   HIGH-PRECISION HUMAN THERMAL RADIATION & SPATIAL TRACKING ENGINE
                     Wi-Fi CSI Human Sensing System
================================================================================
Renders high-definition thermal signatures on a calibrated 5m x 5m room grid
from CSI position estimates with:
- Dual-Gaussian anatomical human footprint (Head core + Torso radiation).
- Decaying thermal wake trails (motion momentum & trajectory history).
- RF Line-of-Sight (LOS) Fresnel beam and radar metric rings.
- Smooth exponential interpolation (zero-jitter tracking).

The renderer never invents a walking route.  It only moves when the runtime
passes a new CSI-derived estimate.
================================================================================
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# High-Precision Thermal Vision Colormap
# Pitch Black -> Cosmic Navy -> Quantum Cyan -> Infrared Amber -> Crimson Red -> White-Hot Core
THERMAL_VISION_CMAP = LinearSegmentedColormap.from_list(
    "csi_flir_thermal",
    [
        (0.00, "#020307"),   # Pure Pitch Black (Ambient empty space)
        (0.10, "#071226"),   # Deep Sub-RF Noise
        (0.22, "#0D3364"),   # Outer RF Multipath Glow
        (0.38, "#00A8E8"),   # Cyan Thermal Radiation
        (0.55, "#FF9E00"),   # Amber Infrared Body Envelope
        (0.72, "#E63946"),   # Red Heat Mantle
        (0.88, "#FFD166"),   # Intense Core Radiation
        (1.00, "#FFFFFF"),   # Peak White-Hot Body Core
    ],
    N=256
)

class SmoothThermalTracker:
    def __init__(self, room_width=5.0, room_height=5.0, resolution=100):
        self.w = room_width
        self.h = room_height
        self.res = resolution
        
        # Room Spatial Coordinate Grid
        self.x_coords = np.linspace(0, self.w, self.res)
        self.y_coords = np.linspace(0, self.h, self.res)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        
        # Persistent Heat Accumulator (maintains fluid fading wake trails)
        self.heat_canvas = np.zeros((self.res, self.res), dtype=float)
        
        # Continuous physics state
        self.curr_x = self.w / 2.0
        self.curr_y = self.h / 2.0
        self.target_x = self.w / 2.0
        self.target_y = self.h / 2.0
        self.prev_x = self.curr_x
        self.prev_y = self.curr_y
        
        # Velocity and momentum
        self.speed = 0.0
        self.last_update = time.monotonic()
        
        # Thermal intensity (0.0 = completely empty, 1.0 = fully present)
        self.curr_intensity = 0.0
        self.target_intensity = 0.0
        
        # Hardware Node Locations (Meters)
        self.tx_pos = (2.5, 0.45)
        self.rx_pos = (2.5, 4.55)
        
        # Spatial Anchors for the 5 AI Classes
        self.anchor_positions = {
            0: (2.5, 2.5, 0.0),    # Empty Room -> intensity fades to 0
            1: (2.5, 2.5, 1.0),    # Center (direct LOS interference)
            2: (3.75, 2.5, 1.0),   # Right Zone
            3: (1.25, 2.5, 1.0),   # Left Zone
            4: (2.5, 2.5, 1.0),    # Moving: retain last CSI-derived estimate
        }

    def update_target(self, label: int, estimated_position=None):
        """Accept a CSI-derived estimate; never generate a simulated trajectory."""
        if label not in self.anchor_positions:
            return

        _, _, target_intensity = self.anchor_positions[label]
        self.target_intensity = target_intensity
        if label == 0:
            return

        if estimated_position is not None:
            x, y = estimated_position
            self.target_x = float(np.clip(x, 0.0, self.w))
            self.target_y = float(np.clip(y, 0.0, self.h))
        elif label in (1, 2, 3):
            x, y, _ = self.anchor_positions[label]
            self.target_x, self.target_y = x, y
        # For label 4, lack of a CSI estimate means retain the last known
        # target rather than draw a fake orbit.

    def step_physics(self, dt: float, lerp_rate=0.42):
        """Smooth CSI estimates and calculate speed in metres per second."""
        self.prev_x = self.curr_x
        self.prev_y = self.curr_y
        
        # Smooth position update
        factor = min(1.0, lerp_rate * max(dt, 0.01) * 6.0)
        self.curr_x += (self.target_x - self.curr_x) * factor
        self.curr_y += (self.target_y - self.curr_y) * factor
        
        # Smooth intensity fade in / fade out
        self.curr_intensity += (self.target_intensity - self.curr_intensity) * min(1.0, 0.9 * max(dt, 0.01))
        
        # Calculate instant spatial velocity (m/frame)
        dist = np.hypot(self.curr_x - self.prev_x, self.curr_y - self.prev_y)
        instant_speed = dist / max(dt, 0.01)
        self.speed = 0.70 * self.speed + 0.30 * instant_speed

    def render_thermal_frame(self, label: int, live_rssi: float = -55.0, estimated_position=None):
        """
        Renders an ultra-high quality, dark thermal spatial radar frame:
        - Decaying motion wake trail.
        - Anatomical human thermal signature (Core + Torso + Aura).
        - Direct RF LOS beam and Fresnel zone lines.
        - Hardware node markers (TX & RX).
        - Telemetry coordinates HUD.
        """
        now = time.monotonic()
        dt = min(max(now - self.last_update, 0.01), 1.0)
        self.last_update = now
        self.update_target(label, estimated_position)
        self.step_physics(dt)
        
        # 1. Decay previous frame (creates realistic fluid thermal wake)
        self.heat_canvas *= 0.85
        
        # 2. Inject Anatomical Human Heat Signature if present
        if self.curr_intensity > 0.02:
            # A) Torso / Body Ellipse (anisotropic Gaussian)
            sigma_x = 0.38
            sigma_y = 0.46
            dx = (self.X - self.curr_x) / sigma_x
            dy = (self.Y - self.curr_y) / sigma_y
            torso_heat = np.exp(-0.5 * (dx**2 + dy**2)) * self.curr_intensity * 0.85
            
            # B) Concentrated Core Heat (Head & Upper Body core)
            core_dx = (self.X - self.curr_x) / 0.20
            core_dy = (self.Y - self.curr_y) / 0.22
            core_heat = np.exp(-0.5 * (core_dx**2 + core_dy**2)) * self.curr_intensity * 0.95
            
            self.heat_canvas += torso_heat + core_heat
            
        # 3. Spatial Gaussian Smoothing & Dynamic Normalization
        smoothed = gaussian_filter(self.heat_canvas, sigma=1.1)
        peak = float(np.max(smoothed))
        vmax = max(1.0, peak)
        normalized = np.clip(smoothed / vmax, 0.0, 1.0)
        
        # 4. Matplotlib Dark Theme Figure
        fig, ax = plt.subplots(figsize=(8.2, 7.8), facecolor="#030408")
        ax.set_facecolor("#030408")
        
        # A) Render Thermal Gradient
        im = ax.imshow(
            normalized,
            cmap=THERMAL_VISION_CMAP,
            origin="lower",
            extent=[0, self.w, 0, self.h],
            vmin=0.0,
            vmax=1.0,
            interpolation="bicubic"
        )
        
        # B) Draw Room Boundaries & Cyberpunk Radar Grid
        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.grid(True, color="#0A192F", linestyle="--", linewidth=0.7, alpha=0.7)
        
        # Radar Concentric Distance Rings around Room Center (2.5m, 2.5m)
        center_x, center_y = 2.5, 2.5
        for r in [0.8, 1.6, 2.3]:
            circle = plt.Circle((center_x, center_y), r, color="#00E5FF", fill=False,
                                linestyle=":", linewidth=0.7, alpha=0.25)
            ax.add_patch(circle)
            
        # Center Crosshair
        ax.axvline(center_x, color="#00E5FF", linestyle=":", linewidth=0.8, alpha=0.30)
        ax.axhline(center_y, color="#00E5FF", linestyle=":", linewidth=0.8, alpha=0.30)
        
        # C) Direct RF Line-of-Sight (LOS) Beam
        ax.plot(
            [self.tx_pos[0], self.rx_pos[0]],
            [self.tx_pos[1], self.rx_pos[1]],
            color="#00E5FF", linestyle="--", linewidth=1.4, alpha=0.45, zorder=3,
            label="RF Line-of-Sight (LOS)"
        )
        # First Fresnel Zone Ellipse boundary (dashed)
        fresnel_w = 0.95
        fresnel_h = 3.9
        fresnel_patch = matplotlib.patches.Ellipse(
            (2.5, 2.5), width=fresnel_w, height=fresnel_h,
            color="#00B4D8", fill=False, linestyle="--", linewidth=0.9, alpha=0.22, zorder=3
        )
        ax.add_patch(fresnel_patch)
        
        # D) Hardware Node Markers
        # Transmitter Node (ESP32 #1 - TX)
        ax.scatter([self.tx_pos[0]], [self.tx_pos[1]], color="#00E5FF", s=220, marker="^",
                   edgecolors="#FFFFFF", linewidth=1.6, zorder=6)
        ax.text(self.tx_pos[0], self.tx_pos[1] - 0.28, "📡 TX (ESP32 #1)", color="#00E5FF",
                fontsize=9, fontweight="bold", ha="center", va="top", zorder=6)
        
        # Receiver Node (ESP32 #2 - RX)
        ax.scatter([self.rx_pos[0]], [self.rx_pos[1]], color="#39FF14", s=220, marker="s",
                   edgecolors="#FFFFFF", linewidth=1.6, zorder=6)
        ax.text(self.rx_pos[0], self.rx_pos[1] + 0.28, "📥 RX (ESP32 #2)", color="#39FF14",
                fontsize=9, fontweight="bold", ha="center", va="bottom", zorder=6)
        
        # E) Human Target Reticle & Live Coordinate HUD
        if self.curr_intensity > 0.12:
            # Concentric Target Rings
            ax.scatter([self.curr_x], [self.curr_y], color="#FFFFFF", s=40, marker="o",
                       edgecolors="#FF0055", linewidth=2.0, zorder=8)
            reticle = plt.Circle((self.curr_x, self.curr_y), 0.35, color="#FF0055", fill=False,
                                 linestyle="-", linewidth=1.2, alpha=0.75, zorder=8)
            ax.add_patch(reticle)
            
            # Live HUD Annotation
            hud_text = f"👤 TARGET: ({self.curr_x:.2f}m, {self.curr_y:.2f}m)\nSPEED: {self.speed:.2f} m/s"
            ax.text(
                self.curr_x, self.curr_y + 0.45, hud_text,
                color="#FFFFFF", fontsize=8.5, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#090E17", edgecolor="#FF0055", linewidth=1.2, alpha=0.9),
                zorder=9
            )
            
        # F) Styling & Border Spines
        ax.set_xticks(np.arange(0, 6, 1))
        ax.set_yticks(np.arange(0, 6, 1))
        ax.set_xticklabels([f"{x}m" for x in range(6)], color="#64748B", fontsize=8)
        ax.set_yticklabels([f"{y}m" for y in range(6)], color="#64748B", fontsize=8)
        
        for spine in ax.spines.values():
            spine.set_color("#00B4D8")
            spine.set_linewidth(1.8)
            spine.set_alpha(0.85)
            
        ax.set_title("WI-FI CSI REAL-TIME SPATIAL THERMAL TRACKER (5.0m × 5.0m)",
                     color="#F1F5F9", fontsize=11, fontweight="bold", pad=12, family="sans-serif")
        
        plt.tight_layout()
        return fig
