"""
================================================================================
   HIGH-PRECISION BIOMETRIC HUMAN THERMAL RADIATION & SPATIAL TRACKING ENGINE
                      Wi-Fi CSI Multi-Node Sensing System
================================================================================
Renders high-definition thermal signatures on a calibrated 5m x 5m room grid
from CSI position estimates and multi-node RF links:
- Multi-Node Sensing Topology:
    1. Node 1: TX (ESP32 #1)
    2. Node 2: EXT (CSI_NODE_EXT Access Point)
    3. Node 3: RX (ESP32 #2 on Laptop COM)
- Anatomical Human Biometric Thermal Footprint:
    * Head & Cervical Core (Peak white-hot radiation).
    * Shoulders & Chest (Broad lateral infrared mantle).
    * Torso & Abdomen (Tapering body heat column).
    * Flank Arm Heat Dispersion & Kinetic Strides.
    * Soft Ambient Infrared Body Aura.
- Decaying Thermal Wake Trails (Motion momentum & trajectory history).
- Multi-Link RF Beams & Calibrated Fresnel Coverage Zones.
- Zero-jitter smooth exponential interpolation.
================================================================================
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# High-Precision Thermal Vision Colormap (FLIR / Infrared Vision)
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
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.heading = 0.0  # radians (0 = forward toward Y+)
        self.last_update = time.monotonic()
        
        # Thermal intensity (0.0 = completely empty, 1.0 = fully present)
        self.curr_intensity = 0.0
        self.target_intensity = 0.0
        
        # 3 Multi-Node Sensing Network Locations (Meters)
        self.tx_pos = (1.0, 0.70)   # Node 1: TX (ESP32 #1 - Left corner)
        self.ext_pos = (4.0, 0.70)  # Node 2: EXT (CSI_NODE_EXT AP - Right corner)
        self.rx_pos = (2.5, 4.40)   # Node 3: RX (ESP32 #2 - Back Center)
        
        # Spatial Anchors for the 3 Primary Detection Zones
        self.zone_anchors = {
            1: (2.50, 2.50),   # Zone 1: Center Crossing (LOS Beam)
            2: (3.75, 2.30),   # Zone 2: Right Zone (Near CSI_NODE_EXT)
            3: (1.25, 2.30),   # Zone 3: Left Zone (Near ESP32 TX)
        }
        
        # Active Zone Name for HUD
        self.active_zone_name = "EMPTY ROOM"

    def update_target(self, label: int, estimated_position=None):
        """Update target coordinates and thermal intensity based on CSI inference."""
        if label == 0:
            self.target_intensity = 0.0
            self.active_zone_name = "EMPTY ROOM"
            return
            
        self.target_intensity = 1.0
        
        if estimated_position is not None:
            x, y = estimated_position
            self.target_x = float(np.clip(x, 0.3, self.w - 0.3))
            self.target_y = float(np.clip(y, 0.3, self.h - 0.3))
            
            # Determine active zone from coordinates
            if self.target_x < 1.8:
                self.active_zone_name = "ZONE 3 (LEFT - TX)"
            elif self.target_x > 3.2:
                self.active_zone_name = "ZONE 2 (RIGHT - EXTENDER)"
            else:
                self.active_zone_name = "ZONE 1 (CENTER - LOS)"
        elif label in self.zone_anchors:
            x, y = self.zone_anchors[label]
            self.target_x, self.target_y = x, y
            names = {1: "ZONE 1 (CENTER)", 2: "ZONE 2 (RIGHT)", 3: "ZONE 3 (LEFT)", 4: "MOTION CORRIDOR"}
            self.active_zone_name = names.get(label, "ACTIVE ZONE")
        elif label == 4:
            self.active_zone_name = "MOTION CORRIDOR"

    def step_physics(self, dt: float, lerp_rate=0.55):
        """Smoothly interpolate target position and calculate velocity & orientation."""
        self.prev_x = self.curr_x
        self.prev_y = self.curr_y
        
        # Smooth exponential interpolation (LERP)
        factor = min(1.0, lerp_rate * max(dt, 0.01) * 7.5)
        self.curr_x += (self.target_x - self.curr_x) * factor
        self.curr_y += (self.target_y - self.curr_y) * factor
        
        # Smooth intensity fade in / fade out
        self.curr_intensity += (self.target_intensity - self.curr_intensity) * min(1.0, 1.4 * max(dt, 0.01))
        
        # Calculate instant spatial velocity and heading vector
        self.vx = (self.curr_x - self.prev_x) / max(dt, 0.01)
        self.vy = (self.curr_y - self.prev_y) / max(dt, 0.01)
        instant_speed = np.hypot(self.vx, self.vy)
        self.speed = 0.65 * self.speed + 0.35 * instant_speed
        
        if instant_speed > 0.08:
            # Update heading direction when moving
            target_heading = np.arctan2(self.vy, self.vx)
            # Smooth angle interpolation
            self.heading = 0.70 * self.heading + 0.30 * target_heading

    def render_thermal_frame(self, label: int, live_rssi: float = -55.0, estimated_position=None):
        """
        Renders an ultra-high quality, dark thermal spatial radar frame:
        - 3-Node Sensing Network (ESP32 #1, CSI_NODE_EXT, ESP32 #2).
        - Anatomical human thermal silhouette (Head + Shoulders + Torso + Aura).
        - Decaying motion wake trail (fluid footsteps).
        - Multi-node RF Line-of-Sight rays and Fresnel zones.
        - Biometric targeting HUD with live coordinates and velocity.
        """
        now = time.monotonic()
        dt = min(max(now - self.last_update, 0.01), 1.0)
        self.last_update = now
        self.update_target(label, estimated_position)
        self.step_physics(dt)
        
        # 1. Decay previous frame (creates realistic fluid thermal wake trailing the person)
        self.heat_canvas *= 0.86
        
        # 2. Inject Biometric Human Thermal Signature if present
        if self.curr_intensity > 0.02:
            # Coordinate transformation rotated by movement heading
            cos_h = np.cos(self.heading)
            sin_h = np.sin(self.heading)
            dx = self.X - self.curr_x
            dy = self.Y - self.curr_y
            
            # Rotated coordinates: X_rot = lateral (shoulders), Y_rot = longitudinal (head-to-feet)
            X_rot = dx * cos_h + dy * sin_h
            Y_rot = -dx * sin_h + dy * cos_h
            
            # A) Head & Brain Core (Concentrated peak white-hot spot)
            head_dx = X_rot / 0.16
            head_dy = (Y_rot - 0.22) / 0.18
            head_heat = np.exp(-0.5 * (head_dx**2 + head_dy**2)) * self.curr_intensity * 1.00
            
            # B) Shoulders & Chest Mantle (Broad lateral infrared signature)
            shoulder_dx = X_rot / 0.42
            shoulder_dy = (Y_rot - 0.06) / 0.20
            shoulder_heat = np.exp(-0.5 * (shoulder_dx**2 + shoulder_dy**2)) * self.curr_intensity * 0.92
            
            # C) Torso & Abdomen Core (Tapering body warmth column)
            torso_dx = X_rot / 0.28
            torso_dy = (Y_rot + 0.15) / 0.38
            torso_heat = np.exp(-0.5 * (torso_dx**2 + torso_dy**2)) * self.curr_intensity * 0.86
            
            # D) Flank Arm Heat Radiations (Left & Right Arms)
            arm_l = np.exp(-0.5 * (((X_rot - 0.32) / 0.18)**2 + ((Y_rot - 0.02) / 0.26)**2)) * self.curr_intensity * 0.65
            arm_r = np.exp(-0.5 * (((X_rot + 0.32) / 0.18)**2 + ((Y_rot - 0.02) / 0.26)**2)) * self.curr_intensity * 0.65
            
            # E) Ambient Infrared Dispersion Aura (Soft body heat envelope)
            body_dist = np.hypot(dx, dy)
            body_aura = np.exp(-0.5 * (body_dist / 0.70)**2) * self.curr_intensity * 0.38
            
            # Inject combined anatomical human heat
            human_footprint = head_heat + shoulder_heat + torso_heat + arm_l + arm_r + body_aura
            self.heat_canvas += human_footprint
            
        # 3. Spatial Gaussian Smoothing & Dynamic FLIR Normalization
        smoothed = gaussian_filter(self.heat_canvas, sigma=1.05)
        peak = float(np.max(smoothed))
        vmax = max(1.0, peak)
        normalized = np.clip(smoothed / vmax, 0.0, 1.0)
        
        # 4. Matplotlib Dark Theme Radar Figure
        fig, ax = plt.subplots(figsize=(8.4, 7.8), facecolor="#030408")
        ax.set_facecolor("#030408")
        
        # A) Render Thermal Gradient
        ax.imshow(
            normalized,
            cmap=THERMAL_VISION_CMAP,
            origin="lower",
            extent=[0, self.w, 0, self.h],
            vmin=0.0,
            vmax=1.0,
            interpolation="bicubic"
        )
        
        # B) Room Boundaries & Radar Metric Grid
        ax.set_xlim(0, self.w)
        ax.set_ylim(0, self.h)
        ax.grid(True, color="#0A1E3F", linestyle="--", linewidth=0.7, alpha=0.65)
        
        # Radar Concentric Distance Rings around Room Center (2.5m, 2.5m)
        center_x, center_y = 2.5, 2.5
        for r in [0.9, 1.8, 2.4]:
            circle = plt.Circle((center_x, center_y), r, color="#00E5FF", fill=False,
                                linestyle=":", linewidth=0.7, alpha=0.22)
            ax.add_patch(circle)
            
        # Center Crosshair
        ax.axvline(center_x, color="#00E5FF", linestyle=":", linewidth=0.8, alpha=0.25)
        ax.axhline(center_y, color="#00E5FF", linestyle=":", linewidth=0.8, alpha=0.25)
        
        # C) Multi-Node RF Beams (Triangular Multi-Static Geometry)
        # 1. Link 1: TX (ESP32 #1) <--> RX (ESP32 #2)
        ax.plot([self.tx_pos[0], self.rx_pos[0]], [self.tx_pos[1], self.rx_pos[1]],
                color="#00E5FF", linestyle="--", linewidth=1.3, alpha=0.40, zorder=3)
        # 2. Link 2: EXT (CSI_NODE_EXT) <--> RX (ESP32 #2)
        ax.plot([self.ext_pos[0], self.rx_pos[0]], [self.ext_pos[1], self.rx_pos[1]],
                color="#D946EF", linestyle="--", linewidth=1.3, alpha=0.40, zorder=3)
        # 3. Link 3: TX (ESP32 #1) <--> EXT (CSI_NODE_EXT)
        ax.plot([self.tx_pos[0], self.ext_pos[0]], [self.tx_pos[1], self.ext_pos[1]],
                color="#FBBF24", linestyle=":", linewidth=1.0, alpha=0.30, zorder=3)

        # D) Calibrated Sensing Zones (Zone 3: Left, Zone 1: Center, Zone 2: Right)
        zones = [
            (1.30, 2.30, "ZONE 3 (LEFT - TX)", "#00E5FF"),
            (2.50, 2.50, "ZONE 1 (CENTER)", "#38BDF8"),
            (3.70, 2.30, "ZONE 2 (RIGHT - EXT)", "#D946EF"),
        ]
        for zx, zy, zname, zcolor in zones:
            zone_patch = plt.Circle((zx, zy), 0.65, color=zcolor, fill=False,
                                    linestyle="--", linewidth=0.8, alpha=0.28, zorder=2)
            ax.add_patch(zone_patch)
            ax.text(zx, zy - 0.78, zname, color=zcolor, fontsize=7.5,
                    ha="center", va="top", alpha=0.60, fontweight="bold")
        
        # E) Hardware Node Markers (3-Node Multi-Static Network)
        # 1. Node 1: TX (ESP32 #1)
        ax.scatter([self.tx_pos[0]], [self.tx_pos[1]], color="#00E5FF", s=220, marker="^",
                   edgecolors="#FFFFFF", linewidth=1.6, zorder=6)
        ax.text(self.tx_pos[0], self.tx_pos[1] - 0.32, "TX (ESP32 #1)", color="#00E5FF",
                fontsize=8.5, fontweight="bold", ha="center", va="top", zorder=6)
        
        # 2. Node 2: EXT (CSI_NODE_EXT Access Point)
        ax.scatter([self.ext_pos[0]], [self.ext_pos[1]], color="#D946EF", s=220, marker="D",
                   edgecolors="#FFFFFF", linewidth=1.6, zorder=6)
        ax.text(self.ext_pos[0], self.ext_pos[1] - 0.32, "EXT (CSI_NODE_EXT)", color="#D946EF",
                fontsize=8.5, fontweight="bold", ha="center", va="top", zorder=6)
        
        # 3. Node 3: RX (ESP32 #2 on Laptop COM)
        ax.scatter([self.rx_pos[0]], [self.rx_pos[1]], color="#39FF14", s=220, marker="s",
                   edgecolors="#FFFFFF", linewidth=1.6, zorder=6)
        ax.text(self.rx_pos[0], self.rx_pos[1] + 0.32, "RX (ESP32 #2)", color="#39FF14",
                fontsize=8.5, fontweight="bold", ha="center", va="bottom", zorder=6)
        
        # F) Biometric Human Target Reticle & Live Coordinate HUD
        if self.curr_intensity > 0.08:
            # Concentric Target Reticle
            ax.scatter([self.curr_x], [self.curr_y], color="#FFFFFF", s=45, marker="o",
                       edgecolors="#FF0055", linewidth=2.2, zorder=8)
            reticle = plt.Circle((self.curr_x, self.curr_y), 0.38, color="#FF0055", fill=False,
                                 linestyle="-", linewidth=1.4, alpha=0.85, zorder=8)
            ax.add_patch(reticle)
            
            # Velocity heading vector arrow
            if self.speed > 0.12:
                arrow_len = min(0.65, self.speed * 0.45)
                ax.arrow(
                    self.curr_x, self.curr_y,
                    arrow_len * np.cos(self.heading), arrow_len * np.sin(self.heading),
                    color="#FF0055", width=0.03, head_width=0.12, head_length=0.12,
                    alpha=0.90, zorder=9
                )
            
            # Live Biometric HUD Callout
            hud_text = (
                f"🧍 BIOMETRIC TARGET IDENTIFIED\n"
                f"COORD: ({self.curr_x:.2f}m, {self.curr_y:.2f}m)\n"
                f"SPEED: {self.speed:.2f} m/s | INTENSITY: {int(self.curr_intensity * 100)}%\n"
                f"{self.active_zone_name}"
            )
            
            hud_y_offset = 0.55 if self.curr_y < 3.8 else -0.75
            ax.text(
                self.curr_x, self.curr_y + hud_y_offset, hud_text,
                color="#FFFFFF", fontsize=8.2, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.40", facecolor="#090E17", edgecolor="#FF0055", linewidth=1.4, alpha=0.92),
                zorder=10
            )
        else:
            # Empty Room Standby Badge
            ax.text(
                2.5, 2.5, "ROOM CLEAR — LISTENING FOR HUMAN PERTURBATION",
                color="#64748B", fontsize=9.0, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#090E17", edgecolor="#1E293B", linewidth=1.0, alpha=0.80),
                zorder=7
            )
            
        # G) Styling & Metric Border Spines
        ax.set_xticks(np.arange(0, 6, 1))
        ax.set_yticks(np.arange(0, 6, 1))
        ax.set_xticklabels([f"{x}m" for x in range(6)], color="#64748B", fontsize=8)
        ax.set_yticklabels([f"{y}m" for y in range(6)], color="#64748B", fontsize=8)
        
        for spine in ax.spines.values():
            spine.set_color("#00E5FF")
            spine.set_linewidth(1.8)
            spine.set_alpha(0.80)
            
        ax.set_title("WI-FI CSI 3-NODE SPATIAL THERMAL TRACKER (5.0m × 5.0m)",
                     color="#F1F5F9", fontsize=11, fontweight="bold", pad=12, family="sans-serif")
        
        plt.tight_layout()
        return fig
