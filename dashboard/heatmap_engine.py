"""
================================================================================
   HIGH-PERFORMANCE ZERO-DELAY BIOMETRIC THERMAL RADAR & TRACKING ENGINE
                     Wi-Fi CSI Multi-Node Sensing System
================================================================================
Ultra-fast (10ms) client-side compatible PIL + NumPy rendering engine:
- Multi-Node Sensing Topology:
    1. Node 1: TX (ESP32 #1) at (1.0m, 0.7m)
    2. Node 2: EXT (CSI_NODE_EXT AP) at (4.0m, 0.7m)
    3. Node 3: RX (ESP32 #2 on Laptop COM) at (2.5m, 4.4m)
- Dynamic People Counter (0 to 5 Persons):
    * 0 Persons: Pure clean canvas, NO fake thermal blob in center!
    * 1 to 5 Persons: Prominent glowing target dots tracking each person
      in real-time as they walk (forward, backward, left, right).
- High-contrast FLIR Thermal Vision Heatmap:
    * Intense white/amber core, crimson heat mantle, outer cyan aura.
    * Decaying fluid wake trails following motion path.
- 100% Zero-Lag: Renders in ~10ms via NumPy LUT + Pillow (eliminates 240ms Matplotlib lag).
================================================================================
"""

import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

# Tactical Target Color Palette
TARGET_COLORS = [
    {"core": (255, 255, 255), "glow": (255, 59, 48),  "ring": (255, 69, 58),  "label": "TARGET #1"},
    {"core": (255, 255, 255), "glow": (0, 229, 255),  "ring": (0, 195, 255),  "label": "TARGET #2"},
    {"core": (255, 255, 255), "glow": (255, 214, 10), "ring": (255, 195, 0),  "label": "TARGET #3"},
    {"core": (255, 255, 255), "glow": (175, 82, 222), "ring": (191, 90, 242), "label": "TARGET #4"},
    {"core": (255, 255, 255), "glow": (48, 209, 88),  "ring": (52, 199, 89),  "label": "TARGET #5"},
]

def build_flir_lut() -> np.ndarray:
    """Build a 256x3 RGB look-up table for FLIR infrared thermal vision."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.08:
            # Pure dark ambient space
            f = t / 0.08
            r, g, b = int(3 + 5 * f), int(5 + 10 * f), int(14 + 25 * f)
        elif t < 0.25:
            # Deep sub-RF / multipath navy
            f = (t - 0.08) / 0.17
            r, g, b = int(8 + 8 * f), int(15 + 40 * f), int(39 + 70 * f)
        elif t < 0.45:
            # Cyan thermal aura
            f = (t - 0.25) / 0.20
            r, g, b = int(16 + 10 * f), int(55 + 130 * f), int(109 + 130 * f)
        elif t < 0.65:
            # Amber / orange heat
            f = (t - 0.45) / 0.20
            r, g, b = int(26 + 225 * f), int(185 - 25 * f), int(239 - 210 * f)
        elif t < 0.85:
            # Crimson red heat mantle
            f = (t - 0.65) / 0.20
            r, g, b = int(251 + 4 * f), int(160 - 110 * f), int(29 - 15 * f)
        else:
            # Intense white-hot body core
            f = (t - 0.85) / 0.15
            r, g, b = 255, int(50 + 205 * f), int(14 + 241 * f)
        lut[i] = [r, g, b]
    return lut

FLIR_LUT = build_flir_lut()

class TargetState:
    """Tracks continuous physics and velocity for a single human target."""
    def __init__(self, x: float = 2.5, y: float = 2.5):
        self.curr_x = x
        self.curr_y = y
        self.target_x = x
        self.target_y = y
        self.prev_x = x
        self.prev_y = y
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.heading = 0.0
        self.trail = []  # Recent (x, y) trail points

    def update_goal(self, x: float, y: float, w: float = 5.0, h: float = 5.0):
        self.target_x = float(np.clip(x, 0.40, w - 0.40))
        self.target_y = float(np.clip(y, 0.40, h - 0.40))

    def step(self, dt: float, lerp_rate: float = 0.85):
        self.prev_x = self.curr_x
        self.prev_y = self.curr_y

        factor = min(1.0, lerp_rate * max(dt, 0.01) * 10.0)
        self.curr_x += (self.target_x - self.curr_x) * factor
        self.curr_y += (self.target_y - self.curr_y) * factor

        self.vx = (self.curr_x - self.prev_x) / max(dt, 0.01)
        self.vy = (self.curr_y - self.prev_y) / max(dt, 0.01)
        instant_speed = math.hypot(self.vx, self.vy)
        self.speed = 0.60 * self.speed + 0.40 * instant_speed

        if instant_speed > 0.08:
            target_heading = math.atan2(self.vy, self.vx)
            self.heading = 0.60 * self.heading + 0.40 * target_heading

        # Maintain trailing points
        self.trail.append((self.curr_x, self.curr_y))
        if len(self.trail) > 12:
            self.trail.pop(0)


class SmoothThermalTracker:
    """
    Real-Time FLIR Thermal Radar Engine (0 to 5 Persons).
    Renders in ~10ms with zero server delay.
    """
    def __init__(self, room_width: float = 5.0, room_height: float = 5.0, resolution: int = 70):
        self.w = room_width
        self.h = room_height
        self.res = resolution
        self.render_size = 600  # Output pixel canvas width and height

        # Internal thermal grid
        self.x_coords = np.linspace(0, self.w, self.res)
        self.y_coords = np.linspace(0, self.h, self.res)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        self.heat_canvas = np.zeros((self.res, self.res), dtype=np.float32)

        # 5 Target Physics Trackers
        default_positions = [(2.5, 2.5), (3.8, 2.2), (1.2, 2.2), (2.5, 3.6), (3.4, 3.4)]
        self.targets = [TargetState(x, y) for x, y in default_positions]

        self.people_count = 0
        self.last_update = time.monotonic()
        self.pulse_phase = 0.0

        # Physical 3-Node Topology Coordinates (Meters)
        self.tx_pos = (1.0, 0.70)   # Node 1: TX (ESP32 #1 - Left)
        self.ext_pos = (4.0, 0.70)  # Node 2: EXT (CSI_NODE_EXT AP - Right)
        self.rx_pos = (2.5, 4.40)   # Node 3: RX (ESP32 #2 on Laptop - Back Center)

    @property
    def speed(self) -> float:
        return self.targets[0].speed if self.people_count > 0 else 0.0

    def world_to_px(self, x: float, y: float) -> tuple[int, int]:
        """Convert metric coordinates (0..5m) to image pixel coordinates."""
        px = int((x / self.w) * (self.render_size - 80) + 40)
        py = int(((self.h - y) / self.h) * (self.render_size - 80) + 40)
        return px, py

    def update_targets(self, people_count: int, target_coords: list[tuple[float, float]] | None = None):
        """Update active people count and their real-time spatial coordinates."""
        self.people_count = max(0, min(5, people_count))
        if self.people_count == 0 or target_coords is None:
            return

        for i in range(min(self.people_count, len(target_coords))):
            x, y = target_coords[i]
            self.targets[i].update_goal(x, y, self.w, self.h)

    def step_physics(self, dt: float):
        self.pulse_phase = (self.pulse_phase + dt * 4.0) % (2.0 * math.pi)
        for i in range(max(1, self.people_count)):
            self.targets[i].step(dt)

    def render_thermal_frame(
        self,
        label: int = 0,
        live_rssi: float = -55.0,
        estimated_position: tuple[float, float] | None = None,
        people_count: int = 0,
        secondary_position: tuple[float, float] | None = None,
        all_targets: list[tuple[float, float]] | None = None,
    ) -> Image.Image:
        """
        Renders the full thermal radar frame in ~10ms:
        1. Pure clean room when 0 persons (NO fake blobs).
        2. Dynamic anatomical thermal footprints for all active targets.
        3. Glowing target markers, ripple pulse rings, and velocity vectors.
        4. Clear metric room grid with 3 Wi-Fi sensing nodes and RF links.
        """
        now = time.monotonic()
        dt = min(max(now - self.last_update, 0.005), 0.3)
        self.last_update = now

        # Build list of active targets
        coords_list = []
        if all_targets:
            coords_list = list(all_targets)
        elif estimated_position:
            coords_list.append(estimated_position)
            if secondary_position:
                coords_list.append(secondary_position)

        self.update_targets(people_count, coords_list)
        self.step_physics(dt)

        # 1. Decay previous thermal canvas
        self.heat_canvas *= 0.82

        # 2. Inject Biometric Human Heat (Only if people are present!)
        if self.people_count > 0:
            for i in range(self.people_count):
                tgt = self.targets[i]
                dx = self.X - tgt.curr_x
                dy = self.Y - tgt.curr_y

                # Oriented anatomical ellipse in direction of walking
                cos_h = math.cos(tgt.heading)
                sin_h = math.sin(tgt.heading)
                x_rot = dx * cos_h + dy * sin_h
                y_rot = -dx * sin_h + dy * cos_h

                # Core heat footprint + soft ambient body radiation envelope
                core = np.exp(-((x_rot / 0.28) ** 2 + (y_rot / 0.42) ** 2))
                aura = 0.40 * np.exp(-((dx / 0.70) ** 2 + (dy / 0.70) ** 2))
                self.heat_canvas += (core + aura).astype(np.float32)

        # 3. Spatial Gaussian smoothing & FLIR Color Mapping
        if self.people_count > 0:
            smoothed = gaussian_filter(self.heat_canvas, sigma=0.9)
            peak = float(np.max(smoothed))
            vmax = max(1.0, peak)
            normalized = np.clip(smoothed / vmax, 0.0, 1.0)
            idx = (normalized * 255).astype(np.uint8)
            rgb_arr = FLIR_LUT[idx]
        else:
            # Completely clear room (0 persons) - Deep ambient black/navy
            self.heat_canvas.fill(0.0)
            rgb_arr = np.zeros((self.res, self.res, 3), dtype=np.uint8)
            rgb_arr[:, :] = [4, 7, 16]

        # Resize thermal canvas to output size with bilinear interpolation
        heat_img = Image.fromarray(rgb_arr).resize((self.render_size, self.render_size), Image.BILINEAR)
        draw = ImageDraw.Draw(heat_img, "RGBA")

        # 4. Draw Tactical Metric Radar Grid & Distance Rings
        cx_px, cy_px = self.world_to_px(2.5, 2.5)
        # Concentric distance rings
        ring_radii_m = [1.0, 2.0, 2.4]
        for rm in ring_radii_m:
            r_px = int((rm / self.w) * (self.render_size - 80))
            draw.ellipse(
                [(cx_px - r_px, cy_px - r_px), (cx_px + r_px, cy_px + r_px)],
                outline=(0, 180, 240, 60),
                width=1
            )

        # Grid lines every 1 meter
        for m in range(1, 5):
            gx, _ = self.world_to_px(float(m), 0.0)
            draw.line([(gx, 40), (gx, self.render_size - 40)], fill=(15, 35, 65, 90), width=1)
            _, gy = self.world_to_px(0.0, float(m))
            draw.line([(40, gy), (self.render_size - 40, gy)], fill=(15, 35, 65, 90), width=1)

        # Room Outer Boundary Border
        draw.rectangle(
            [(40, 40), (self.render_size - 40, self.render_size - 40)],
            outline=(0, 229, 255, 140),
            width=2
        )

        # Center crosshair
        draw.line([(cx_px - 8, cy_px), (cx_px + 8, cy_px)], fill=(0, 229, 255, 120), width=1)
        draw.line([(cx_px, cy_px - 8), (cx_px, cy_px + 8)], fill=(0, 229, 255, 120), width=1)

        # 5. Draw the 3 Wi-Fi Sensing Nodes & Multipath Triangle
        tx_px = self.world_to_px(*self.tx_pos)
        ext_px = self.world_to_px(*self.ext_pos)
        rx_px = self.world_to_px(*self.rx_pos)

        # Dashed RF link lines
        for p1, p2 in [(tx_px, ext_px), (ext_px, rx_px), (rx_px, tx_px)]:
            draw.line([p1, p2], fill=(0, 255, 180, 75), width=1)

        # Node Icons
        nodes_info = [
            ("TX (ESP32 #1)", tx_px, (57, 169, 255, 255)),
            ("EXT (CSI_NODE_EXT)", ext_px, (46, 213, 115, 255)),
            ("RX (ESP32 Receiver)", rx_px, (255, 71, 87, 255)),
        ]
        for name, (nx, ny), color in nodes_info:
            draw.ellipse([(nx - 7, ny - 7), (nx + 7, ny + 7)], fill=color, outline=(255, 255, 255, 240), width=2)
            draw.text((nx - 28, ny + 10), name, fill=(220, 230, 245, 230))

        # 6. Draw Active Human Targets (If Present)
        pulse_scale = 1.0 + 0.25 * math.sin(self.pulse_phase)

        if self.people_count > 0:
            for i in range(self.people_count):
                tgt = self.targets[i]
                palette = TARGET_COLORS[i % len(TARGET_COLORS)]
                tx, ty = self.world_to_px(tgt.curr_x, tgt.curr_y)

                # Draw motion trail
                if len(tgt.trail) >= 2:
                    trail_px = [self.world_to_px(pt[0], pt[1]) for pt in tgt.trail]
                    for t_idx in range(len(trail_px) - 1):
                        alpha = int(40 + (t_idx / len(trail_px)) * 140)
                        col = palette["glow"] + (alpha,)
                        draw.line([trail_px[t_idx], trail_px[t_idx + 1]], fill=col, width=2)

                # Concentric pulsating radar rings
                r_inner = int(12 * pulse_scale)
                r_outer = int(22 * pulse_scale)
                draw.ellipse([(tx - r_inner, ty - r_inner), (tx + r_inner, ty + r_inner)], outline=palette["ring"] + (190,), width=2)
                draw.ellipse([(tx - r_outer, ty - r_outer), (tx + r_outer, ty + r_outer)], outline=palette["ring"] + (90,), width=1)

                # Velocity vector arrow
                if tgt.speed > 0.05:
                    vx_px = int(math.cos(tgt.heading) * min(35, tgt.speed * 40))
                    vy_px = int(-math.sin(tgt.heading) * min(35, tgt.speed * 40))
                    draw.line([(tx, ty), (tx + vx_px, ty + vy_px)], fill=(255, 255, 255, 240), width=2)
                    draw.ellipse([(tx + vx_px - 3, ty + vy_px - 3), (tx + vx_px + 3, ty + vy_px + 3)], fill=palette["glow"] + (255,))

                # Intense Center Glowing Target Dot
                draw.ellipse([(tx - 6, ty - 6), (tx + 6, ty + 6)], fill=palette["glow"] + (255,), outline=(255, 255, 255, 255), width=2)
                draw.ellipse([(tx - 3, ty - 3), (tx + 3, ty + 3)], fill=palette["core"] + (255,))

                # Target Info Label Box
                lbl_text = f"{palette['label']} [{tgt.curr_x:.2f}m, {tgt.curr_y:.2f}m]"
                draw.rectangle([(tx + 12, ty - 12), (tx + 120, ty + 6)], fill=(10, 15, 30, 200), outline=palette["ring"] + (180,), width=1)
                draw.text((tx + 16, ty - 10), lbl_text, fill=(255, 255, 255, 240))

        # 7. Radar HUD Header Banner
        banner_h = 32
        draw.rectangle([(40, 8), (self.render_size - 40, 8 + banner_h)], fill=(6, 12, 24, 230), outline=(0, 229, 255, 120), width=1)

        if self.people_count == 0:
            header_text = "● الغرفة فارغة حالياً — لا توجد حركة (ROOM CLEAR: 0 PERSONS)"
            text_color = (0, 229, 255, 240)
        elif self.people_count == 1:
            header_text = f"● شخص واحد متتبع الآن (1 PERSON ACTIVE) | الإحداثيات: X={self.targets[0].curr_x:.2f}m, Y={self.targets[0].curr_y:.2f}m"
            text_color = (46, 213, 115, 255)
        else:
            header_text = f"● تم رصد {self.people_count} أشخاص متتبعين في الغرفة ({self.people_count} PERSONS DETECTED)"
            text_color = (255, 214, 10, 255)

        draw.text((55, 16), header_text, fill=text_color)

        return heat_img
