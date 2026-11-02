#!/usr/bin/env python3
"""
Collect five labeled Wi-Fi CSI datasets from the ESP32 receiver.

Run:
    python3 -m pip install -r requirements.txt
    python3 collect_data.py --list-ports
    python3 collect_data.py --port /dev/cu.usbserial-XXXX

The receiver must already be flashed and connected to the transmitter.
Each scenario is recorded for 60 seconds. Packets are saved only when
they contain exactly 64 amplitudes, matching the project CSV format.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import serial
from serial.tools import list_ports

BAUD = 115200
DURATION_SECONDS = 60
NUM_SUBCARRIERS = 64
MIN_SAMPLES = 500

SCENARIOS = {
    0: "Empty Room",
    1: "Person Center",
    2: "Person Right",
    3: "Person Left",
    4: "Person Moving",
}


def show_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports found. Connect the ESP32 with a USB data cable.")
        return
    print("Serial ports found:")
    for port in ports:
        print(f"  {port.device}  -  {port.description}")


def parse_csi_line(raw_line):
    """Return (timestamp_ms, rssi, amplitudes) or None for non-CSI/invalid lines."""
    line = raw_line.strip()
    if not line.startswith("CSI_START,") or not line.endswith(",CSI_END"):
        return None

    try:
        parts = line[:-len(",CSI_END")].split(",")
        if len(parts) < 4:
            return None

        timestamp_ms = int(parts[1])
        rssi = int(parts[2])
        num_sub = int(parts[3])
        amplitudes = [float(value) for value in parts[4:]]

        if num_sub != NUM_SUBCARRIERS or len(amplitudes) != NUM_SUBCARRIERS:
            return None

        return timestamp_ms, rssi, amplitudes
    except (ValueError, IndexError):
        return None


def collect_one(ser, label, output_dir):
    scenario_name = SCENARIOS[label]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"raw_label{label}_{timestamp}.csv"

    print("\n" + "=" * 58)
    print(f"Scenario {label}: {scenario_name}")
    print(f"Duration: {DURATION_SECONDS} seconds")
    input("Set up this scenario, then press ENTER to begin...")

    ser.reset_input_buffer()
    header = (
        ["timestamp_ms", "rssi", "label"]
        + [f"amp_{i}" for i in range(NUM_SUBCARRIERS)]
    )

    count = 0
    start = time.monotonic()
    with filename.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        while time.monotonic() - start < DURATION_SECONDS:
            raw = ser.readline().decode("utf-8", errors="ignore")
            parsed = parse_csi_line(raw)
            if parsed is None:
                continue

            timestamp_ms, rssi, amplitudes = parsed
            writer.writerow([timestamp_ms, rssi, label] + amplitudes)
            count += 1

            if count % 100 == 0:
                elapsed = int(time.monotonic() - start)
                print(f"  {count} samples collected ({elapsed}/{DURATION_SECONDS} sec)")

    print(f"Saved: {filename}")
    print(f"Valid samples: {count}")
    if count < MIN_SAMPLES:
        print(
            f"WARNING: fewer than {MIN_SAMPLES} samples. "
            "Tell the project team; check transmitter packets, Wi-Fi connection, "
            "and whether the receiver prints 64 amplitudes per packet."
        )
    else:
        print("Sample target reached.")


def main():
    parser = argparse.ArgumentParser(description="ESP32 Wi-Fi CSI data collector")
    parser.add_argument("--port", help="Serial port, e.g. /dev/cu.usbserial-XXXX")
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports")
    default_output = Path(__file__).resolve().parents[1] / "data" / "raw"
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Folder for CSV files (default: project data/raw)",
    )
    args = parser.parse_args()

    if args.list_ports:
        show_ports()
        return

    if not args.port:
        parser.error("Specify --port, or use --list-ports to find the ESP32 port.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Opening serial port...")
    try:
        with serial.Serial(args.port, BAUD, timeout=1) as ser:
            print(f"Connected to {args.port} at {BAUD} baud.")
            print("Keep this terminal open. Do not open ESP-IDF serial monitor at the same time.")
            for label in SCENARIOS:
                collect_one(ser, label, output_dir)
    except serial.SerialException as exc:
        print(f"Could not open serial port: {exc}")
        print("Check the port name, USB cable, and that no other app is using the port.")
        return

    print("\nAll five collection sessions finished.")
    print(f"CSV files are in: {output_dir.resolve()}")
    print("Check that every CSV has at least 500 data rows before sending to Member 3.")


if __name__ == "__main__":
    main()
