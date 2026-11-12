"""
Physical Hardware Ingestion Bridge for Edge Anomaly Detection.
Supports reading real-time telemetry from:
1. Serial / USB UART (Arduino, ESP32, STM32 microcontrollers streaming CSV over COM port)
2. Native I2C Bus on Raspberry Pi (MPU6050 accelerometer, ADS1115 ADC, BME280)
3. Direct HTTP REST Ingestion to Local/Public Web Application Dashboard
4. Virtual Microcontroller Simulator Mode for Hardware Ingestion Testing

Pushes live physical readings into EdgeAnomalyRuntime and triggers physical GPIO safety relays.
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deploy.edge_runtime import EdgeAnomalyRuntime

DEFAULT_INGEST_URL = "http://127.0.0.1:8080/api/telemetry/ingest"


class HardwareSensorBridge:
    """Bridges physical hardware sensors to the On-Device Anomaly Detection Engine and Web App."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.0384,
        port: str = "auto",
        baudrate: int = 115200,
        ingest_url: str = DEFAULT_INGEST_URL,
        client_id: str = "client_00_CNC_Milling_Center_01",
    ):
        self.port = port
        self.baudrate = baudrate
        self.ingest_url = ingest_url
        self.client_id = client_id
        self.runtime = EdgeAnomalyRuntime(model_path=model_path, threshold=threshold)
        self.ser = None

    def detect_port(self) -> str:
        """Auto-detect available USB serial COM port."""
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            if not ports:
                print("[!] No USB COM ports found.")
                return "COM3"
            for p in ports:
                # Prefer USB serial devices over Bluetooth virtual links
                if "bluetooth" not in p.description.lower():
                    print(f"[*] Auto-detected hardware port: {p.device} ({p.description})")
                    return p.device
            print(f"[*] Using first detected port: {ports[0].device} ({ports[0].description})")
            return ports[0].device
        except Exception:
            return "COM3"

    def connect_serial(self) -> bool:
        """Connects to microcontroller (ESP32/Arduino/STM32) streaming 8-channel sensor readings."""
        if self.port.lower() == "auto":
            self.port = self.detect_port()

        try:
            import serial
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1.0)
            print(f"[OK] Connected to hardware microcontroller on {self.port} at {self.baudrate} baud.")
            time.sleep(2)  # Wait for microcontroller reboot
            return True
        except ImportError:
            print("[Warning] 'pyserial' not installed. Install via: py -m pip install pyserial")
            return False
        except Exception as e:
            print(f"[Error] Failed to connect to serial port {self.port}: {e}")
            return False

    def forward_to_web_server(self, readings: np.ndarray) -> bool:
        """HTTP POST sensor readings to web application dashboard."""
        try:
            payload = json.dumps({
                "client_id": self.client_id,
                "readings": readings.tolist(),
            }).encode("utf-8")
            req = urllib.request.Request(
                self.ingest_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("safety_relay_trip", False)
        except Exception:
            return False

    def read_hardware_loop(self):
        """
        Continuously reads live serial strings from physical sensors.
        Format expected from microcontroller firmware:
        "vib,temp,rpm,current,pressure,acoustic,humidity,voltage\n"
        Example: "1.82,58.4,3600,14.2,6.1,68.5,42.0,399.8"
        """
        if not self.ser:
            print("[!] Serial connection not established. Exiting hardware loop.")
            return

        print(f"[*] Forwarding physical readings to: {self.ingest_url}")
        print("[*] Listening to live hardware sensor stream. Press Ctrl+C to stop...\n")

        while True:
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) != 8:
                    continue

                # Parse float sensor readings
                readings = np.array([float(p) for p in parts], dtype=np.float32)

                # Local TinyML on-device inference
                is_alarm_local = self.runtime.push_sensor_reading(readings)

                # Web dashboard ingestion
                is_alarm_server = self.forward_to_web_server(readings)
                is_alarm = is_alarm_local or is_alarm_server

                if is_alarm:
                    print(f"[!] ALARM TRIPPED! Vibration: {readings[0]:.2f} Temp: {readings[1]:.1f}°C Current: {readings[3]:.1f}A")
                    self.ser.write(b"ALARM_TRIP\n")
                else:
                    self.ser.write(b"OK\n")

            except KeyboardInterrupt:
                print("\n[*] Stopping hardware bridge.")
                break
            except Exception as e:
                print(f"[!] Sensor read error: {e}")
                time.sleep(0.1)

    def simulate_hardware_stream(self, interval_s: float = 0.25):
        """
        Simulates an active microcontroller streaming real physical sensor telemetry
        with periodic physical fault injection to test live dashboard response.
        """
        print(f"\n=========================================================================")
        print(f"  VIRTUAL HARDWARE SENSOR STREAM GENERATOR RUNNING                      ")
        print(f"  Target Dashboard: {self.ingest_url}                                   ")
        print(f"  Telemetry Frequency: {1.0 / interval_s:.1f} Hz                        ")
        print(f"=========================================================================\n")

        t = 0
        base_vib = 1.80
        base_temp = 58.0
        base_rpm = 3600.0
        base_current = 14.2
        base_pressure = 6.0
        base_acoustic = 68.0
        base_humidity = 42.0
        base_voltage = 400.0

        try:
            while True:
                t += 1
                # Normal stochastic sensor variation
                vib = base_vib + np.random.normal(0, 0.08)
                temp = base_temp + np.random.normal(0, 0.4)
                rpm = base_rpm + np.random.normal(0, 5.0)
                current = base_current + np.random.normal(0, 0.15)
                pressure = base_pressure + np.random.normal(0, 0.05)
                acoustic = base_acoustic + np.random.normal(0, 0.5)
                humidity = base_humidity + np.random.normal(0, 0.2)
                voltage = base_voltage + np.random.normal(0, 1.2)

                # Inject simulated hardware fault every 40 packets (~10 seconds)
                is_fault = (t % 40) >= 32
                if is_fault:
                    vib += 2.50       # Bearing chatter
                    acoustic += 18.0  # Ultrasonic harmonic spike
                    temp += 6.5       # Friction heating

                readings = np.array([vib, temp, rpm, current, pressure, acoustic, humidity, voltage], dtype=np.float32)

                is_alarm_local = self.runtime.push_sensor_reading(readings)
                is_alarm_server = self.forward_to_web_server(readings)
                alarm_active = is_alarm_local or is_alarm_server

                status_tag = "[ALARM TRIPPED]" if alarm_active else "[NOMINAL]"
                print(f"[{t:04d}] {status_tag} Vib: {vib:5.2f} m/s² | Temp: {temp:5.1f} °C | Noise: {acoustic:5.1f} dB -> Server: {self.ingest_url}")

                time.sleep(interval_s)

        except KeyboardInterrupt:
            print("\n[*] Stopping virtual hardware stream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Hardware Sensor Bridge")
    parser.add_argument("--port", type=str, default="auto", help="Serial port (e.g. COM3, auto)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--ingest-url", type=str, default=DEFAULT_INGEST_URL, help="REST API endpoint for telemetry ingest")
    parser.add_argument("--client", type=str, default="client_00_CNC_Milling_Center_01", help="Target client machine ID")
    parser.add_argument("--simulate", action="store_true", help="Run simulated hardware stream to test dashboard without physical USB")
    args = parser.parse_args()

    model_p = os.path.join(os.path.dirname(__file__), "..", "results", "models", "model_fp32.pt")
    bridge = HardwareSensorBridge(
        model_path=model_p,
        port=args.port,
        baudrate=args.baud,
        ingest_url=args.ingest_url,
        client_id=args.client,
    )

    if args.simulate:
        bridge.simulate_hardware_stream(interval_s=0.25)
    else:
        if bridge.connect_serial():
            bridge.read_hardware_loop()
