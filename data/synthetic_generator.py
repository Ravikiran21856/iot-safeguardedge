"""
Synthetic IIoT and Smart Home Multivariate Sensor Generator.
Simulates realistic continuous telemetry for edge devices across diverse operational profiles:
- CNC Milling Machines (High RPM, periodic cutting tool load)
- Hydraulic Stamping Presses (High pressure pulses, transient current peaks)
- Continuous Induction Motors / Conveyors (Harmonic vibration, thermal buildup)
- Smart Home HVAC Systems (Cycling compressor, ambient thermal dynamics)
- Smart Solar Inverter / Storage (Voltage fluctuations, diurnal solar cycles)

Injects realistic physical failure modes and cyber-physical tampering anomalies:
- Bearing mechanical wear & catastrophic friction
- Thermal runaway & cooling fan failure
- Hydraulic seal leak & pressure loss
- Motor rotor lock / stall current surge
- Electrical bus transient spikes / undervoltage
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class SensorProfile:
    """Defines physical baseline operating ranges for a specific IoT client."""
    def __init__(
        self,
        name: str,
        category: str,
        base_vibration: float,
        vibration_freq: float,
        base_temp: float,
        base_rpm: float,
        base_current: float,
        base_pressure: float,
        base_acoustic: float,
        base_humidity: float,
        base_voltage: float,
    ):
        self.name = name
        self.category = category  # 'IIoT_Manufacturing' or 'Smart_Home'
        self.base_vibration = base_vibration
        self.vibration_freq = vibration_freq
        self.base_temp = base_temp
        self.base_rpm = base_rpm
        self.base_current = base_current
        self.base_pressure = base_pressure
        self.base_acoustic = base_acoustic
        self.base_humidity = base_humidity
        self.base_voltage = base_voltage


DEFAULT_PROFILES = [
    SensorProfile(
        name="CNC_Milling_Center_01",
        category="IIoT_Manufacturing",
        base_vibration=1.8,
        vibration_freq=25.0,
        base_temp=58.0,
        base_rpm=3600.0,
        base_current=14.5,
        base_pressure=6.0,
        base_acoustic=68.0,
        base_humidity=42.0,
        base_voltage=400.0,
    ),
    SensorProfile(
        name="Hydraulic_Stamping_Press_02",
        category="IIoT_Manufacturing",
        base_vibration=3.2,
        vibration_freq=8.0,
        base_temp=65.0,
        base_rpm=600.0,
        base_current=28.0,
        base_pressure=180.0,
        base_acoustic=74.0,
        base_humidity=45.0,
        base_voltage=400.0,
    ),
    SensorProfile(
        name="Conveyor_Induction_Motor_03",
        category="IIoT_Manufacturing",
        base_vibration=1.2,
        vibration_freq=15.0,
        base_temp=52.0,
        base_rpm=1450.0,
        base_current=8.5,
        base_pressure=2.5,
        base_acoustic=59.0,
        base_humidity=48.0,
        base_voltage=400.0,
    ),
    SensorProfile(
        name="SmartHome_HVAC_HeatPump_04",
        category="Smart_Home",
        base_vibration=0.6,
        vibration_freq=5.0,
        base_temp=38.0,
        base_rpm=1200.0,
        base_current=6.2,
        base_pressure=22.0,
        base_acoustic=48.0,
        base_humidity=52.0,
        base_voltage=230.0,
    ),
    SensorProfile(
        name="SmartHome_Solar_Inverter_05",
        category="Smart_Home",
        base_vibration=0.2,
        vibration_freq=2.0,
        base_temp=44.0,
        base_rpm=0.0,  # Static solid-state inverter
        base_current=12.0,
        base_pressure=1.0,
        base_acoustic=38.0,
        base_humidity=50.0,
        base_voltage=230.0,
    ),
]

SENSOR_NAMES = [
    "vibration_m_s2",
    "temperature_c",
    "rotational_speed_rpm",
    "current_amperes",
    "pressure_bar",
    "acoustic_emission_db",
    "humidity_pct",
    "voltage_volts",
]


class SensorDataGenerator:
    """Generates continuous multivariate sensor streams with realistic physical dynamics and anomalies."""

    def __init__(self, profiles: Optional[List[SensorProfile]] = None, seed: int = 42):
        self.profiles = profiles or DEFAULT_PROFILES
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_normal_stream(
        self,
        profile: SensorProfile,
        n_samples: int,
        sample_rate_hz: float = 1.0,
    ) -> np.ndarray:
        """Generates healthy baseline operational time-series data."""
        t = np.arange(n_samples) / sample_rate_hz
        
        # 1. Vibration: Multi-harmonic sinusoidal with random perturbations
        vib = (
            profile.base_vibration
            + 0.25 * np.sin(2 * np.pi * profile.vibration_freq * 0.05 * t)
            + 0.10 * np.cos(2 * np.pi * profile.vibration_freq * 0.12 * t)
            + self.rng.normal(0, 0.05, n_samples)
        )
        vib = np.maximum(0.01, vib)

        # 2. Temperature: Diurnal cycle + operational heating drift + Gaussian noise
        temp_drift = 1.5 * np.sin(2 * np.pi * (t / (n_samples * 0.6)))
        temp = profile.base_temp + temp_drift + self.rng.normal(0, 0.3, n_samples)

        # 3. Rotational Speed (RPM): Dynamic load variations
        if profile.base_rpm > 0:
            rpm = (
                profile.base_rpm
                + 0.02 * profile.base_rpm * np.sin(0.1 * t)
                + self.rng.normal(0, 0.005 * profile.base_rpm, n_samples)
            )
        else:
            rpm = np.zeros(n_samples)

        # 4. Current: Proportional to mechanical load + switching ripple
        current = (
            profile.base_current
            + 0.08 * profile.base_current * np.sin(0.15 * t + 0.4)
            + self.rng.normal(0, 0.04 * profile.base_current, n_samples)
        )

        # 5. Pressure: Hydraulic or coolant pressure
        pressure = (
            profile.base_pressure
            + 0.05 * profile.base_pressure * np.sin(0.08 * t)
            + self.rng.normal(0, 0.02 * profile.base_pressure, n_samples)
        )

        # 6. Acoustic Emission: High-frequency operational hum
        acoustic = (
            profile.base_acoustic
            + 0.8 * np.sin(0.2 * t)
            + self.rng.normal(0, 0.6, n_samples)
        )

        # 7. Ambient Humidity
        humidity = (
            profile.base_humidity
            + 3.0 * np.cos(2 * np.pi * (t / (n_samples * 0.8)))
            + self.rng.normal(0, 0.4, n_samples)
        )

        # 8. Voltage: Grid regulation fluctuations
        voltage = (
            profile.base_voltage
            + 1.8 * np.sin(0.05 * t)
            + self.rng.normal(0, 0.8, n_samples)
        )

        data = np.stack(
            [vib, temp, rpm, current, pressure, acoustic, humidity, voltage],
            axis=1,
        )
        return data

    def inject_anomalies(
        self,
        normal_data: np.ndarray,
        anomaly_ratio: float = 0.08,
        min_duration: int = 12,
        max_duration: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Injects realistic multi-sensor anomalous events into normal telemetry.
        Returns:
            corrupted_data: np.ndarray [n_samples, n_features]
            labels: np.ndarray [n_samples] (0 = normal, 1 = anomalous)
        """
        n_samples, n_features = normal_data.shape
        data = normal_data.copy()
        labels = np.zeros(n_samples, dtype=np.int32)

        target_anomaly_samples = int(n_samples * anomaly_ratio)
        current_anomaly_samples = 0

        # Available anomaly signatures
        anomaly_types = [
            "bearing_wear_vibration_spike",
            "thermal_runaway",
            "pressure_drop_leak",
            "motor_stall_current_surge",
            "electrical_transient_surge",
            "acoustic_cavitation_noise",
        ]

        while current_anomaly_samples < target_anomaly_samples:
            duration = int(self.rng.integers(min_duration, max_duration))
            if duration + current_anomaly_samples > target_anomaly_samples + 20:
                duration = max(min_duration, target_anomaly_samples - current_anomaly_samples)

            # Pick a random start location (avoid overlapping existing anomalies)
            max_start = n_samples - duration - 10
            if max_start <= 10:
                break
            start_idx = int(self.rng.integers(10, max_start))

            # Skip if already labeled as anomalous in that region
            if np.any(labels[start_idx : start_idx + duration] == 1):
                continue

            atype = self.rng.choice(anomaly_types)
            t_slice = slice(start_idx, start_idx + duration)
            ramp = np.linspace(0.5, 2.5, duration)

            if atype == "bearing_wear_vibration_spike":
                # High vibration amplitude escalation + acoustic spike
                data[t_slice, 0] += data[t_slice, 0] * (2.2 * ramp) + self.rng.normal(0, 0.8, duration)
                data[t_slice, 5] += 18.0 * ramp

            elif atype == "thermal_runaway":
                # Uncontrolled temperature rise + current increase
                temp_delta = np.linspace(5.0, 35.0, duration)
                data[t_slice, 1] += temp_delta
                data[t_slice, 3] += data[t_slice, 3] * 0.4

            elif atype == "pressure_drop_leak":
                # Hydraulic rupture / cavitation: sharp pressure drop
                data[t_slice, 4] = np.maximum(0.2, data[t_slice, 4] * np.linspace(0.8, 0.15, duration))
                data[t_slice, 5] += 12.0  # acoustic hissing

            elif atype == "motor_stall_current_surge":
                # Motor lock: RPM collapses to zero, current surges 3-5x
                data[t_slice, 2] = np.maximum(0.0, data[t_slice, 2] * np.linspace(0.9, 0.0, duration))
                data[t_slice, 3] += data[t_slice, 3] * 3.5
                data[t_slice, 0] += 1.5 * np.sin(np.arange(duration))

            elif atype == "electrical_transient_surge":
                # Grid surge / high voltage spikes & harmonic distortion
                data[t_slice, 7] += self.rng.choice([-45.0, 60.0]) * np.exp(-np.linspace(0, 3, duration))
                data[t_slice, 3] += self.rng.normal(0, 2.5, duration)

            elif atype == "acoustic_cavitation_noise":
                # High frequency ultrasonic chatter and erratic pressure flutter
                data[t_slice, 5] += 25.0 + self.rng.normal(0, 4.0, duration)
                data[t_slice, 4] += self.rng.normal(0, data[t_slice, 4].mean() * 0.3, duration)

            labels[t_slice] = 1
            current_anomaly_samples += duration

        return data, labels

    def generate_federated_dataset(
        self,
        n_samples_per_client: int = 3500,
        train_ratio: float = 0.70,
        val_ratio: float = 0.10,
        test_anomaly_ratio: float = 0.09,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Creates federated partition for each registered IoT client.
        - Train set: 100% normal (unsupervised reconstruction baseline)
        - Val set: 100% normal (used for setting dynamic anomaly threshold)
        - Test set: Normal + injected realistic anomalies with ground truth labels
        """
        federated_data = {}
        for idx, profile in enumerate(self.profiles):
            client_id = f"client_{idx:02d}_{profile.name}"
            
            # Generate continuous baseline stream
            total_needed = int(n_samples_per_client * 1.3)
            raw_stream = self.generate_normal_stream(profile, n_samples=total_needed)

            n_train = int(n_samples_per_client * train_ratio)
            n_val = int(n_samples_per_client * val_ratio)
            n_test = n_samples_per_client - n_train - n_val

            train_raw = raw_stream[:n_train]
            val_raw = raw_stream[n_train : n_train + n_val]
            test_raw = raw_stream[n_train + n_val : n_train + n_val + n_test]

            # In unsupervised anomaly detection, training and validation represent baseline nominal operations
            train_labels = np.zeros(n_train, dtype=np.int32)
            val_labels = np.zeros(n_val, dtype=np.int32)

            # Test set receives realistic physical and cyber anomalies
            test_data, test_labels = self.inject_anomalies(
                test_raw, anomaly_ratio=test_anomaly_ratio
            )

            federated_data[client_id] = {
                "profile": profile,
                "train_data": train_raw,
                "train_labels": train_labels,
                "val_data": val_raw,
                "val_labels": val_labels,
                "test_data": test_data,
                "test_labels": test_labels,
            }

        return federated_data

