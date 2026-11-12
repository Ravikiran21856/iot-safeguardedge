# Real-Time Physical Hardware Equipment Guide

This document details the exact **Hardware Bill of Materials (BOM)**, wiring pinouts, communication protocols, and assembly instructions required to deploy the **Lightweight Federated Anomaly Detection Engine** on real physical IoT/IIoT equipment.

---

## 1. Hardware Architecture Overview

```
 [ Physical Machine / Motor Rig ]
   |
   +--> MPU-6050 Accelerometer  ----(I2C: SDA/SCL)---+
   +--> DHT22 / BME280 Thermal  ----(Digital: GPIO)--+
   +--> ACS712 Current Sensor   ----(Analog: ADC)----+--> [ Edge Microcontroller / Gateway ]
   +--> LM393 Optical Tachometer ---(Pulse: GPIO)----+     (ESP32 / Raspberry Pi 4/5 / STM32)
   +--> Piezoresistive Pressure  ---(Analog: ADC)----+                 |
   +--> MAX4466 Microphone / Mic ---(Analog: ADC)----+        [ INT8 Quantized Autoencoder ]
                                                                       |
 [ 5V Safety Relay / Interlock ] <---(Trigger: GPIO4)------------------+
   (Shuts off motor power on anomaly trip)
```

---

## 2. Bill of Materials (BOM)

### A. Core Edge Compute Board (Choose One)

| Equipment | Model / Specs | Interface | Approx. Cost | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Edge Microcontroller (Option 1 - Recommended)** | **ESP32-S3 DevKit** (Dual-Core 240MHz, 8MB PSRAM, Wi-Fi/BLE) | USB-C, I2C, SPI, ADC | **\$6 - \$9** | On-device TinyML INT8 inference or high-speed sensor acquisition. |
| **Edge Gateway (Option 2)** | **Raspberry Pi 4 / 5** (2GB or 4GB RAM) | Linux OS, USB, GPIO, I2C | **\$45 - \$65** | Industrial edge gateway running full Python server + web dashboard. |
| **Industrial IPC (Option 3 - Factory Grade)** | **Siemens SIMATIC IOT2050** or **Advantech UNO-2271G** | RS-485 Modbus, DIN-Rail | **\$250 - \$450** | Factory shop floor deployment in industrial electrical cabinets. |

---

### B. Sensor Kit (Matches the 8 Multivariate Telemetry Channels)

| Channel # | Physical Parameter | Recommended Sensor Module | Protocol / Interface | Operating Voltage | Approx. Cost |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **Ch 0** | **Vibration** (m/s²) | **MPU-6050** or **ADXL345** (3-axis Digital Accelerometer) | I2C (`0x68`) | 3.3V - 5V | **\$2.50** |
| **Ch 1** | **Temperature** (°C) | **BME280** or **DHT22 (AM2302)** | I2C (`0x76`) or Digital 1-Wire | 3.3V - 5V | **\$3.00** |
| **Ch 2** | **Rotational Speed** (RPM) | **LM393 Optical Slot Sensor** or **A3144 Hall Effect Sensor** | Digital Pulse (Interrupt GPIO) | 3.3V - 5V | **\$1.50** |
| **Ch 3** | **Current** (Amperes) | **ACS712 (5A / 20A)** or **SCT-013 Split-Core Current Transformer** | Analog ADC (0 - 3.3V/5V) | 5V | **\$3.00** |
| **Ch 4** | **Pressure** (Bar) | **0 - 1.2 MPa (175 PSI) G1/4" Pressure Transducer** | Analog Voltage (0.5V - 4.5V) | 5V | **\$7.50** |
| **Ch 5** | **Acoustic / Ultrasonic** (dB) | **MAX4466 Electret Microphone** with adjustable gain | Analog Audio Envelope | 3.3V - 5V | **\$2.50** |
| **Ch 6** | **Humidity** (%) | Integrated in **BME280** / **DHT22** | I2C / 1-Wire | 3.3V | (Included) |
| **Ch 7** | **Voltage** (Volts) | **0 - 25V DC Voltage Divider Sensor** or **ZMPT101B AC Transformer** | Analog ADC | 3.3V - 5V | **\$2.00** |

---

### C. Safety Actuators & Machine Test Rig

| Equipment | Model / Specs | Purpose | Approx. Cost |
| :--- | :--- | :--- | :--- |
| **Safety Interlock Relay** | **5V 1-Channel Optocoupler Relay Module** (SRD-05VDC-SL-C) | Trips and cuts motor power instantly upon anomaly alarm | **\$1.50** |
| **Alarm Indicator** | **5V Active Buzzer + High-Brightness Red LED** | Audio/visual alarm for machine operators | **\$1.00** |
| **Mechanical Test Load** | **12V DC Hobby Motor with Speed Controller (PWM)** or 12V Cooling Fan | Generates real rotational vibrations, current load, and heat | **\$6.00** |
| **Analog-to-Digital Converter** (If using Raspberry Pi) | **ADS1115 16-Bit 4-Channel I2C ADC Module** | Translates analog pressure/current/voltage to digital I2C | **\$3.50** |
| **Prototyping Essentials** | Breadboard + 40-pin Male-to-Female & Male-to-Male Jumper Wires | Solderless wiring and component interconnects | **\$4.00** |

> **Total Hardware Investment (Maker/Lab Setup):** **$\approx \$35 - \$50$** (excluding Raspberry Pi) or **$\approx \$80 - \$100$** (complete standalone setup with Raspberry Pi).

---

## 3. Physical Wiring Guide (Pinout for ESP32 / Arduino / Raspberry Pi)

### ESP32-S3 Microcontroller Wiring

| Component | Component Pin | ESP32-S3 Pin | Notes |
| :--- | :--- | :--- | :--- |
| **MPU-6050** | VCC, GND, SDA, SCL | 3.3V, GND, **GPIO 21 (SDA)**, **GPIO 22 (SCL)** | I2C Accelerometer |
| **BME280** | VCC, GND, SDA, SCL | 3.3V, GND, **GPIO 21 (SDA)**, **GPIO 22 (SCL)** | Shares I2C bus (Address `0x76`) |
| **LM393 Optical RPM** | VCC, GND, DO | 3.3V, GND, **GPIO 19** | Digital interrupt pin counting rotations |
| **ACS712 Current Sensor** | VCC, GND, OUT | 5V, GND, **GPIO 34 (ADC1)** | Analog current measurement |
| **Pressure Transducer** | VCC, GND, Signal | 5V, GND, **GPIO 35 (ADC1)** | Analog hydraulic/pneumatic pressure |
| **MAX4466 Microphone** | VCC, GND, OUT | 3.3V, GND, **GPIO 36 (ADC1)** | Ultrasonic/acoustic emission level |
| **Voltage Sensor** | VCC, GND, Signal | 3.3V, GND, **GPIO 39 (ADC1)** | Bus voltage divider |
| **5V Relay Module** | VCC, GND, IN | 5V, GND, **GPIO 4 (OUTPUT)** | Controls motor power interlock |

---

## 4. How to Create Real Physical Anomalies in the Lab

To test the system live with physical hardware, you can inject real physical failure modes:

1. **Bearing Wear / Vibration Failure:**
   - Attach a small piece of blue-tac or a small screw/nut off-center to the motor shaft.
   - The unbalance immediately causes mechanical vibration harmonic spikes on the MPU-6050.
2. **Thermal Runaway:**
   - Gently warm the BME280/DHT22 with a hairdryer or block the fan intake.
   - The temperature will ramp past the nominal operating boundary.
3. **Motor Jam / Rotor Stall:**
   - Gently apply friction or press the motor shaft with a glove to stop rotation.
   - The RPM collapses to zero, and the ACS712 current sensor surges $3\text{--}4\times$.
4. **Hydraulic / Air Pressure Loss:**
   - Release a small pneumatic valve or pinch the input tubing connected to the pressure transducer.

---

## 5. Software-to-Hardware Execution

Connect your microcontroller via USB, then run:

```powershell
# 1. Install serial communication driver
py -m pip install pyserial

# 2. Start the live hardware ingestion bridge
py deploy/hardware_bridge.py --port COM3 --baud 115200
```

The bridge reads real hardware telemetry at $20\text{ Hz}$, executes the INT8 quantized model locally, and sends the `ALARM_TRIP` command back over serial to immediately activate the physical 5V relay module if an anomaly is detected!

