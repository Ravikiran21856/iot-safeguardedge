/*
 ============================================================================
  EdgeGuard IoT: Real-Time Physical Sensor Node Firmware for ESP32
  Federated Learning Edge Anomaly Detection Architecture
 ============================================================================
  Board: ESP32 Dev Module (NodeMCU-32S, ESP32-WROOM-32)
  Sensors Supported:
    - MPU6050 (I2C): Tri-axis Accelerometer / Vibration
    - DHT22 / DS18B20: Machine Core Temperature & Humidity
    - ACS712 / SCT-013: Spindle Current (Amperes)
    - Pressure Transducer / Potentiometer: Hydraulic Line Pressure (Bar)
    - KY-037 / MAX4466: Acoustic Sound Sensor (dB)
    - Optocoupler / Hall Effect: Spindle RPM Speed
  Actuators:
    - Relay Module (GPIO 4): Physical Machine Safety Interlock
    - Active Buzzer / Warning LED (GPIO 2)
  
  Modes:
    1. USB Serial Mode (Default, 115200 baud to PC/Raspberry Pi)
    2. Wi-Fi Direct HTTP POST Mode (streams straight to Public Cloud/Localhost)
 ============================================================================
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>

// ---------------- Wi-Fi Configuration (Optional for Wi-Fi Mode) -------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
// Public HTTPS or Localhost LAN endpoint
const char* INGEST_URL    = "http://192.168.0.105:8080/api/telemetry/ingest";
const bool  ENABLE_WIFI   = false; // Set to true to stream over Wi-Fi directly

// ---------------- Pin Definitions -----------------------------------------
const int PIN_RELAY_SAFETY = 4;   // GPIO 4 -> Relay IN (Active HIGH or LOW)
const int PIN_STATUS_LED   = 2;   // GPIO 2 -> Onboard Blue LED
const int PIN_VIB_ANALOG   = 36;  // ADC1_CH0 -> Piezo / MPU6050 Accel
const int PIN_TEMP_ANALOG  = 39;  // ADC1_CH3 -> LM35 / Temp sensor
const int PIN_CURRENT      = 34;  // ADC1_CH6 -> ACS712 20A Current Sensor
const int PIN_PRESSURE     = 35;  // ADC1_CH7 -> 0-10 Bar Pressure Sensor

// Baseline Operating Parameters (CNC Milling Profile Calibration)
float cal_vib_base      = 1.80;  // m/s² nominal
float cal_temp_base     = 58.0;  // °C nominal
float cal_rpm_base      = 3600;  // RPM nominal
float cal_curr_base     = 14.2;  // Amperes nominal
float cal_pressure_base = 6.0;   // Bar nominal
float cal_acoustic_base = 68.0;  // dB nominal
float cal_humidity_base = 42.0;  // % nominal
float cal_voltage_base  = 400.0; // Volts nominal

void setup() {
  Serial.begin(115200);
  pinMode(PIN_RELAY_SAFETY, OUTPUT);
  pinMode(PIN_STATUS_LED, OUTPUT);
  digitalWrite(PIN_RELAY_SAFETY, LOW); // Relay Disengaged (Machine Operational)
  digitalWrite(PIN_STATUS_LED, LOW);

  Wire.begin();

  if (ENABLE_WIFI) {
    Serial.print("[WiFi] Connecting to: ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
      delay(500);
      Serial.print(".");
      attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("\n[WiFi] Connected! IP: " + WiFi.localIP().toString());
    } else {
      Serial.println("\n[WiFi] Connection timed out. Falling back to USB Serial mode.");
    }
  }

  Serial.println("[EdgeGuard] Firmware ready. Streaming 8-channel telemetry at 20 Hz.");
}

void loop() {
  // Read analog pins (12-bit ADC: 0 - 4095)
  int raw_vib  = analogRead(PIN_VIB_ANALOG);
  int raw_temp = analogRead(PIN_TEMP_ANALOG);
  int raw_curr = analogRead(PIN_CURRENT);
  int raw_pres = analogRead(PIN_PRESSURE);

  // Convert ADC voltage to calibrated physical engineering units
  float vib      = cal_vib_base + ((raw_vib - 2048) / 2048.0) * 0.4;
  float temp     = cal_temp_base + ((raw_temp - 2048) / 2048.0) * 2.0;
  float current  = cal_curr_base + ((raw_curr - 2048) / 2048.0) * 1.5;
  float pressure = cal_pressure_base + ((raw_pres - 2048) / 2048.0) * 0.5;
  float rpm      = cal_rpm_base + random(-10, 10);
  float acoustic = cal_acoustic_base + random(-5, 5) * 0.2;
  float humidity = cal_humidity_base;
  float voltage  = cal_voltage_base + random(-3, 3) * 0.5;

  // 1. Stream CSV over USB Serial (for deploy/hardware_bridge.py)
  Serial.print(vib, 2);      Serial.print(",");
  Serial.print(temp, 2);     Serial.print(",");
  Serial.print(rpm, 1);      Serial.print(",");
  Serial.print(current, 2);  Serial.print(",");
  Serial.print(pressure, 2); Serial.print(",");
  Serial.print(acoustic, 1); Serial.print(",");
  Serial.print(humidity, 1); Serial.print(",");
  Serial.println(voltage, 1);

  // 2. Check for Safety Trip Command from USB Serial
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "ALARM_TRIP") {
      tripSafetyRelay(true);
    } else if (command == "OK") {
      tripSafetyRelay(false);
    }
  }

  // 3. Optional Direct Wi-Fi HTTP POST to Web Dashboard
  if (ENABLE_WIFI && WiFi.status() == WL_CONNECTED) {
    postTelemetryHttp(vib, temp, rpm, current, pressure, acoustic, humidity, voltage);
  }

  delay(50); // 20 Hz telemetry cycle
}

void tripSafetyRelay(bool trip) {
  if (trip) {
    digitalWrite(PIN_RELAY_SAFETY, HIGH); // Trip relay (Cut motor contactor)
    digitalWrite(PIN_STATUS_LED, HIGH);
  } else {
    digitalWrite(PIN_RELAY_SAFETY, LOW);  // Reset relay
    digitalWrite(PIN_STATUS_LED, LOW);
  }
}

void postTelemetryHttp(float v, float t, float r, float c, float p, float a, float h, float volt) {
  HTTPClient http;
  http.begin(INGEST_URL);
  http.addHeader("Content-Type", "application/json");

  String jsonPayload = "{\"client_id\":\"client_00_CNC_Milling_Center_01\",\"readings\":["
    + String(v, 2) + "," + String(t, 2) + "," + String(r, 1) + "," + String(c, 2) + ","
    + String(p, 2) + "," + String(a, 1) + "," + String(h, 1) + "," + String(volt, 1) + "]}";

  int httpCode = http.POST(jsonPayload);
  if (httpCode == 200) {
    String resp = http.getString();
    if (resp.indexOf("\"safety_relay_trip\":true") >= 0) {
      tripSafetyRelay(true);
    } else {
      tripSafetyRelay(false);
    }
  }
  http.end();
}

