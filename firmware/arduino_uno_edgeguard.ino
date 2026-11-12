/*
 ============================================================================
  EdgeGuard IoT: Physical Sensor Node Firmware for Arduino Uno / Nano
 ============================================================================
  Board: Arduino Uno R3 / Nano (ATmega328P)
  Streams CSV format at 115200 baud over USB:
  "vib,temp,rpm,current,pressure,acoustic,humidity,voltage\n"
 ============================================================================
*/

const int PIN_RELAY_SAFETY = 4; // Digital Pin 4 -> Relay Module IN
const int PIN_PIEZO_VIB    = A0;
const int PIN_TEMP_LM35    = A1;
const int PIN_CURRENT_ACS  = A2;
const int PIN_PRESSURE     = A3;

void setup() {
  Serial.begin(115200);
  pinMode(PIN_RELAY_SAFETY, OUTPUT);
  digitalWrite(PIN_RELAY_SAFETY, LOW); // Relay off (Normal Operation)
}

void loop() {
  // Read analog voltages (10-bit ADC: 0 - 1023)
  int a0 = analogRead(PIN_PIEZO_VIB);
  int a1 = analogRead(PIN_TEMP_LM35);
  int a2 = analogRead(PIN_CURRENT_ACS);
  int a3 = analogRead(PIN_PRESSURE);

  // Scaled physical engineering units
  float vib      = 1.80 + ((a0 - 512) / 512.0) * 0.35;
  float temp     = 58.0 + ((a1 - 512) / 512.0) * 1.8;
  float rpm      = 3600.0;
  float current  = 14.2 + ((a2 - 512) / 512.0) * 1.2;
  float pressure = 6.0 + ((a3 - 512) / 512.0) * 0.4;
  float acoustic = 68.0;
  float humidity = 42.0;
  float voltage  = 400.0;

  // Stream CSV string
  Serial.print(vib, 2);      Serial.print(F(","));
  Serial.print(temp, 2);     Serial.print(F(","));
  Serial.print(rpm, 1);      Serial.print(F(","));
  Serial.print(current, 2);  Serial.print(F(","));
  Serial.print(pressure, 2); Serial.print(F(","));
  Serial.print(acoustic, 1); Serial.print(F(","));
  Serial.print(humidity, 1); Serial.print(F(","));
  Serial.println(voltage, 1);

  // Check PC Safety Trip command
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "ALARM_TRIP") {
      digitalWrite(PIN_RELAY_SAFETY, HIGH); // Trip Relay
    } else if (cmd == "OK") {
      digitalWrite(PIN_RELAY_SAFETY, LOW);  // Reset Relay
    }
  }

  delay(50); // 20 Hz telemetry cycle
}

