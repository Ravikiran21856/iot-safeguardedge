#!/usr/bin/env bash
set -e

echo "========================================================================="
echo "  DEPLOYING IOT ON-DEVICE FEDERATED ANOMALY DETECTION ENGINE (LINUX)     "
echo "========================================================================="

echo "[1/3] Installing Dependencies..."
python3 -m pip install -r requirements.txt --quiet

echo "[2/3] Exporting Embedded Microcontroller and TinyML Edge Models..."
python3 deploy/export_edge_models.py

echo "[3/3] Launching Edge Production Service..."
python3 app/server.py

