@echo off
echo =========================================================================
echo   DEPLOYING IOT ON-DEVICE FEDERATED ANOMALY DETECTION ENGINE (WINDOWS)
echo =========================================================================

echo [1/3] Verifying Python Environment and Dependencies...
py -m pip install -r backend/requirements.txt --quiet

echo [2/3] Exporting Embedded Microcontroller and TinyML Edge Models...
py deploy/export_edge_models.py

echo [3/3] Starting Production Server and Web Application...
start "" "http://127.0.0.1:8080"
py backend/server.py

pause

