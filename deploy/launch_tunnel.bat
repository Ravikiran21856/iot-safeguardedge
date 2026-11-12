@echo off
echo =========================================================================
echo   LAUNCHING PUBLIC HTTPS TUNNEL FOR IOT SECURITY DASHBOARD
echo =========================================================================

if not exist "cloudflared.exe" (
    echo Downloading Cloudflare Tunnel binary...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"
)

echo Starting tunnel to http://127.0.0.1:8080 ...
.\cloudflared.exe tunnel --url http://127.0.0.1:8080
pause

