@echo off
title IoT EdgeGuard 24x7 Server Runner
color 0b
echo =========================================================================
echo   STARTING IOT EDGEGUARD 24x7 PERSISTENT SERVER AND HTTPS TUNNEL
echo =========================================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\run_forever.ps1"
pause

