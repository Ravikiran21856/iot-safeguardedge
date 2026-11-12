@echo off
title IoT EdgeGuard 24x7 Cloud Deployment Assistant
color 0b
echo =========================================================================
echo   LAUNCHING 24x7 CLOUD DEPLOYMENT ASSISTANT (HUGGING FACE / GITHUB)
echo =========================================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\push_to_cloud.ps1"
pause

