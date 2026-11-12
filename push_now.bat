@echo off
title Push IoT EdgeGuard to GitHub
color 0a
echo =========================================================================
echo   PUSHING ALL 55 FILES TO GITHUB (Ravikiran21856/iot-safeguardedge)
echo =========================================================================
echo.
set "PATH=C:\Users\RAVI KIRAN\AppData\Local\Programs\MinGit\cmd;%PATH%"
cd /d "c:\Users\RAVI KIRAN\OneDrive\Desktop\IoT SECURITY"

echo Connecting to https://github.com/Ravikiran21856/iot-safeguardedge.git ...
git remote remove origin 2>nul
git remote add origin https://github.com/Ravikiran21856/iot-safeguardedge.git

echo.
echo Executing Git Push... (A GitHub sign-in browser tab will open if needed)
git push -u origin main

echo.
echo =========================================================================
echo   PUSH PROCESS FINISHED!
echo =========================================================================
pause

