# =========================================================================
#  1-Click Cloud Deployment Assistant for IoT EdgeGuard
#  Pushes code to GitHub or Hugging Face Spaces for 24/7 Online Hosting
# =========================================================================

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
if (!$Root) { $Root = Get-Location }
Set-Location $Root

Clear-Host
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "   IOT EDGEGUARD: 24/7 PERMANENT CLOUD DEPLOYMENT ASSISTANT              " -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This assistant deploys your project to a 24/7 Cloud Server so that your  " -ForegroundColor White
Write-Host "dashboard stays online forever, even when your laptop is turned OFF.     " -ForegroundColor White
Write-Host ""

# Find Git executable
$gitCmd = "git"
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    if (Test-Path "C:\Program Files\Git\cmd\git.exe") {
        $gitCmd = "C:\Program Files\Git\cmd\git.exe"
    } elseif (Test-Path "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe") {
        $gitCmd = "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe"
    } else {
        Write-Host "[!] Git command not detected in current PATH." -ForegroundColor Yellow
        Write-Host "    Tip: You can upload 'cloud_deployment_bundle.zip' directly to Hugging Face" -ForegroundColor Cyan
        Write-Host "    in your web browser without needing to install Git at all!" -ForegroundColor Cyan
        Write-Host ""
    }
}

Write-Host "Choose where you want to deploy for 24/7 hosting:" -ForegroundColor Yellow
Write-Host "  [1] Hugging Face Spaces (100% Free Forever, No Credit Card, Recommended)" -ForegroundColor Green
Write-Host "  [2] GitHub Repository   (To connect to Render.com or Railway.app)" -ForegroundColor Cyan
Write-Host "  [3] Open Pre-Packaged 2.55 MB Zip Archive (For manual drag-and-drop upload)" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Enter choice (1, 2, or 3)"

if ($choice -eq "3") {
    $zipPath = Join-Path $Root "cloud_deployment_bundle.zip"
    if (Test-Path $zipPath) {
        Write-Host "`n[OK] Opening folder containing cloud_deployment_bundle.zip..." -ForegroundColor Green
        explorer.exe /select,"$zipPath"
    }
    exit
}

$repoUrl = ""
if ($choice -eq "1") {
    Write-Host "`n--- Hugging Face Spaces Deployment ---" -ForegroundColor Green
    Write-Host "1. Go to https://huggingface.co/new-space" -ForegroundColor DarkCyan
    Write-Host "2. Create a Space with SDK = 'Docker (Blank)'" -ForegroundColor DarkCyan
    Write-Host "3. Copy the Git repository URL from the Space page." -ForegroundColor DarkCyan
    Write-Host "   (Format: https://huggingface.co/spaces/YOUR_USERNAME/iot-edgeguard)`n" -ForegroundColor DarkGray
    $repoUrl = Read-Host "Paste your Hugging Face Space Git URL"
} elseif ($choice -eq "2") {
    Write-Host "`n--- GitHub Repository Deployment ---" -ForegroundColor Cyan
    Write-Host "1. Go to https://github.com/new" -ForegroundColor DarkCyan
    Write-Host "2. Create a repository named 'iot-edgeguard'" -ForegroundColor DarkCyan
    Write-Host "3. Copy your repository HTTPS URL." -ForegroundColor DarkCyan
    Write-Host "   (Format: https://github.com/YOUR_USERNAME/iot-edgeguard.git)`n" -ForegroundColor DarkGray
    $repoUrl = Read-Host "Paste your GitHub Repository URL"
}

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "[Error] No URL provided. Aborted." -ForegroundColor Red
    pause
    exit
}

if (!(Get-Command $gitCmd -ErrorAction SilentlyContinue) -and !(Test-Path $gitCmd)) {
    Write-Host "`n[!] Git is not installed locally on this machine." -ForegroundColor Yellow
    Write-Host "    You don't need Git! You can upload 'cloud_deployment_bundle.zip' directly" -ForegroundColor Green
    Write-Host "    via the web browser in 1 minute." -ForegroundColor Green
    $zipPath = Join-Path $Root "cloud_deployment_bundle.zip"
    if (Test-Path $zipPath) { explorer.exe /select,"$zipPath" }
    Write-Host "`nOpening the zip archive folder now. Drag the files into your browser!" -ForegroundColor Cyan
    pause
    exit
}

Write-Host "`n[1/4] Initializing local git repository..." -ForegroundColor Yellow
& $gitCmd init
& $gitCmd branch -M main

Write-Host "[2/4] Staging files (excluding binaries & logs)..." -ForegroundColor Yellow
& $gitCmd add .

Write-Host "[3/4] Committing project code..." -ForegroundColor Yellow
& $gitCmd commit -m "IoT EdgeGuard 24/7 Cloud Deployment with Docker and TinyML"

Write-Host "[4/4] Setting remote origin to: $repoUrl" -ForegroundColor Yellow
& $gitCmd remote remove origin 2>$null
& $gitCmd remote add origin $repoUrl

Write-Host "`nPushing to remote repository (a browser/login prompt may appear)..." -ForegroundColor Cyan
& $gitCmd push -u origin main --force

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=========================================================================" -ForegroundColor Green
    Write-Host "  SUCCESS! PROJECT DEPLOYED TO CLOUD REPOSITORY                           " -ForegroundColor Green
    Write-Host "=========================================================================" -ForegroundColor Green
    if ($choice -eq "1") {
        Write-Host "Your Hugging Face Space is now building! Visit your Space page to watch   " -ForegroundColor White
        Write-Host "the build progress. In 2 minutes, your dashboard will be online 24/7.     " -ForegroundColor White
    } else {
        Write-Host "Next step: Go to https://render.com -> New Web Service -> Select this repo." -ForegroundColor White
        Write-Host "Render will build the Docker container and give you a permanent 24/7 link." -ForegroundColor White
    }
} else {
    Write-Host "`n[Notice] If the push was rejected, check that your authentication is valid." -ForegroundColor Yellow
}

Write-Host ""
pause

