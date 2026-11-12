# =========================================================================
#  EdgeGuard 24/7 Persistent Supervisor & Auto-Healing Tunnel Runner
#  Keeps Python Web Server and Cloudflare Tunnel running indefinitely.
#  Auto-restarts on crash, prevents Windows sleep, updates public_url.txt.
# =========================================================================

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  IOT EDGEGUARD: 24/7 CONTINUOUS RUNNER & AUTO-RECOVERY SUPERVISOR        " -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "[Info] Working Directory: $Root" -ForegroundColor DarkGray

# Prevent Windows from entering Idle Sleep while this runner is active
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class Sleeper {
    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
}
"@
[Sleeper]::SetThreadExecutionState([Sleeper]::ES_CONTINUOUS -bor [Sleeper]::ES_SYSTEM_REQUIRED) | Out-Null
Write-Host "[Power] Sleep prevention activated: PC will stay awake for 24/7 service." -ForegroundColor Green

$ServerProc = $null
$TunnelProc = $null
$LogDir = Join-Path $Root "results"
if (!(Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$TunnelLog = Join-Path $LogDir "tunnel_daemon.log"
$UrlFile = Join-Path $LogDir "public_url.txt"

function Start-ServerProcess {
    Write-Host "[Watchdog] Launching IoT Web Server (py -u app/server.py)..." -ForegroundColor Yellow
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = "py"
    $pinfo.Arguments = "-u app/server.py"
    $pinfo.WorkingDirectory = $Root
    $pinfo.UseShellExecute = $false
    $pinfo.CreateNoWindow = $true
    return [System.Diagnostics.Process]::Start($pinfo)
}

function Start-TunnelProcess {
    Write-Host "[Watchdog] Launching Cloudflare HTTPS Tunnel..." -ForegroundColor Yellow
    if (Test-Path $TunnelLog) { Remove-Item $TunnelLog -Force -ErrorAction SilentlyContinue }
    
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = Join-Path $Root "cloudflared.exe"
    $pinfo.Arguments = "tunnel --url http://127.0.0.1:8080 --logfile `"$TunnelLog`""
    $pinfo.WorkingDirectory = $Root
    $pinfo.UseShellExecute = $false
    $pinfo.CreateNoWindow = $true
    return [System.Diagnostics.Process]::Start($pinfo)
}

function Find-PublicUrl {
    if (Test-Path $TunnelLog) {
        $content = Get-Content $TunnelLog -ErrorAction SilentlyContinue
        foreach ($line in $content) {
            if ($line -match "https://[a-zA-Z0-9-]+\.trycloudflare\.com") {
                return $matches[0]
            }
        }
    }
    return $null
}

# Initial Startup
$ServerProc = Start-ServerProcess
Start-Sleep -Seconds 3
$TunnelProc = Start-TunnelProcess

Write-Host "[Watchdog] Waiting for Cloudflare DNS propagation..." -ForegroundColor Yellow
$foundUrl = $null
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 1
    $foundUrl = Find-PublicUrl
    if ($foundUrl) { break }
}

if ($foundUrl) {
    Write-Host ""
    Write-Host "=========================================================================" -ForegroundColor Green
    Write-Host "  ACTIVE PUBLIC HTTPS URL (SHAREABLE TO ANY DEVICE WORLDWIDE):           " -ForegroundColor Green
    Write-Host "  $foundUrl" -ForegroundColor White -BackgroundColor DarkGreen
    Write-Host "=========================================================================" -ForegroundColor Green
    Write-Host "  Local LAN:  http://192.168.0.105:8080" -ForegroundColor Cyan
    Write-Host "  Localhost:  http://127.0.0.1:8080" -ForegroundColor Cyan
    Write-Host "=========================================================================" -ForegroundColor Green
    
    @"
Public HTTPS URL:
$foundUrl

Local LAN URL (for devices on the same Wi-Fi / Hotspot):
http://192.168.0.105:8080

Localhost URL:
http://127.0.0.1:8080
"@ | Out-File -FilePath $UrlFile -Encoding utf8
} else {
    Write-Host "[Notice] Tunnel starting in background. Checking log..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "[Supervisor] Monitoring server and tunnel health. Press Ctrl+C to stop." -ForegroundColor DarkCyan

# Infinite watchdog loop
try {
    while ($true) {
        Start-Sleep -Seconds 5

        # Check Python server
        if ($null -eq $ServerProc -or $ServerProc.HasExited) {
            Write-Host "[Watchdog] Python server terminated unexpectedly. Auto-restarting..." -ForegroundColor Red
            $ServerProc = Start-ServerProcess
        }

        # Check Tunnel
        if ($null -eq $TunnelProc -or $TunnelProc.HasExited) {
            Write-Host "[Watchdog] Cloudflare Tunnel disconnected. Auto-reconnecting..." -ForegroundColor Red
            $TunnelProc = Start-TunnelProcess
            Start-Sleep -Seconds 4
            $newUrl = Find-PublicUrl
            if ($newUrl) {
                Write-Host "[Watchdog] Reconnected! New URL: $newUrl" -ForegroundColor Green
            }
        }
    }
}
finally {
    Write-Host "[Supervisor] Stopping background processes..." -ForegroundColor Yellow
    if ($ServerProc -and !$ServerProc.HasExited) { $ServerProc.Kill() }
    if ($TunnelProc -and !$TunnelProc.HasExited) { $TunnelProc.Kill() }
    [Sleeper]::SetThreadExecutionState([Sleeper]::ES_CONTINUOUS) | Out-Null
    Write-Host "[Supervisor] Clean exit completed." -ForegroundColor Green
}

