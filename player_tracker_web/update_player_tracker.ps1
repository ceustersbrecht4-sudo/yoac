# Updates Player Tracker with the intro page demo, drop-zone feedback, live
# processing preview and results reveal, then restarts the app.
# Run from the player_tracker_web folder:
#   powershell -ExecutionPolicy Bypass -File .\update_player_tracker.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$base = "https://raw.githubusercontent.com/ceustersbrecht4-sudo/yoac/8cf4d225d26527dd25e6fb8f4a440a635b906111/player_tracker_web"
$files = @("app.py", "templates/index.html", "templates/landing.html", "templates/_cookie_banner.html")

if (-not (Test-Path (Join-Path $root "app.py"))) {
    Write-Host "app.py not found next to this script. Put it in the player_tracker_web folder." -ForegroundColor Red
    exit 1
}

# Stop the running app so the new version is the one that answers.
Stop-Process -Name python, py -Force -ErrorAction SilentlyContinue

# Download everything first; only replace files once all downloads worked.
$tmp = Join-Path $env:TEMP "player_tracker_update"
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
foreach ($f in $files) {
    $dest = Join-Path $tmp $f
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    Invoke-WebRequest "$base/$f" -OutFile $dest -UseBasicParsing
    if ((Get-Item $dest).Length -eq 0) {
        Write-Host "Download of $f came back empty - nothing changed." -ForegroundColor Red
        exit 1
    }
}

# Back up the current versions, then put the new ones in place.
$backup = Join-Path $root ("backup_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
foreach ($f in $files) {
    $target = Join-Path $root $f
    if (Test-Path $target) {
        $saved = Join-Path $backup $f
        New-Item -ItemType Directory -Force -Path (Split-Path $saved) | Out-Null
        Copy-Item $target $saved -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item (Join-Path $tmp $f) $target -Force
    Write-Host "OK  $f" -ForegroundColor Green
}
Write-Host "Old versions saved in $backup" -ForegroundColor Green

Write-Host ""
Write-Host "Starting Player Tracker... open http://127.0.0.1:5000 when it's ready." -ForegroundColor Cyan
Write-Host "Press Ctrl+C in this window to stop it." -ForegroundColor Cyan
Set-Location $root
py app.py
