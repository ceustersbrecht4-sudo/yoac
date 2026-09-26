# Updates Player Tracker (login, intro page, upload page, two-team sorting)
# and restarts the app. Accounts (users.json, secret.key) are never touched.
# Run from the player_tracker_web folder:
#   powershell -ExecutionPolicy Bypass -File .\update_player_tracker.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$base = "https://raw.githubusercontent.com/ceustersbrecht4-sudo/yoac/64eecc6dc6ff34806b434f79c2fe41ab7f6e0d9c/player_tracker_web"
$files = @("app.py", "teams.py", "accounts.py", "security.py", "content_check.py", "quota.py", "compact.py", "tracker.py", "analysis.py", "pitch.py", "rugby.py", "README.md", "templates/_usage.html", "legal.py", "SECURITY.md", "requirements.txt", "templates/account.html", "templates/legal.html", "templates/login.html", "templates/index.html", "templates/landing.html", "templates/_cookie_banner.html",
           "static/brand/yoac-logo.png", "static/brand/yoac-wordmark.png", "static/brand/yoac-icon-64.png", "static/brand/yoac-icon-192.png",
           "static/brand/yoac-outline.png", "static/brand/yoac-stencil.png", "static/brand/yoac-y-icon.png", "static/brand/yoac-icon-italic-320.png",
           "static/fonts/barlow-condensed-800.woff2", "static/fonts/barlow-condensed-900-italic.woff2", "static/fonts/LICENSE-barlow-condensed.txt")

if (-not (Test-Path (Join-Path $root "app.py"))) {
    Write-Host "app.py not found next to this script. Put it in the player_tracker_web folder." -ForegroundColor Red
    exit 1
}

# Stop the running app so the new version is the one that answers.
Stop-Process -Name python, py -Force -ErrorAction SilentlyContinue

# Files the owner fills in: only added when they don't exist yet, never overwritten.
$ownerFiles = @("legal_info.json", "plans.json")
foreach ($f in $ownerFiles) {
    $target = Join-Path $root $f
    if (-not (Test-Path $target)) {
        Invoke-WebRequest "$base/$f" -OutFile $target -UseBasicParsing
        Write-Host "OK  $f (new - fill in your details with Notepad)" -ForegroundColor Green
    } else {
        Write-Host "OK  $f kept (your details are not overwritten)" -ForegroundColor Green
    }
}

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
# Optional package that draws the QR code for two-step login set-up.
# Optional packages: QR code for 2FA set-up, and the production web server.
py -m pip install --quiet --disable-pip-version-check "qrcode>=7.4" "waitress>=3.0"
py app.py
