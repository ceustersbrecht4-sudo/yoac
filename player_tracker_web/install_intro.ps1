# Installs the Player Tracker intro page + cookie banner, then starts the app.
# Run from the player_tracker_web folder:
#   powershell -ExecutionPolicy Bypass -File .\install_intro.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$appPy = Join-Path $root "app.py"
$templates = Join-Path $root "templates"
$utf8 = New-Object System.Text.UTF8Encoding($false)

if (-not (Test-Path $appPy)) {
    Write-Host "app.py not found next to this script. Put install_intro.ps1 in the player_tracker_web folder." -ForegroundColor Red
    exit 1
}
New-Item -ItemType Directory -Force -Path $templates | Out-Null

$landing = @'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Player Tracker</title>
  <style>
    :root {
      --bg: #0f1419;
      --surface: #1a2129;
      --text: #e8edf2;
      --muted: #9aa7b4;
      --accent: #2ecc71;
      --accent-dark: #27ae60;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px 16px;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      line-height: 1.6;
    }
    main {
      max-width: 640px;
      width: 100%;
      text-align: center;
    }
    h1 {
      font-size: clamp(2rem, 6vw, 3rem);
      margin: 0 0 16px;
    }
    .lead {
      font-size: 1.15rem;
      color: var(--muted);
      margin: 0 0 32px;
    }
    .steps {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 16px;
      margin: 0 0 40px;
      padding: 0;
      list-style: none;
      text-align: left;
    }
    .steps li {
      background: var(--surface);
      border-radius: 12px;
      padding: 16px;
    }
    .steps strong {
      display: block;
      color: var(--accent);
      margin-bottom: 4px;
    }
    .cta {
      display: inline-block;
      background: var(--accent);
      color: #0f1419;
      font-weight: 700;
      font-size: 1.1rem;
      text-decoration: none;
      padding: 14px 40px;
      border-radius: 999px;
      transition: background 0.15s;
    }
    .cta:hover, .cta:focus-visible { background: var(--accent-dark); }
  </style>
</head>
<body>
  <main>
    <h1>Player Tracker</h1>
    <p class="lead">
      Upload your match footage and Player Tracker finds every player on the pitch,
      follows them through the whole video, and groups them by their team colours.
      You get the footage back with each player tracked and labelled, so you can
      review positioning and movement without tagging anything by hand.
    </p>

    <ol class="steps">
      <li><strong>1. Upload</strong>Choose a match video from your computer.</li>
      <li><strong>2. Track</strong>Players are detected and followed frame by frame.</li>
      <li><strong>3. Review</strong>Watch the result with players marked in team colours.</li>
    </ol>

    <a class="cta" href="{{ url_for('index') }}">Get Started</a>
  </main>

  {% include "_cookie_banner.html" %}
</body>
</html>
'@

$banner = @'
<div id="cookie-banner" role="region" aria-label="Cookie consent" hidden>
  <p>
    We use cookies to keep the site working and to remember your preferences.
  </p>
  <div class="cookie-actions">
    <button type="button" data-consent="declined">Decline</button>
    <button type="button" data-consent="accepted" class="cookie-accept">Accept</button>
  </div>
</div>

<style>
  #cookie-banner {
    position: fixed;
    left: 16px;
    right: 16px;
    bottom: 16px;
    max-width: 640px;
    margin: 0 auto;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 16px 20px;
    background: #1a2129;
    color: #e8edf2;
    border: 1px solid #2c3640;
    border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 0.95rem;
    z-index: 1000;
  }
  #cookie-banner[hidden] { display: none; }
  #cookie-banner p { margin: 0; flex: 1 1 240px; }
  #cookie-banner .cookie-actions { display: flex; gap: 8px; }
  #cookie-banner button {
    font: inherit;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 999px;
    border: 1px solid #4a5561;
    background: transparent;
    color: inherit;
    cursor: pointer;
  }
  #cookie-banner button.cookie-accept {
    background: #2ecc71;
    border-color: #2ecc71;
    color: #0f1419;
  }
</style>

<script>
  (function () {
    var banner = document.getElementById("cookie-banner");
    if (!/(?:^|;\s*)cookie_consent=/.test(document.cookie)) {
      banner.hidden = false;
    }
    banner.addEventListener("click", function (e) {
      var choice = e.target.getAttribute("data-consent");
      if (!choice) return;
      document.cookie = "cookie_consent=" + choice +
        "; max-age=" + 60 * 60 * 24 * 365 + "; path=/; SameSite=Lax";
      banner.hidden = true;
    });
  })();
</script>
'@

[IO.File]::WriteAllText((Join-Path $templates "landing.html"), $landing, $utf8)
[IO.File]::WriteAllText((Join-Path $templates "_cookie_banner.html"), $banner, $utf8)
Write-Host "OK  templates\landing.html and templates\_cookie_banner.html written" -ForegroundColor Green

$code = [IO.File]::ReadAllText($appPy)
if ($code -match 'def landing\(\)') {
    Write-Host "OK  app.py already has the intro page route" -ForegroundColor Green
} else {
    $pattern = '@app\.route\("/"\)(\r?\n)def index\(\):'
    if ($code -notmatch $pattern) {
        Write-Host "Could not find the index route in app.py - nothing changed. Send app.py to Claude." -ForegroundColor Red
        exit 1
    }
    Copy-Item $appPy "$appPy.bak" -Force
    $new = '@app.route("/")$1def landing():$1    return render_template("landing.html")$1$1$1@app.route("/upload")$1def index():'
    $code = ([regex]$pattern).Replace($code, $new, 1)
    [IO.File]::WriteAllText($appPy, $code, $utf8)
    Write-Host "OK  app.py updated (backup saved as app.py.bak)" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting Player Tracker... open http://127.0.0.1:5000 when it's ready." -ForegroundColor Cyan
Write-Host "Press Ctrl+C in this window to stop it." -ForegroundColor Cyan
Set-Location $root
py app.py
