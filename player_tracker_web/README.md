# YOAC Player Tracker

Rugby match analysis for coaches: upload match footage, and YOAC tracks every
player, sorts them into the two teams by shirt colour (the referee is left
out), and builds a coach report with the moments to review.

It runs on your own laptop. Your account and videos stay on it.

## Start it (Windows)

1. Install Python 3 from https://www.python.org/downloads/ if it isn't installed
   (tick "Add python.exe to PATH" during setup).
2. Double-click **start.bat**. The first start installs what the app needs and
   downloads the detection model, which takes a few minutes.
3. Open **http://127.0.0.1:5000** in your browser, create an account and log in.

Keep the black window open while you use the app; closing it stops the app.

## On your phone

Use the "On your phone" address shown in the black window (the phone must be on
the same Wi-Fi). If the phone can't connect, allow the app through Windows
Firewall. Run this once in PowerShell *as administrator*:

    New-NetFirewallRule -DisplayName "YOAC Player Tracker" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -Profile Private,Domain

and set your Wi-Fi network to **Private** in Windows settings.

## What's in here

| File | What it does |
| --- | --- |
| `app.py` | The web app: pages, upload, progress, results, coach report |
| `accounts.py` | Log in / create account (passwords stored as hashes in `users.json`) |
| `tracker.py` | Finds and tracks players with YOLO, draws the tracked video |
| `teams.py` | Sorts players into the two teams by shirt colour, sets officials aside |
| `analysis.py` | Coach report: line gaps, dog-legs, numbers, breakdown |
| `templates/` | The pages: login, front page, upload tool, cookie banner |
| `static/` | YOAC logos and the Barlow Condensed font (SIL OFL) |
| `start.bat` | Double-click to install requirements and start the app |

Created while running: `uploads/` and `outputs/` (videos and results),
`users.json` (accounts) and `secret.key` (keeps you logged in). Delete
`users.json` if you forget your password, then create a new account.
