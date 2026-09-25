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

## Before others use it: fill in your details

Open **legal_info.json** with Notepad and replace everything in [square brackets]
with your own details (name or club, address, e-mail, enterprise/VAT number if you
have one). These appear in the privacy policy, terms of use and legal notice. The
legal pages show an orange "Not finished yet" notice until you've done this.
`video_retention_days` sets how long videos are kept before they're deleted
automatically (0 = keep until deleted by hand).

## Legal

- **Privacy (GDPR):** accounts and videos stay on this computer; passwords are
  hashed; everyone must accept the terms and privacy policy when signing up and
  confirm they may use a video before uploading it; videos can be deleted at any
  time and are deleted automatically after the retention period; accounts can be
  deleted on the privacy page.
- **Cookies:** only essential cookies. The optional "Recent videos" list is only
  saved after pressing OK in the cookie banner.
- **Open source:** the tracking engine (Ultralytics YOLO) is licensed AGPL-3.0.
  Everyone who uses the app over a network can download its source code from
  Licences > "Download the source code" (/source). If you want to sell the app or
  keep your code private, you need an Ultralytics Enterprise licence instead.
- These documents are careful templates, not legal advice. Have them checked
  before offering the app to the public or charging for it.

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
| `legal.py`, `legal_info.json` | Privacy, cookie, terms, legal notice and licence pages; your details; source download |
| `tracker.py` | Finds and tracks players with YOLO, draws the tracked video |
| `teams.py` | Sorts players into the two teams by shirt colour, sets officials aside |
| `analysis.py` | Coach report: line gaps, dog-legs, numbers, breakdown |
| `templates/` | The pages: login, front page, upload tool, cookie banner |
| `static/` | YOAC logos and the Barlow Condensed font (SIL OFL) |
| `start.bat` | Double-click to install requirements and start the app |

Created while running: `uploads/` and `outputs/` (videos and results),
`users.json` (accounts) and `secret.key` (keeps you logged in). Delete
`users.json` if you forget your password, then create a new account.
