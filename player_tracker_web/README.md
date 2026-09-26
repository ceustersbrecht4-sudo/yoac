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
3. Open **http://127.0.0.1:5000** in your browser and create the first account.
   **The first account becomes the owner.** You land on the Account page: turn on
   two-step login there, then make invite codes for your staff. Nobody can create
   an account without an invite code from the owner.

Keep the black window open while you use the app; closing it stops the app.

## Before others use it: fill in your details

Open **legal_info.json** with Notepad and replace everything in [square brackets]
with your own details (name or club, address, e-mail, enterprise/VAT number if you
have one). These appear in the privacy policy, terms of use and legal notice. The
legal pages show an orange "Not finished yet" notice until you've done this.
`video_retention_days` sets how long videos are kept before they're deleted
automatically (0 = keep until deleted by hand).

## Rugby measurements

Besides line shape (gaps, dog-legs, numbers, width), the coach report now has:

- **Ruck speed** for every ruck the camera sees from start to end: the time from the
  players meeting at the tackle until the group breaks up (usually the ball coming out).
  Quick ball is under 3 s, slow ball over 6 s; the slowest rucks are shown as moments.
  Scrums are counted too. This works on every video.
- **Metres, after marking the pitch.** In the **Pitch** tab, click at least 4 spots where two
  pitch lines cross (e.g. "Left 22 × Near 5 m line") on a wide shot, then press *Save and
  measure*. The app follows the camera's panning and zooming from there (once per video;
  a few minutes for a full match) and shows the players on a pitch seen from above, so you
  can check the marking. Then the report adds:
  - **Defensive line speed** in m/s in the first 1.5 s after each ruck. The defending team
    is the one whose line stands closest to the ruck. Under 1.5 m/s is flagged as passive,
    2.5 m/s and more as good.
  - **Mauls** (a breakdown that moves 3 m or more) told apart from rucks.
  Mark an extra frame after a replay or camera cut; each marked frame needs 4 points.

All thresholds are named at the top of `rugby.py`. These are measurements from one camera:
use them to find moments to review, and check them on the video.

## Plans and usage limits

Every account has a plan with four limits, the things that cost money once the app is online:

| | Free | Plus | Pro |
|---|---|---|---|
| Storage (videos kept) | 2 GB | 25 GB | 100 GB |
| Video analysed per month | 120 min | 600 min | 3000 min |
| Watching + downloading per month | 10 GB | 100 GB | 400 GB |
| Videos deleted automatically after | 30 days | 180 days | 365 days |

Change the numbers in **plans.json** with Notepad (applies straight away) and pick each
account's plan on the Account page. The owner has no limits. Uploads over a limit are refused
with a message; minutes of videos refused by the content check are given back. When the
monthly watching/download allowance is used up, videos and reports stay, but videos can't be
played or downloaded until the 1st of next month. Online payments can be connected later.

**Space saving:** every upload is shrunk before it's analysed (at most 1280 pixels wide, 30
frames a second, no sound). Phone footage usually becomes 5 to 15 times smaller and the
analysis gets faster; the original file is deleted. Tracked videos are saved compactly too.

**Whole server:** `"_server"` in plans.json pauses uploads for everyone when the disk has
less than `min_free_disk_gb` free (default 5 GB), or when all videos together would pass
`max_total_storage_gb` (0 = no limit). Leftover files from a crash are cleaned up after a day.

If you already had a plans.json, it is kept: the new limits use the defaults above until
you add them to your file.

## Putting it online

The app is built for the Wi-Fi first. To put it on the internet, run it behind a web server
that provides HTTPS (for example Caddy, which gets certificates automatically) and set:

| Setting | Value | Why |
|---|---|---|
| `YOAC_HOST` | `127.0.0.1` | Only the web server in front can reach the app |
| `YOAC_PORT` | e.g. `5000` | Port the web server forwards to |
| `YOAC_ALLOWED_HOSTS` | your domain, e.g. `tracker.yoac.be` | Requests for other names are refused |
| `YOAC_BEHIND_PROXY` | `1` | Rate limits see each visitor's real address |
| `YOAC_HTTPS` | `1` | Login cookie only sent encrypted; browsers always use HTTPS |

Only set `YOAC_BEHIND_PROXY` when there really is a web server in front, otherwise anyone
could fake their address. Example Caddyfile: `tracker.yoac.be { reverse_proxy 127.0.0.1:5000 }`.
Also allow uploads of 2 GB in that web server, and read SECURITY.md first.

## Security

See **SECURITY.md** for the full security review, what was fixed and the known limits.

- Each video is private to the account that uploaded it; the owner can see and
  delete every upload (Account page > Uploads) and remove accounts.
- Uploads are checked automatically: only real videos that show a pitch and players
  are analysed. Anything else is refused and deleted straight away.

- Invite-only sign-up; the owner role and all admin actions are checked on the server.
- Two-step login (TOTP authenticator app) with 8 one-time recovery codes.
- Passwords: at least 10 characters, not containing the username, not a common
  password, and checked against known data breaches (Have I Been Pwned; only the
  first 5 characters of a SHA-1 fingerprint are sent, never the password).
- Rate limits: wrong passwords per device (10 / 10 min) and per account
  (5 / 15 min, from anywhere), sign-ups, 2FA codes and password changes.
- The login is an HttpOnly, SameSite cookie that page scripts can't read; a strict
  Content Security Policy blocks injected scripts; forms have CSRF tokens.
- Changing or resetting a password, or turning 2FA off, logs the account out everywhere.
- Locked out? The owner can give you a temporary password on the Account page
  (it turns your 2FA off; you choose a new password at the next login). If the
  owner is locked out, delete `users.json` to start again (this removes all accounts).
- The connection inside your Wi-Fi is plain HTTP: only use the app on a network you trust.

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
| `accounts.py` | Log in, invite-only sign-up, 2FA, account page and owner admin (hashes in `users.json`) |
| `security.py` | Security headers/CSP, host check, rate limits, password rules + breach check, TOTP codes |
| `content_check.py` | Refuses uploads that aren't match footage (no pitch / no players) |
| `quota.py`, `plans.json` | Plans and usage limits (storage, monthly minutes, monthly watching/downloads, keep time) |
| `compact.py` | Shrinks each upload before analysis to save space |
| `legal.py`, `legal_info.json` | Privacy, cookie, terms, legal notice and licence pages; your details; source download |
| `tracker.py` | Finds and tracks players with YOLO, draws the tracked video |
| `teams.py` | Sorts players into the two teams by shirt colour, sets officials aside |
| `analysis.py` | Coach report: line gaps, dog-legs, numbers, breakdown |
| `pitch.py` | From pixels to metres: the marked pitch points and following the camera |
| `rugby.py` | Rucks, mauls, scrums, ruck speed and defensive line speed |
| `templates/` | The pages: login, front page, upload tool, cookie banner |
| `static/` | YOAC logos and the Barlow Condensed font (SIL OFL) |
| `start.bat` | Double-click to install requirements and start the app |

Created while running: `uploads/` and `outputs/` (videos and results),
`users.json` (accounts), `invites.json` (invite codes) and `secret.key` (keeps you logged in). Delete
`users.json` if you forget your password, then create a new account.
