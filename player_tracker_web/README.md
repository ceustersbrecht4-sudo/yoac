# YOAC Player Tracker

Rugby, soccer and basketball analysis for coaches: upload match footage, pick the
sport, and YOAC tracks every player, sorts them into the two teams by shirt
colour (the referee is left out), and builds a coach report with the moments
to review.

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

## Soccer

Pick **Soccer** when you upload. Tracking, teams and pitch marking are shared with rugby; the report is its own
(`soccer.py`):

- **Goalkeepers** wear their own colour, so the colour step first puts them with the officials. A keeper is the
  "other" person who is nearly always the last one at an end of the play, level with its middle (assistant referees
  are at the edge, the referee in the middle). The last outfield player in front of a keeper is one of their own
  defenders (the offside law), so that gives the keeper's team. 11 a side: anyone past that is a sub or staff.
- **Pitch marking**: goal lines, the 6-yard and penalty box lines, halfway, touchlines and the box sides, plus the
  centre and penalty spots (with the "Middle" line). The boxes are the same size on any pitch; length 40-120 m and
  width 25-90 m (default 105 x 68).
- **Which end each team defends**: the last player near each goal is almost always a defender of that end.
- **Who has the ball** (when the ball is seen): a team's player within 2 m of it for 5 frames in a row gets it,
  and keeps it until the other team does or nobody is on it for 3 s. When possession is known for 30% or more of the
  measured footage, the report splits attacking (with the ball) from defending (without). Otherwise it uses the
  coach's Attacking / Defending choice.
- **Team shape in metres** (pitch marked): back line height (the deepest four outfield players), length, width,
  the biggest gap between lines (a new line starts 5 m further up), whether the back four are level (within 5 m),
  and the shape (4-4-2, 4-3-3...) when all ten outfield players are in view. A measurement only counts when the
  space 5 m past the last player is on screen, so nobody can be hidden out of the picture.
- **Flags**: low / mid / high block (28 m and 45 m from goal, FIFA line height), stretched (over 40 m long) or
  compact (35 m or less), too wide without the ball (over 45 m), space between the lines (over 15 m), back line not
  level, ball carrier left alone (nobody within 8 m; pressure = within 5 m), narrow with the ball (under 40 m),
  rest defence (fewer than 3 outfield players 10 m behind the ball in the other half).
- **Set pieces** (`soccer_play.py`): a ball that sits still for 1 s or more is a restart, and where it sits says
  which: within 2.5 m of a corner flag = corner, the penalty spot = penalty, the centre spot = kick-off, the goal
  area = goal kick, the touchline = throw-in, anywhere else = free kick (only when nobody stands on the ball for
  0.5 s and the teams aren't both on it). For each: who took it, who got the first touch, and whether the taking
  team still had it 5 s later; at corners the numbers in the box and whether the defenders marked man-to-man
  (within 1.5 m of an attacker) or zonally; at free kicks any defender nearer than 8 m (9.15 m in law), and
  whether it was in shooting range (35 m); goal kicks short or long (first touch within 30 m).
- **After losing the ball**: how long until a player gets within 5 m of it (within 2 s = a counter-press) and how
  many outfield players are behind the ball 5 s later (fewer than 6 is flagged).
- **Passes and possible offside**: a pass is the ball going 5 m or more from one teammate to another within 2.5 s.
  A pass received by a player who stood more than 1 m past the second-last defender (and the ball, in the other
  half) when it was played is flagged, only when the space behind the line is on screen.
- **1v1s** (`soccer_duels.py`, works in close-ups like rugby's carries): a player on the ball with one defender within
  2 body lengths and nobody else within 3. Defending: side-on or square and knees bent or upright (pose model),
  about an arm's length away (0.45-1.3 body lengths), and who came away with the ball. On the ball: running at
  the defender's side or straight at them, a change of pace, and whether the ball was kept.
- **Tactics** a coach can pick (23): high press, mid block, low block, high line / offside trap, two banks of four,
  counter-press, man-marking, zonal, show them wide (touchline trap), zonal or man-to-man at corners; build up
  from the back, direct / long ball, switch the play, possession play, overload the ball side, hold the width,
  overlapping runs, counter-attack, rest defence, short corners, short or long goal kicks. Each says how much of
  the time (or of the spells, lost balls, corners or goal kicks) the team played it and whether it worked. The
  ones that need the ball say so when it was seen too rarely.

The ball is small from the stand and the COCO detector misses it often, so possession-based checks need a clip
where the ball stays in view. Every threshold is named at the top of `soccer.py`.

## Basketball

Pick **Basketball** when you upload, and mark the court (the corners of the key, halfway at a sideline, the court
corners; FIBA 28 x 15 m by default, NBA sizes fit too). Five a side. `basketball.py`:

- **Who is attacking**: the ball says who has it (a player within 2 m, 5 frames in a row); when the ball is hidden,
  the team nearer the basket of the half the players are in is defending (defenders stand between their player
  and the basket). Which basket each team attacks follows from that.
- **Half-court or transition**: 8 of 10 players in view in one half = a half-court set.
- **Offence**: spacing (distance to the nearest teammate: under 3.5 m crowded, 4.5 m or more good), a player in the
  corner, possible 3 seconds in the key, more than 8 s to get over halfway, possessions past 20 s of the 24 s clock.
- **Defence**: goal-side (an attacker has a defender nearer the basket than them: under 60% flagged, 80% good),
  help in the paint with the ball 6 m or more out, pressure on the ball (within 2 m), man-to-man or zone (60% or
  more staying with the same attacker = man; 30% or fewer = zone, with its shape read from the lines, e.g. 2-3).
- **Ball screens**: an attacker standing still (under 0.5 m in 0.5 s) within 1.3 m of the ball handler's defender.
  In the 1.5 s after the handler comes off it: switch, hedge or trap, drop, or over the top; worked = the ball or the
  handler in the key within 3 s.
- **Free throws** from the players standing still along the key, **on-ball defence** with the pose model (knees
  bent, about an arm's length away; close-ups included).
- **Tactics** a coach can pick (17): man-to-man, 2-3 / 3-2 / 1-3-1 zone, full-court press, pack the paint, switch /
  hedge / drop on screens, get back in transition; push the pace, use the clock, pick-and-roll, 5-out, 4-out 1-in,
  fill the corners, post-ups.

Shots, makes and rebounds aren't measured yet: the ball in the air can't be placed on the court from one camera.

## Rugby measurements

Besides line shape (gaps, dog-legs, numbers, width), the coach report now has:

- **Ruck speed** for every ruck the camera sees from start to end, timed the way coaches time it: from the
  tackle (ball carrier and tackler going down together) until the ball is out. When the ball is seen leaving
  the ruck that's the end; otherwise it's the group breaking up, which can read a little long. Rucks are banded
  the way analysts band them: quick under 3 s, usable 3 to 6 s, slow over 6 s (elite teams get 60 to 70%
  quick). The slowest
  rucks are shown as moments. Scrums are counted too: 10+ players with a pack of 4+ from each team that
  doesn't move (a big group that moves is a maul). This works on every video.
- **Metres, after marking the pitch.** In the **Pitch** tab, click at least 4 spots where two
  pitch lines cross (e.g. "Left 22 × Near 5 m line") on a wide shot, then press *Save and
  measure*. The app follows the camera's panning and zooming from there (once per video;
  a few minutes for a full match) and shows the players on a pitch seen from above, so you
  can check the marking. Then the report adds:
  - **Defensive line speed** in m/s in the first 1.5 s after each ruck. The defending team
    is the one whose line stands closest to the ruck. Under 1.5 m/s is flagged as passive,
    2.5 m/s and more as good.
  - **Mauls** (a breakdown that moves 3 m or more) told apart from rucks.
  - **Possible offside at the ruck** (Law 15): a defender standing more than 1 m in front of the offside
    line (the hindmost foot of the ruck on their side) for most of the last second before the ball comes out.
  - **Blitz or drift** after each ruck: the line coming forward hard (2.5 m/s and more) or sliding out towards
    the touchline (spreading at 1 m/s and more, faster than it comes forward).
  - **Backs too close at the scrum** (Law 19): backs less than 4 m (5 m in law, minus measuring error) behind
    their own scrum's hindmost foot. The player nearest the scrum is taken to be the scrum-half, players right
    next to the scrum count as pack, and scrums within 5 m of a try line aren't checked (the try line is the
    offside line there).
  - **Lineouts**: both teams in a single line, about a metre apart, between the 5 m and 15 m lines, for at
    least a second; and how many turned into a maul.
  Mark an extra frame after a replay or camera cut; each marked frame needs 4 points.

All thresholds are named at the top of `rugby.py`. These are measurements from one camera:
use them to find moments to review, and check them on the video.

## Carries into contact

When a report is made for a team that's attacking, every carry into contact is checked, close-ups included
(everything is measured against the player's own size, so zoom doesn't matter): did the carrier run at space
(a gap or a defender's edge) or straight at a defender, with footwork or not; were they low or upright at
contact and leading with the shoulder or square (from a pose model: `yolov8m-pose.pt`, downloaded on first use,
about 50 MB); and did their legs keep driving forward after contact. Worked out once per video and saved
(`outputs/<id>.carries.json`). Thresholds are at the top of `contact.py`.

## 15 a side

A rugby union team has 15 players on the pitch (30 in total, plus the officials). When the app sees more than 15
of one team in a frame, the extras are replacements warming up, staff or crowd in a similar colour: it keeps the 15
nearest the rest of the play and leaves the others out of the report, and says so. `PLAYERS_PER_TEAM` in
`analysis.py` sets the number (7 for sevens).

## Tactics

In the coach report, after choosing one team and attacking or defending, tick the tactics you planned: blitz,
drift or passive hold; gap trap, fan or compete at every ruck; two or three in the backfield; and in attack
quick tempo, pick and go, play off 9 or off 10, wide, overload one side, flat or deep, and 1-3-3-1 or 2-4-2 pods.
The report says for each one in what share of the rucks it was played and whether it worked (the rules are
listed on the Coach report page, and in `tactics.py`). Most need the pitch marked. A gap trap you picked turns the
"holes in the defensive line" warning into a note, and is checked as a trap instead.

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

## The front pages

After logging in you land on **Upload**: a working upload box next to a frame of your latest analysed
match. Uploading sends you to the **Processing** page, which follows your video through the real stages
(shrink, track, teams, draw) and takes you to the tracked video and report when it's done. The menu opens
the other pages: **How it works**, **Your match** (close-ups from your latest video and numbers counted
from its tracking data), **Coach report** (what the report flags and the rule behind each flag),
**Pricing** and **FAQ**. Each page ends with a link to the next one.

Only real footage is shown as footage. Until you've analysed a video, the pages show drawings in the
tracker's style instead, and say so.

To show one fixed match to every account instead, pick a finished video (its id is in the address:
`/upload#job=...`) and run once:

    python demo.py JOB_ID

This writes `static/demo/raw.jpg`, `tracked.jpg` and `demo.json`. Delete those three files to go back
to each coach's own latest video. Only use footage you're allowed to show to everyone with an account.

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
| `soccer.py` | Soccer: goalkeepers, possession, team shape in metres, soccer tactics |
| `soccer_play.py` | Soccer set pieces, pressing after losing the ball, passes and possible offside |
| `soccer_duels.py` | Soccer 1v1s with the pose model (close-ups included); also basketball on-ball defence |
| `basketball.py` | Basketball: possession, spacing, the key, help defence, screens, zones, free throws, tactics |
| `demo.py` | The front page's frame, close-ups and measured numbers, from a real analysed video |
| `templates/` | The pages: login, front pages (`_front.html` + one file per page), upload tool, cookie banner |
| `static/` | YOAC logos and the Barlow Condensed font (SIL OFL) |
| `start.bat` | Double-click to install requirements and start the app |

Created while running: `uploads/` and `outputs/` (videos and results),
`users.json` (accounts), `invites.json` (invite codes) and `secret.key` (keeps you logged in). Delete
`users.json` if you forget your password, then create a new account.
