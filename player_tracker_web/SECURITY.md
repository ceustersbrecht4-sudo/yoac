# Security review: YOAC Player Tracker

Date: 25 September 2026. Scope: every route in `app.py`, `accounts.py`, `legal.py`,
`security.py`, `content_check.py`, and every page in `templates/`.

## What was found and fixed

| # | Issue | Risk | Fix |
|---|---|---|---|
| 1 | **Any account could open any other account's videos** (status, video, frames, snapshots, report, hide, delete) by job ID; the front page showed a frame of *anyone's* latest video (IDOR) | High | Every job stores its uploader; all job routes check access on the server. Others get the same 404 as a missing video. The front-page demo only uses your own videos. The owner can see everything, for moderation. |
| 2 | **No check on what was uploaded**: any video could be stored and shared (gore, explicit, anything) | High | `content_check.py`: the file must be a readable video (max 4096 px, max 2 h) whose frames mostly show a pitch; after detection it must contain several players in a good share of frames. Refused uploads are deleted immediately. 3 refusals pause uploads for 24 h. |
| 3 | No upload limits (disk filling, CPU hogging) | Medium | 20 uploads per account per day, max 3 waiting per account, 2 GB per file. |
| 4 | Upload, hide and delete (JavaScript actions) relied on SameSite cookies alone for CSRF protection | Medium | They now also need the CSRF token in an `X-CSRF-Token` header. |
| 5 | DNS rebinding: a malicious website could make a browser on the Wi-Fi talk to the app | Medium | Requests are only answered for localhost, private network addresses and the laptop's own name (`YOAC_ALLOWED_HOSTS` adds more). |
| 6 | Sign-up told people without an invite whether a username existed | Low | The invite is checked first. |
| 7 | Error messages could show internal file paths | Low | Unexpected errors show a general message; details only go to the console. |
| 8 | A malformed "hide teams" request could crash that request | Low | The request body is validated. |
| 9 | Removing an account left its videos behind | Low | An account's videos are deleted with it. |
| 10 | Flask's development server was used | Low | The app uses Waitress (a production web server) when installed; it is in `requirements.txt`. |
| 11 | Uploads failed if the uploads folder was deleted while the app was running | Low | The folders are recreated when needed. |

## Already in place (from the earlier review)

- Login: HttpOnly + SameSite session cookie (not readable by scripts), signed with a random key.
- Strict Content Security Policy with a per-request nonce; no inline event handlers; clickjacking protection.
- Invite-only sign-up; owner role checked on the server; two-step login (TOTP) with hashed recovery codes.
- Rate limits per address and per account on login, sign-up, 2FA codes and password changes.
- Password rules and a Have I Been Pwned breach check (k-anonymity).
- CSRF tokens on every form; safe redirects only; hashed invite codes.
- Uploads: file names are never used as paths (random IDs), extensions are allow-listed.
- Templates escape all user text; the page scripts only insert user text as text, never as HTML.

## How it was tested

122 automated checks (4 suites), including: another account opening someone's video
through every route, uploads without a CSRF token, a foreign Host header, a gore-like
clip, an indoor clip, a text file renamed to .mp4, a pitch with no players, a crashing
analysis, upload limits, username probing, owner-only moderation, and all earlier
login, 2FA, legal and deletion checks. Also run end to end in Chromium on the Waitress
server: 0 errors, 0 Content Security Policy violations.

## Known limits (honest)

- **The content check is a filter, not a guarantee.** It looks for a pitch and players; it
  can't understand what happens in a video. Something unsuitable filmed on a pitch could get
  through, and unusual real footage (very close-up, snow) could be refused. That's why every
  upload is linked to an account, videos are private to the uploader, and the owner can
  review, delete and remove accounts.
- **Plain HTTP on the Wi-Fi.** Traffic between the phone and the laptop isn't encrypted.
  Only use trusted networks. Putting the app online properly (HTTPS) would fix this.
- **The laptop itself.** Anyone with access to the laptop's files can read the videos and
  `users.json` (passwords are hashed). Use a Windows password and disk encryption (BitLocker).
- **Video decoding.** Uploads are decoded by FFmpeg/OpenCV. Keep them updated
  (`py -m pip install --upgrade opencv-python imageio-ffmpeg ultralytics`).
- Rate limits are kept in memory and reset when the app restarts.
