# Player Tracker: intro page and interactive moments

This folder holds the updated Player Tracker files:

- `app.py`: intro page at `/`, upload tool at `/upload`, plus `/frame` (live
  processing preview) and `/demo/raw.jpg` / `/demo/tracked.jpg` (landing page slider).
- `templates/landing.html`: intro page with the before/after demo slider.
- `templates/index.html`: upload tool with drop-zone feedback, a live frame strip
  while processing, and the raw-to-tracked reveal when the result is ready.
- `templates/_cookie_banner.html`: cookie consent banner.

## Install on the laptop

In PowerShell:

```powershell
cd "$HOME\Downloads\player_tracker_web"; Invoke-WebRequest "https://raw.githubusercontent.com/ceustersbrecht4-sudo/yoac/claude/locate-app-py-sobmfj/player_tracker_web/update_player_tracker.ps1" -OutFile update_player_tracker.ps1 -UseBasicParsing; powershell -ExecutionPolicy Bypass -File .\update_player_tracker.ps1
```

It backs up the current files into a `backup_<date>` folder, installs the new ones
and restarts the app.

## Demo slider

The slider on the intro page uses a frame from the most recently analysed video.
To pin a specific image instead, put `raw.jpg` and `tracked.jpg` (the same frame,
before and after) in `static/demo/`.
