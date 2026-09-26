"""
Local web app around track_players_v6: upload a match video, click Analyze,
get back the tracked video.

Run:
    py -m pip install -r requirements.txt
    py app.py
Then open http://127.0.0.1:5000
"""

import glob
import json
import os
import queue
import shutil
import socket
import threading
import time
import traceback
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import cv2

import accounts
import analysis
import compact
import numpy as np
import pitch
import quota
import rugby
from accounts import init_accounts
from content_check import ContentRejected, check_players, check_upload
from legal import init_legal, retention_days
from security import Throttle, init_online, init_security_headers
from teams import assign_teams
from tracker import BUCKET_COLORS, count_buckets, detect_players, render_video, video_info

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
OUTPUT_DIR = os.path.join(HERE, "outputs")
DEMO_DIR = os.path.join(HERE, "static", "demo")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
# Match officials (teams.OTHER) aren't tracked in the video by default; the
# Teams tab can still bring them back.
HIDDEN_BY_DEFAULT = ["other"]

# Abuse limits per account.
MAX_ACTIVE_JOBS = 3                     # videos queued or being analysed at once
uploads_per_day = Throttle(20, 86400)   # uploads in 24 hours
rejects_per_day = Throttle(3, 86400)    # uploads refused by the content check in 24 hours
image_requests = Throttle(300, 60)      # video frames and snapshots per minute (each one decodes video)
report_requests = Throttle(30, 60)      # coach reports per minute
redraws_per_hour = Throttle(20, 3600)   # "Update video" redraws per hour
calibrations_per_hour = Throttle(30, 3600)  # pitch markings saved per hour
# Pages that send video or images back: counted against the monthly transfer.
TRANSFER_ENDPOINTS = {"video", "frame", "snapshot", "demo"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB
app.config["TEMPLATES_AUTO_RELOAD"] = True  # page edits show up without a restart
init_accounts(app)  # every page needs a logged-in user; see accounts.py
init_legal(app)     # privacy / cookies / terms / legal notice / licences, and /source
init_security_headers(app)  # Content Security Policy and other protective headers
init_online(app)            # HTTPS cookies, HSTS and the real visitor address behind a proxy (when online)

# job_id -> {status, done, total, message, teams, colors, hidden, version,
#            filename, input, output, started}
jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()


def _meta_path(job_id):
    return os.path.join(OUTPUT_DIR, f"{job_id}.json")


def _detections_path(job_id):
    return os.path.join(OUTPUT_DIR, f"{job_id}.detections.json")


def _calib_path(job_id):
    return os.path.join(OUTPUT_DIR, f"{job_id}.calib.json")


def _motion_path(job_id):
    return os.path.join(OUTPUT_DIR, f"{job_id}.motion.npy")


def _save_meta(job_id):
    # Finished jobs are written to disk so results survive a server restart.
    with jobs_lock:
        job = jobs[job_id]
        meta = {k: job.get(k) for k in ("filename", "input", "output", "teams", "colors", "hidden", "version",
                                        "owner", "uploaded", "original_name", "seconds")}
    with open(_meta_path(job_id), "w") as f:
        json.dump(meta, f)


def _load_saved_jobs():
    for path in glob.glob(os.path.join(OUTPUT_DIR, "*.json")):
        job_id = os.path.splitext(os.path.basename(path))[0]
        if "." in job_id:  # detections, pitch marking... not a job's own file
            continue
        try:
            with open(path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if not os.path.exists(meta.get("output") or ""):
            continue
        # Videos from before accounts existed have no owner: only the app's
        # owner can see those.
        jobs[job_id] = dict(meta, status="done", done=0, total=0, message="Done", started=None)


def _can_see(job):
    """Each video is private to the account that uploaded it; the app's owner
    can see every video (to moderate uploads)."""
    return job.get("owner") == accounts.current_user() or accounts.user_is_owner()


def _job_or_404(job_id):
    """The job, if it exists AND this user may see it. Someone else's video
    gives the same 404 as a missing one, so IDs can't be probed."""
    job = jobs.get(job_id)
    if not job or not _can_see(job):
        abort(404)
    return job


def _update(job_id, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)


def _progress(job_id, message):
    # Each stage gets its own progress bar and time estimate.
    _update(job_id, message=message, done=0, total=0, started=time.time())
    return lambda done, total: _update(job_id, done=done, total=total)


def _shrink(job_id):
    """Replace the upload with a much smaller copy (see compact.py)."""
    job = jobs[job_id]
    src = job["input"]
    dst = os.path.join(UPLOAD_DIR, f"{job_id}.small.mp4")
    if not compact.shrink(src, dst, job.get("seconds", 0), progress=_progress(job_id, "Shrinking the video to save space...")):
        return
    if os.path.getsize(dst) >= os.path.getsize(src):
        os.remove(dst)  # it was already small: keep the original
        return
    _update(job_id, input=dst)
    try:
        os.remove(src)
    except OSError:
        pass


def _analyze(job_id):
    _shrink(job_id)
    job = jobs[job_id]
    detections = detect_players(job["input"], progress=_progress(job_id, "Tracking players..."))
    check_players(detections)  # no match in it -> ContentRejected
    detections, colors = assign_teams(job["input"], detections, progress=_progress(job_id, "Sorting players into teams..."))
    with open(_detections_path(job_id), "w") as f:
        json.dump(detections, f, separators=(",", ":"))
    teams = count_buckets(detections)
    hidden = [b for b in HIDDEN_BY_DEFAULT if b in teams]
    # Colours are known now, so the processing preview can draw boxes in team
    # colours while the final video is still being drawn.
    _update(job_id, colors=colors, hidden=hidden)
    render_video(job["input"], detections, job["output"], hidden, colors=colors,
                 progress=_progress(job_id, "Drawing the video..."))
    _update(job_id, teams=teams, colors=colors, hidden=hidden)


def _render(job_id, hidden):
    job = jobs[job_id]
    with open(_detections_path(job_id)) as f:
        detections = json.load(f)
    version = job["version"] + 1
    # New file name per version, so the browser doesn't show a cached copy
    # and the old file can still be streamed while the new one is written.
    new_output = os.path.join(OUTPUT_DIR, f"{job_id}_v{version}.mp4")
    render_video(job["input"], detections, new_output, hidden, colors=job.get("colors"),
                 progress=_progress(job_id, "Updating video..."))
    old_output = job["output"]
    _update(job_id, output=new_output, hidden=hidden, version=version)
    try:
        os.remove(old_output)
    except OSError:
        pass  # still being streamed; harmless to leave behind


def _follow_camera(job_id):
    """How the camera moved, frame to frame (see pitch.py). Done once per
    video, the first time its pitch is marked."""
    job = jobs[job_id]
    with open(_detections_path(job_id)) as f:
        detections = json.load(f)
    moves = pitch.camera_motion(job["input"], detections, progress=_progress(job_id, "Following the camera..."))
    tmp = _motion_path(job_id) + ".part.npy"
    np.save(tmp, moves)
    os.replace(tmp, _motion_path(job_id))


def worker():
    # One video at a time: YOLO on a laptop already uses all the CPU/GPU it
    # can get, so running jobs in parallel would just make each one slower.
    while True:
        kind, job_id, arg = job_queue.get()
        _update(job_id, status="processing", started=time.time(), done=0, total=0,
                message={"analyze": "Loading model...", "motion": "Following the camera..."}.get(kind, "Updating video..."))
        try:
            if kind == "analyze":
                _analyze(job_id)
            elif kind == "motion":
                _follow_camera(job_id)
            else:
                _render(job_id, arg)
            _update(job_id, status="done", message="Done")
            _save_meta(job_id)
        except ContentRejected as e:
            # Not match footage: remove the upload and anything made from it.
            with jobs_lock:
                job = dict(jobs.get(job_id) or {})
            for path in _job_files(job_id, job):
                try:
                    os.remove(path)
                except OSError:
                    pass
            if job.get("owner"):
                rejects_per_day.hit(job["owner"])
                quota.record_seconds(job["owner"], -job.get("seconds", 0))  # refused: minutes given back
            _update(job_id, status="error", message=str(e))
        except Exception as e:
            traceback.print_exc()  # details stay in the console, not on the page
            friendly = str(e) if isinstance(e, ValueError) else "Something went wrong while analysing this video."
            if kind == "analyze":
                _update(job_id, status="error", message=friendly)
            elif kind == "motion":
                _update(job_id, status="done", message="Couldn't follow the camera in this video, so the report stays in body lengths.")
            else:
                # The previous video is still fine - go back to it.
                _update(job_id, status="done", message="Could not update the video. The previous version is kept.")
        finally:
            job_queue.task_done()


@app.route("/")
def landing():
    return render_template("landing.html")


def _storage_used(user_key):
    """Bytes on disk for this account's videos and everything made from them."""
    with jobs_lock:
        theirs = [(jid, dict(j)) for jid, j in jobs.items() if j.get("owner") == user_key]
    total = 0
    for jid, job in theirs:
        for path in _job_files(jid, job):
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
    return total


def _usage(user_key):
    return quota.summary(user_key, _storage_used(user_key))


def _folder_bytes(*folders):
    total = 0
    for folder in folders:
        for entry in os.scandir(folder) if os.path.isdir(folder) else ():
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                pass
    return total


@app.before_request
def _transfer_limit():
    """Stop sending video and images once this month's transfer is used up,
    and don't let one account hammer the pages that decode video."""
    if request.endpoint not in TRANSFER_ENDPOINTS | {"pitch_view"} or not accounts.current_user():
        return None
    user = accounts.current_user()
    problem = _transfer_blocked(user) if request.endpoint in TRANSFER_ENDPOINTS else None
    if problem:
        return jsonify(error=problem, limit="transfer"), 429
    if request.endpoint in ("frame", "snapshot", "demo", "pitch_view"):
        if image_requests.wait(user):
            return jsonify(error="Too many requests. Wait a minute and try again."), 429
        image_requests.hit(user)
    return None


@app.after_request
def _count_transfer(resp):
    if request.endpoint in TRANSFER_ENDPOINTS and resp.status_code in (200, 206):
        quota.record_sent(accounts.current_user(), resp.content_length or 0)
    return resp


@app.route("/upload")
def index():
    return render_template("index.html", usage=_usage(accounts.current_user()))


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify(error="No video uploaded."), 400
    if request.form.get("rights") != "1":
        return jsonify(error="Please confirm you're allowed to use this video before analysing it."), 400

    user = accounts.current_user()
    name = secure_filename(file.filename) or "video.mp4"
    base, ext = os.path.splitext(name)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported file type. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}"), 400
    if rejects_per_day.wait(user):
        return jsonify(error="Uploads are paused for your account for 24 hours because several videos weren't "
                             "match footage. Contact the owner of this app if this is a mistake."), 429
    wait = uploads_per_day.wait(user)
    if wait:
        return jsonify(error=f"You've reached today's upload limit. Try again in {wait // 3600 + 1} hours."), 429
    with jobs_lock:
        active = sum(1 for j in jobs.values() if j.get("owner") == user and j["status"] in ("queued", "processing"))
    if active >= MAX_ACTIVE_JOBS:
        return jsonify(error=f"You already have {active} videos waiting or being analysed. Wait for one to finish."), 429
    incoming = request.content_length or 0
    usage = _usage(user)
    # While it's being prepared the upload needs its full size; it's shrunk
    # before analysis, so what stays is usually far smaller.
    problem = quota.storage_problem(usage, incoming)
    if problem:
        return jsonify(error=problem, limit="storage"), 403
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    problem = quota.server_problem(incoming, _folder_bytes(UPLOAD_DIR, OUTPUT_DIR), shutil.disk_usage(UPLOAD_DIR).free)
    if problem:
        return jsonify(error=problem, limit="server"), 507

    job_id = uuid.uuid4().hex[:12]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext.lower()}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    os.makedirs(UPLOAD_DIR, exist_ok=True)  # in case the folder was deleted while running
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file.save(input_path)
    uploads_per_day.hit(user)
    try:
        facts = check_upload(input_path)  # a real, sensible-sized video that shows a pitch
    except ContentRejected as e:
        try:
            os.remove(input_path)
        except OSError:
            pass
        rejects_per_day.hit(user)
        return jsonify(error=str(e)), 422
    problem = quota.minutes_problem(usage, facts["seconds"])
    if problem:
        os.remove(input_path)
        return jsonify(error=problem, limit="minutes"), 403
    quota.record_seconds(user, facts["seconds"])

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "done": 0,
            "total": 0,
            "message": "Waiting in queue...",
            "teams": None,
            "hidden": [],
            "version": 0,
            "filename": f"{base}_tracked.mp4",
            "input": input_path,
            "output": output_path,
            "started": None,
            "owner": user,                    # who uploaded it (access + accountability)
            "seconds": facts["seconds"],      # counted against the monthly analysis minutes
            "uploaded": int(time.time()),
            "original_name": name,
        }
    job_queue.put(("analyze", job_id, None))
    return jsonify(job_id=job_id)


def _job_files(job_id, job):
    files = set(glob.glob(os.path.join(OUTPUT_DIR, f"{job_id}*")))
    files.update(glob.glob(os.path.join(UPLOAD_DIR, f"{job_id}*")))
    for key in ("input", "output"):
        if job.get(key):
            files.add(job[key])
    return files


def _delete_job(job_id):
    """Remove a job and every file made from it: upload, tracked video(s),
    detections and saved metadata."""
    with jobs_lock:
        job = jobs.pop(job_id, None)
    if job is None:
        return False
    with _prepared_lock:
        _prepared.pop(job_id, None)
    for path in _job_files(job_id, job):
        try:
            os.remove(path)
        except OSError:
            pass
    return True


@app.route("/delete/<job_id>", methods=["POST"])
def delete(job_id):
    """Right to erasure: delete a video and everything the app made from it."""
    with jobs_lock:
        job = _job_or_404(job_id)
        if job["status"] not in ("done", "error"):
            return jsonify(error="This video is still being processed. Delete it once it's finished."), 409
    _delete_job(job_id)
    return jsonify(ok=True)


def _keep_days(owner):
    """Days a video is kept: the shortest of the retention period in
    legal_info.json and the uploader's plan (0 = no limit)."""
    limits = [d for d in (retention_days(), quota.keep_days_for(owner) if owner else 0) if d]
    return min(limits) if limits else 0


def _purge_old_jobs():
    """Storage limitation: delete finished videos older than the retention
    period in legal_info.json or the uploader's plan, whichever is shorter."""
    now = time.time()
    with jobs_lock:
        done = [(jid, dict(j)) for jid, j in jobs.items() if j["status"] in ("done", "error")]
    old = []
    for jid, job in done:
        days = _keep_days(job.get("owner"))
        born = job.get("uploaded") or (os.path.getmtime(job["output"]) if os.path.exists(job.get("output") or "") else now)
        if days and born < now - days * 86400:
            old.append(jid)
    for jid in old:
        _delete_job(jid)
    if old:
        print(f"Deleted {len(old)} video(s) past their keep time.")


def _clean_stray_files():
    """Files no job knows about (left by a crash mid-upload or mid-render)
    are removed after a day, so they can't slowly fill the disk."""
    with jobs_lock:
        known = set(jobs)
    cutoff = time.time() - 86400
    for folder in (UPLOAD_DIR, OUTPUT_DIR):
        for entry in os.scandir(folder) if os.path.isdir(folder) else ():
            job_id = entry.name[:12]
            try:
                if entry.is_file() and job_id not in known and entry.stat().st_mtime < cutoff:
                    os.remove(entry.path)
            except OSError:
                pass


def _purge_loop():
    last = 0
    while True:
        try:
            quota.flush_sent()
            if time.time() - last > 3600:
                last = time.time()
                _purge_old_jobs()
                _clean_stray_files()
        except Exception:
            traceback.print_exc()
        time.sleep(60)


@app.route("/hide/<job_id>", methods=["POST"])
def hide(job_id):
    """Redraw the video without the given colour buckets."""
    with jobs_lock:
        job = _job_or_404(job_id)
        if job["status"] != "done":
            return jsonify(error="This video is still being processed."), 409
        if not os.path.exists(_detections_path(job_id)):
            return jsonify(error="This video was analysed before removing colours was possible. Please analyse it again."), 409
        if redraws_per_hour.wait(accounts.current_user()):
            return jsonify(error="You've redrawn videos a lot in the last hour. Try again later."), 429
        body = request.get_json(silent=True, force=True)
        wanted = body.get("hidden", []) if isinstance(body, dict) else []
        hidden = sorted({b for b in wanted if isinstance(b, str) and b in (job["teams"] or {})})
        job.update(status="queued", message="Waiting in queue...")
    redraws_per_hour.hit(accounts.current_user())
    job_queue.put(("render", job_id, hidden))
    return jsonify(ok=True)


def _transfer_blocked(user_key):
    # Storage isn't needed to answer this, so skip measuring it.
    return quota.transfer_problem(quota.summary(user_key, 0))


@app.route("/status/<job_id>")
def status(job_id):
    blocked = _transfer_blocked(accounts.current_user())
    with jobs_lock:
        job = _job_or_404(job_id)
        queued_ahead = sum(
            1 for j in jobs.values() if j["status"] == "queued"
        ) - 1 if job["status"] == "queued" else 0
        return jsonify(
            status=job["status"],
            done=job["done"],
            total=job["total"],
            message=job["message"],
            teams=job["teams"],
            colors=_hex_colors(job),
            hidden=job["hidden"],
            version=job["version"],
            can_hide=os.path.exists(_detections_path(job_id)),
            elapsed=(time.time() - job["started"]) if job["started"] else 0,
            queued_ahead=max(0, queued_ahead),
            transfer_blocked=blocked,
        )


# job_id -> per-frame analysis data (same for every team/phase choice)
_prepared = {}
_prepared_lock = threading.Lock()


def _prepared_frames(job_id, video_path):
    with _prepared_lock:
        if job_id not in _prepared:
            cap = cv2.VideoCapture(video_path)
            ok, first = cap.read()
            cap.release()
            size = (first.shape[1], first.shape[0]) if ok else None
            with open(_detections_path(job_id)) as f:
                _prepared.clear()  # keep only one video in memory
                _prepared[job_id] = analysis.prepare(json.load(f), size)
                _sizes[job_id] = size
        return _prepared[job_id]


_sizes = {}
# job_id -> (key, per-frame pixel -> metres mappings, calibration)
_mapped = {}
_mapped_lock = threading.Lock()


def _load_calibration(job_id):
    try:
        with open(_calib_path(job_id)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _keyframes(calib):
    """{frame: homography} from the saved marked points (raises
    pitch.CalibrationError if a view doesn't work)."""
    across, along = pitch.lines_across(calib["length"]), pitch.lines_along(calib["width"])
    views = {}
    for p in calib["points"]:
        views.setdefault(p["f"], []).append((p["x"], p["y"], across[p["across"]][1], along[p["along"]][1]))
    return {f: pitch.solve(pts) for f, pts in views.items()}


def _mappings(job_id, job):
    """(maps, calibration) for a video whose pitch is marked and camera
    followed, else (None, calibration-or-None)."""
    calib = _load_calibration(job_id)
    if not calib or not calib.get("points") or not os.path.exists(_motion_path(job_id)):
        return None, calib
    key = (os.path.getmtime(_calib_path(job_id)), os.path.getmtime(_motion_path(job_id)))
    frames = _prepared_frames(job_id, job["input"])
    with _mapped_lock:
        cached = _mapped.get(job_id)
        if cached and cached[0] == key:
            return cached[1], calib
        try:
            keyframes = _keyframes(calib)
        except (pitch.CalibrationError, KeyError, TypeError):
            return None, calib
        maps = pitch.frame_mappings(len(frames), np.load(_motion_path(job_id)), keyframes)
        maps = pitch.check_with_players(maps, frames, calib["length"], calib["width"])
        _mapped.clear()  # one video in memory
        _mapped[job_id] = (key, maps)
        return maps, calib


def _hex_colors(job):
    return {b: "#%02x%02x%02x" % tuple(bgr[::-1]) for b, bgr in (job.get("colors") or {}).items()}


def _main_teams(job):
    """The two biggest colour buckets = the two teams."""
    counts = {b: n for b, n in (job["teams"] or {}).items() if b not in ("ball", "unknown", "other", "unsure")}
    return sorted(counts, key=counts.get, reverse=True)[:2]


@app.route("/report/<job_id>")
def report(job_id):
    """
    Coaching report. ?team=<bucket>|both
      single team: &phase=attack|defence|mixed
      both:        &attacker=<bucket>|none
    """
    job = _job_or_404(job_id)
    if job["status"] != "done":
        abort(404)
    if not os.path.exists(_detections_path(job_id)):
        return jsonify(error="This video was analysed before reports existed. Please analyse it again."), 409
    if report_requests.wait(accounts.current_user()):
        return jsonify(error="Too many reports at once. Wait a minute and try again."), 429
    report_requests.hit(accounts.current_user())

    main = _main_teams(job)
    if len(main) < 2:
        return jsonify(error="Couldn't find two teams in this video."), 409
    a, b = main
    team = request.args.get("team", "both")

    if team == "both":
        attacker = request.args.get("attacker", "none")
        if attacker == a:
            plan = [(a, b, "attack"), (b, a, "defence")]
        elif attacker == b:
            plan = [(b, a, "attack"), (a, b, "defence")]
        else:
            plan = [(a, b, "mixed"), (b, a, "mixed")]
    else:
        if team not in main:
            return jsonify(error=f"Unknown team '{team}'."), 400
        phase = request.args.get("phase", "mixed")
        if phase not in ("attack", "defence", "mixed"):
            phase = "mixed"
        plan = [(team, b if team == a else a, phase)]

    fps, _ = video_info(job["input"])
    frames = _prepared_frames(job_id, job["input"])
    sections = [analysis.analyse_team(frames, fps, t, o, p) for t, o, p in plan]

    # Rugby layer: breakdowns (always) and line speed (pitch marked).
    maps, calib = _mappings(job_id, job)
    events = rugby.breakdowns(frames, fps, maps, _sizes.get(job_id))
    speeds = rugby.line_speeds(frames, fps, maps, events, main)
    order = {"issue": 0, "info": 1, "good": 2}
    for sec in sections:
        point = rugby.line_speed_point(speeds, sec["team"], sec["opponent"], sec["phase"], fps)
        if point:
            sec["stats"]["line_speed"] = point.pop("speed")
            sec["points"].append(point)
            sec["points"].sort(key=lambda p: order[p["kind"]])
    wide = sum(1 for f in frames if f[0])
    mapped = sum(1 for m in (maps or []) if m is not None)
    pitch_info = {
        "marked": bool(calib and calib.get("points")),
        "measuring": bool(calib and calib.get("points")) and maps is None,
        "coverage": round(100 * mapped / wide) if maps and wide else 0,
    }
    return jsonify(fps=fps, teams=main, sections=sections, breakdowns=rugby.breakdown_summary(events, fps),
                   pitch=pitch_info)


def _video_size(job):
    cap = cv2.VideoCapture(job["input"])
    ok, first = cap.read()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return ((first.shape[1], first.shape[0]) if ok else (0, 0)), count, fps


@app.route("/calibration/<job_id>", methods=["GET", "POST"])
def calibration(job_id):
    """The coach's marked pitch points for this video (see pitch.py)."""
    job = _job_or_404(job_id)
    if not os.path.exists(_detections_path(job_id)):
        return jsonify(error="This video was analysed before pitch marking existed. Please analyse it again."), 409
    if request.method == "GET":
        calib = _load_calibration(job_id) or {}
        (w, h), count, fps = _video_size(job)
        length = calib.get("length", pitch.DEFAULT_LENGTH)
        width = calib.get("width", pitch.DEFAULT_WIDTH)
        return jsonify(length=length, width=width, points=calib.get("points", []), size=[w, h], frames=count, fps=fps,
                       across=[[k, v[0]] for k, v in pitch.lines_across(length).items()],
                       along=[[k, v[0]] for k, v in pitch.lines_along(width).items()],
                       followed=os.path.exists(_motion_path(job_id)))

    user = accounts.current_user()
    with jobs_lock:
        if job["status"] != "done":
            return jsonify(error="This video is still being processed. Try again when it's finished."), 409
    body = request.get_json(silent=True, force=True)
    if not isinstance(body, dict):
        return jsonify(error="Bad request."), 400
    if body.get("clear"):
        try:
            os.remove(_calib_path(job_id))
        except OSError:
            pass
        return jsonify(ok=True, cleared=True)
    if calibrations_per_hour.wait(user):
        return jsonify(error="You've saved the pitch marking a lot in the last hour. Try again later."), 429
    try:
        length = float(body.get("length", pitch.DEFAULT_LENGTH))
        width = float(body.get("width", pitch.DEFAULT_WIDTH))
    except (TypeError, ValueError):
        return jsonify(error="Pitch length and width must be numbers."), 400
    if not (40 <= length <= 100 and 25 <= width <= 70):
        return jsonify(error="Pitch length must be 40-100 m (try line to try line) and width 25-70 m."), 400
    (w, h), count, _ = _video_size(job)
    across, along = pitch.lines_across(length), pitch.lines_along(width)
    raw = body.get("points")
    if not isinstance(raw, list) or not 4 <= len(raw) <= 40:
        return jsonify(error="Mark at least 4 points (and at most 40)."), 400
    points = []
    for p in raw:
        try:
            pt = {"f": int(p["f"]), "x": float(p["x"]), "y": float(p["y"]),
                  "across": str(p["across"]), "along": str(p["along"])}
        except (KeyError, TypeError, ValueError):
            return jsonify(error="A marked point is incomplete."), 400
        if not (0 <= pt["f"] < max(count, 1) and 0 <= pt["x"] <= w and 0 <= pt["y"] <= h):
            return jsonify(error="A marked point is outside the video."), 400
        if pt["across"] not in across or pt["along"] not in along:
            return jsonify(error="Pick a line across and a line along the pitch for every point."), 400
        points.append(pt)
    if len({p["f"] for p in points}) > 8:
        return jsonify(error="Use at most 8 different frames."), 400
    calib = {"length": length, "width": width, "points": points}
    try:
        _keyframes(calib)
    except pitch.CalibrationError as e:
        frame_views = {}
        for p in points:
            frame_views.setdefault(p["f"], 0)
            frame_views[p["f"]] += 1
        few = [f for f, n in frame_views.items() if n < 4]
        if few:
            return jsonify(error=f"The frame at {_clock(few[0] / (_video_size(job)[2] or 25))} has fewer than 4 points. "
                                 "Each frame you mark needs at least 4 (or remove its points)."), 400
        return jsonify(error=str(e)), 400
    calibrations_per_hour.hit(user)
    tmp = _calib_path(job_id) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(calib, f)
    os.replace(tmp, _calib_path(job_id))
    if os.path.exists(_motion_path(job_id)):
        return jsonify(ok=True, queued=False)
    with jobs_lock:
        job.update(status="queued", message="Waiting in queue...")
    job_queue.put(("motion", job_id, None))
    return jsonify(ok=True, queued=True)


def _clock(t):
    t = int(t)
    return f"{t // 60}:{t % 60:02d}"


@app.route("/pitch/<job_id>")
def pitch_view(job_id):
    """Players on a top-down pitch at one frame, to check the marking."""
    job = _job_or_404(job_id)
    if job["status"] != "done" or not os.path.exists(_detections_path(job_id)):
        abort(404)
    try:
        frame_no = min(10_000_000, max(0, int(request.args.get("f", 0))))
    except ValueError:
        abort(400)
    maps, calib = _mappings(job_id, job)
    frames = _prepared_frames(job_id, job["input"])
    H = maps[frame_no] if maps and frame_no < len(maps) else None
    out = {"length": (calib or {}).get("length", pitch.DEFAULT_LENGTH),
           "width": (calib or {}).get("width", pitch.DEFAULT_WIDTH), "mapped": H is not None, "players": []}
    if H is not None:
        players = [p for p in frames[frame_no][0] if p["bucket"] not in ("other", "unsure")]
        xy = pitch.project(H, [(p["x"], p["y"]) for p in players])
        out["players"] = [{"x": round(float(x), 1), "y": round(float(y), 1), "team": p["bucket"]}
                          for p, (x, y) in zip(players, xy)]
    return jsonify(out)


@app.route("/snapshot/<job_id>")
def snapshot(job_id):
    """One frame of the original video with the chosen team's boxes and the
    moment's highlight (red) drawn on. ?f=frame&team=&opp=&hl=..."""
    job = _job_or_404(job_id)
    if not os.path.exists(_detections_path(job_id)):
        abort(404)
    try:
        frame_no = min(10_000_000, max(0, int(request.args.get("f", 0))))
    except ValueError:
        abort(400)
    team, opp = request.args.get("team", ""), request.args.get("opp", "")

    cap = cv2.VideoCapture(job["input"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        abort(404)

    frames = _prepared_frames(job_id, job["input"])
    team_color = tuple((job.get("colors") or {}).get(team) or BUCKET_COLORS.get(team, (0, 210, 100)))
    if 0 <= frame_no < len(frames):
        for p in frames[frame_no][0]:
            x1, y1, x2, y2 = p["box"]
            if p["bucket"] == team:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)
                cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 2)
            elif p["bucket"] == opp:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (160, 160, 160), 1)

    red = (0, 0, 255)
    for part in request.args.get("hl", "")[:2000].split(";")[:40]:
        kind, _, nums = part.partition(":")
        try:
            v = [int(float(n)) for n in nums.split(",")]
        except ValueError:
            continue
        if len(v) != 4:
            continue
        if kind == "line":
            cv2.line(frame, (v[0], v[1]), (v[2], v[3]), red, 3)
        elif kind == "box":
            pad = 6
            cv2.rectangle(frame, (v[0] - pad, v[1] - pad), (v[2] + pad, v[3] + pad), red, 3)

    return _jpeg(frame)


def _read_frame(path, frame_no):
    cap = cv2.VideoCapture(path)
    if frame_no > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _jpeg(frame, max_width=960, max_age=0):
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    resp = app.response_class(jpg.tobytes(), mimetype="image/jpeg")
    if max_age:
        resp.headers["Cache-Control"] = f"public, max-age={max_age}"
    return resp


@app.route("/frame/<job_id>")
def frame(job_id):
    """One raw frame of an uploaded video, used by the live processing preview.
    ?f=frame&w=width&boxes=1 draws every player in their team colour once the
    teams are known."""
    job = _job_or_404(job_id)
    try:
        frame_no = min(10_000_000, max(0, int(request.args.get("f", 0))))
        width = min(960, max(120, int(request.args.get("w", 480))))
    except ValueError:
        abort(400)
    img = _read_frame(job["input"], frame_no)
    if img is None:
        abort(404)
    # Shrink first and draw after, so the boxes stay visible in a thumbnail.
    scale = min(1.0, width / img.shape[1])
    if scale < 1:
        img = cv2.resize(img, (width, int(img.shape[0] * scale)))

    colors = job.get("colors")
    if request.args.get("boxes") == "1" and colors and os.path.exists(_detections_path(job_id)):
        frames = _prepared_frames(job_id, job["input"])
        hidden = set(job.get("hidden") or [])
        if frame_no < len(frames):
            for p in frames[frame_no][0]:
                color = colors.get(p["bucket"]) or BUCKET_COLORS.get(p["bucket"])
                if color is None or p["bucket"] in hidden:
                    continue
                x1, y1, x2, y2 = (int(v * scale) for v in p["box"])
                cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c) for c in color), 2)
    return _jpeg(img, width, max_age=3600)


def _demo_job():
    """Your most recently finished analysis - its footage is the landing page
    demo. Only ever your own video, never someone else's."""
    me = accounts.current_user()
    done = [j for j in list(jobs.values())
            if j.get("owner") == me and j.get("status") == "done" and os.path.exists(j.get("output") or "")
            and os.path.exists(j.get("input") or "")]
    return max(done, key=lambda j: os.path.getmtime(j["output"]), default=None)


@app.route("/demo/<kind>.jpg")
def demo(kind):
    """Matching raw and tracked frames for the before/after slider on the
    landing page. static/demo/raw.jpg + tracked.jpg win if they exist;
    otherwise a frame from your latest analysed video is used."""
    if kind not in ("raw", "tracked"):
        abort(404)
    fixed = os.path.join(DEMO_DIR, f"{kind}.jpg")
    if os.path.exists(os.path.join(DEMO_DIR, "raw.jpg")) and os.path.exists(os.path.join(DEMO_DIR, "tracked.jpg")):
        return send_file(fixed, mimetype="image/jpeg", max_age=300)

    job = _demo_job()
    if not job:
        abort(404)
    cap = cv2.VideoCapture(job["output"])
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    frame_no = count // 3  # a third in: past any kick-off close-ups
    img = _read_frame(job["input"] if kind == "raw" else job["output"], frame_no)
    if img is None:
        abort(404)
    return _jpeg(img, 1280, max_age=300)


@app.route("/video/<job_id>")
def video(job_id):
    job = _job_or_404(job_id)
    if job["status"] != "done":
        abort(404)
    as_download = request.args.get("download") == "1"
    return send_file(
        job["output"],
        mimetype="video/mp4",
        as_attachment=as_download,
        download_name=job["filename"],
        conditional=True,  # lets the browser seek within the video
    )


def lan_ip():
    # Finds the address other devices on your Wi-Fi use to reach this laptop.
    # (Connecting a UDP socket sends nothing; it just picks the right network card.)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return None


def _uploads_overview():
    """All videos with who uploaded them, for the owner's moderation list."""
    with jobs_lock:
        return sorted(({"id": jid, "owner": j.get("owner"), "name": j.get("original_name") or j.get("filename"),
                        "uploaded": j.get("uploaded"), "status": j["status"]} for jid, j in jobs.items()),
                      key=lambda u: u["uploaded"] or 0, reverse=True)


def _delete_user_videos(user_key):
    """When an account is removed, its videos go too."""
    with jobs_lock:
        theirs = [jid for jid, j in jobs.items() if j.get("owner") == user_key and j["status"] in ("done", "error")]
    for jid in theirs:
        _delete_job(jid)


accounts.uploads_overview = _uploads_overview
accounts.usage_for = _usage
accounts.delete_user_videos = _delete_user_videos
accounts.delete_video = _delete_job

_load_saved_jobs()
threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=_purge_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("YOAC_PORT", "5000"))
    # 0.0.0.0 = reachable from other devices on the same Wi-Fi. Online, behind
    # a web server that handles HTTPS, set YOAC_HOST=127.0.0.1.
    host = os.environ.get("YOAC_HOST", "0.0.0.0")
    ip = lan_ip()
    print("\n" + "=" * 56)
    print(f"  On this laptop:  http://127.0.0.1:{port}")
    if ip:
        print(f"  On your phone:   http://{ip}:{port}")
        print("  (phone must be on the same Wi-Fi as this laptop)")
    print("=" * 56 + "\n")
    try:
        from waitress import serve  # production-grade server, if installed
    except ImportError:
        app.run(host=host, port=port, debug=False, threaded=True)
    else:
        serve(app, host=host, port=port, threads=8, connection_limit=200, channel_timeout=120,
              max_request_body_size=app.config["MAX_CONTENT_LENGTH"] + 1024 * 1024)
