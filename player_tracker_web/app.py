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
import socket
import threading
import time
import traceback
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import cv2

import analysis
from teams import assign_teams
from tracker import BUCKET_COLORS, count_buckets, detect_players, render_video, video_info

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(HERE, "uploads")
OUTPUT_DIR = os.path.join(HERE, "outputs")
DEMO_DIR = os.path.join(HERE, "static", "demo")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB
app.config["TEMPLATES_AUTO_RELOAD"] = True  # page edits show up without a restart

# job_id -> {status, done, total, message, teams, colors, hidden, version,
#            filename, input, output, started}
jobs = {}
jobs_lock = threading.Lock()
job_queue = queue.Queue()


def _meta_path(job_id):
    return os.path.join(OUTPUT_DIR, f"{job_id}.json")


def _detections_path(job_id):
    return os.path.join(OUTPUT_DIR, f"{job_id}.detections.json")


def _save_meta(job_id):
    # Finished jobs are written to disk so results survive a server restart.
    with jobs_lock:
        job = jobs[job_id]
        meta = {k: job.get(k) for k in ("filename", "input", "output", "teams", "colors", "hidden", "version")}
    with open(_meta_path(job_id), "w") as f:
        json.dump(meta, f)


def _load_saved_jobs():
    for path in glob.glob(os.path.join(OUTPUT_DIR, "*.json")):
        if path.endswith(".detections.json"):
            continue
        job_id = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if not os.path.exists(meta.get("output", "")):
            continue
        jobs[job_id] = dict(meta, status="done", done=0, total=0, message="Done", started=None)


def _update(job_id, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)


def _progress(job_id, message):
    # Each stage gets its own progress bar and time estimate.
    _update(job_id, message=message, done=0, total=0, started=time.time())
    return lambda done, total: _update(job_id, done=done, total=total)


def _analyze(job_id):
    job = jobs[job_id]
    detections = detect_players(job["input"], progress=_progress(job_id, "Tracking players..."))
    detections, colors = assign_teams(job["input"], detections, progress=_progress(job_id, "Sorting players into teams..."))
    with open(_detections_path(job_id), "w") as f:
        json.dump(detections, f, separators=(",", ":"))
    # Colours are known now, so the processing preview can draw boxes in team
    # colours while the final video is still being drawn.
    _update(job_id, colors=colors)
    render_video(job["input"], detections, job["output"], colors=colors, progress=_progress(job_id, "Drawing the video..."))
    _update(job_id, teams=count_buckets(detections), colors=colors, hidden=[])


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


def worker():
    # One video at a time: YOLO on a laptop already uses all the CPU/GPU it
    # can get, so running jobs in parallel would just make each one slower.
    while True:
        kind, job_id, arg = job_queue.get()
        _update(job_id, status="processing", started=time.time(), done=0, total=0,
                message="Loading model..." if kind == "analyze" else "Updating video...")
        try:
            if kind == "analyze":
                _analyze(job_id)
            else:
                _render(job_id, arg)
            _update(job_id, status="done", message="Done")
            _save_meta(job_id)
        except Exception as e:
            traceback.print_exc()
            if kind == "analyze":
                _update(job_id, status="error", message=str(e))
            else:
                # The previous video is still fine - go back to it.
                _update(job_id, status="done", message=f"Could not update the video: {e}")
        finally:
            job_queue.task_done()


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/upload")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify(error="No video uploaded."), 400

    name = secure_filename(file.filename) or "video.mp4"
    base, ext = os.path.splitext(name)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        return jsonify(error=f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}"), 400

    job_id = uuid.uuid4().hex[:12]
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext.lower()}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}.mp4")
    file.save(input_path)

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
        }
    job_queue.put(("analyze", job_id, None))
    return jsonify(job_id=job_id)


@app.route("/hide/<job_id>", methods=["POST"])
def hide(job_id):
    """Redraw the video without the given colour buckets."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            abort(404)
        if job["status"] != "done":
            return jsonify(error="This video is still being processed."), 409
        if not os.path.exists(_detections_path(job_id)):
            return jsonify(error="This video was analysed before removing colours was possible. Please analyse it again."), 409
        hidden = request.get_json(silent=True, force=True) or {}
        hidden = sorted({str(b) for b in hidden.get("hidden", []) if b in (job["teams"] or {})})
        job.update(status="queued", message="Waiting in queue...")
    job_queue.put(("render", job_id, hidden))
    return jsonify(ok=True)


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            abort(404)
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
        return _prepared[job_id]


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
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        abort(404)
    if not os.path.exists(_detections_path(job_id)):
        return jsonify(error="This video was analysed before reports existed. Please analyse it again."), 409

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
    return jsonify(fps=fps, teams=main, sections=sections)


@app.route("/snapshot/<job_id>")
def snapshot(job_id):
    """One frame of the original video with the chosen team's boxes and the
    moment's highlight (red) drawn on. ?f=frame&team=&opp=&hl=..."""
    job = jobs.get(job_id)
    if not job or not os.path.exists(_detections_path(job_id)):
        abort(404)
    try:
        frame_no = int(request.args.get("f", 0))
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
    for part in request.args.get("hl", "").split(";"):
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
    job = jobs.get(job_id)
    if not job:
        abort(404)
    try:
        frame_no = max(0, int(request.args.get("f", 0)))
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
        if frame_no < len(frames):
            for p in frames[frame_no][0]:
                color = colors.get(p["bucket"]) or BUCKET_COLORS.get(p["bucket"])
                if color is None:
                    continue
                x1, y1, x2, y2 = (int(v * scale) for v in p["box"])
                cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c) for c in color), 2)
    return _jpeg(img, width, max_age=3600)


def _demo_job():
    """The most recently finished analysis - its footage is the landing page demo."""
    done = [j for j in list(jobs.values())
            if j.get("status") == "done" and os.path.exists(j.get("output") or "")
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
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
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


_load_saved_jobs()
threading.Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    port = 5000
    ip = lan_ip()
    print("\n" + "=" * 56)
    print(f"  On this laptop:  http://127.0.0.1:{port}")
    if ip:
        print(f"  On your phone:   http://{ip}:{port}")
        print("  (phone must be on the same Wi-Fi as this laptop)")
    print("=" * 56 + "\n")
    # 0.0.0.0 = reachable from other devices on the same Wi-Fi, not just
    # this laptop. It is still not reachable from the internet.
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
