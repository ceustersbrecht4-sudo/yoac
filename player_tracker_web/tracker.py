"""
Detection + tracking logic from track_players_v6.py, split into functions
the web app can call. The detection logic is unchanged from v6; the
differences are:
- detect_players() returns the boxes instead of drawing them, so the video
  can be (re)drawn later - after teams.py has sorted players into teams, or
  with some colours hidden
- the output is re-encoded to H.264 so browsers can play it inline
  (OpenCV's "mp4v" codec downloads fine but won't play in a <video> tag)
"""

import os
import subprocess

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

CUSTOM_TRACKER_YAML = os.path.join(HERE, "custom_bytetrack.yaml")
PERSON_CLASS = 0
BALL_CLASS = 32
PERSON_CONF_MIN = 0.35
BALL_CONF_MIN = 0.10


def _find_model():
    # Reuse the yolov8m.pt already sitting next to track_players_v6.py so it
    # isn't downloaded again; otherwise ultralytics downloads it on first use.
    for candidate in (os.path.join(HERE, "yolov8m.pt"), os.path.join(HERE, "..", "yolov8m.pt")):
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return "yolov8m.pt"


MODEL_PATH = _find_model()


def write_custom_tracker_config(path):
    content = """tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 90
match_thresh: 0.8
fuse_score: True
"""
    with open(path, "w") as f:
        f.write(content)


def classify_team(frame, x1, y1, x2, y2):
    # Sample the torso area only: horizontally centered, vertically
    # between shoulders and waist - avoids head (skin tone) and legs.
    w, h = x2 - x1, y2 - y1
    tx1 = int(x1 + w * 0.25)
    tx2 = int(x1 + w * 0.75)
    ty1 = int(y1 + h * 0.20)
    ty2 = int(y1 + h * 0.50)
    tx1, ty1 = max(0, tx1), max(0, ty1)
    tx2, ty2 = min(frame.shape[1], tx2), min(frame.shape[0], ty2)

    if tx2 <= tx1 or ty2 <= ty1:
        return "unknown", (0, 210, 100)

    patch = frame[ty1:ty2, tx1:tx2]
    if patch.size == 0:
        return "unknown", (0, 210, 100)

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    s_vals, v_vals = hsv[:, :, 1], hsv[:, :, 2]

    avg_sat = float(np.mean(s_vals))
    avg_val = float(np.mean(v_vals))

    # Low saturation = white/black/grey kit, hue is meaningless there.
    if avg_sat < 40:
        if avg_val > 140:
            return "white", (200, 200, 200)
        else:
            return "black", (40, 40, 40)

    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    dominant_hue = int(np.argmax(hist))

    if dominant_hue < 10 or dominant_hue >= 170:
        return "red", (0, 0, 220)
    elif dominant_hue < 25:
        return "orange", (0, 140, 255)
    elif dominant_hue < 35:
        return "yellow", (0, 220, 220)
    elif dominant_hue < 85:
        return "green", (0, 180, 0)
    elif dominant_hue < 130:
        return "blue", (220, 100, 0)
    else:
        return "purple/pink", (200, 0, 200)


def _to_browser_mp4(src, dst):
    """Re-encode to H.264 + faststart so it plays in a browser. Returns True on success."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", src,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-crf", "26", "-movflags", "+faststart", dst,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(dst)


# Box colour (BGR) per bucket - same colours classify_team returns.
BUCKET_COLORS = {
    "white": (200, 200, 200), "black": (40, 40, 40), "red": (0, 0, 220),
    "orange": (0, 140, 255), "yellow": (0, 220, 220), "green": (0, 180, 0),
    "blue": (220, 100, 0), "purple/pink": (200, 0, 200), "unknown": (0, 210, 100),
    "ball": (0, 140, 255),
}


def draw_detections(frame, detections, hidden=(), colors=None):
    """Draw one frame's detections, skipping any bucket in `hidden`.
    Each detection is [x1, y1, x2, y2, bucket, track_id, conf]. `colors`
    maps bucket -> BGR (the teams' real kit colours); falls back to v6's."""
    for x1, y1, x2, y2, bucket, track_id, conf in detections:
        if bucket in hidden:
            continue
        color = tuple((colors or {}).get(bucket) or BUCKET_COLORS.get(bucket, (0, 210, 100)))
        if bucket == "ball":
            label = f"Ball ({conf:.2f})"
        else:
            label = f"{bucket.capitalize()} {track_id}" if track_id is not None else bucket.capitalize()

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        text_color = (255, 255, 255) if sum(color) < 380 else (10, 10, 10)
        cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


def count_buckets(all_detections):
    counts = {}
    for detections in all_detections:
        for d in detections:
            counts[d[4]] = counts.get(d[4], 0) + 1
    return counts


class _VideoOut:
    """Collects frames into an mp4v file, then converts it for browsers.

    The writer is created from the first frame's actual size rather than
    the file's reported width/height: phone videos filmed upright are
    stored sideways plus a rotation flag, so the two can disagree."""

    def __init__(self, output_path, fps):
        self.output_path = output_path
        self.raw_path = output_path + ".raw.mp4"
        self.fps = fps
        self.writer = None
        self.frames = 0

    def write(self, frame):
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.raw_path, fourcc, self.fps, (width, height))
        self.writer.write(frame)
        self.frames += 1

    def release(self):
        if self.writer is not None:
            self.writer.release()

    def finish(self):
        if self.frames == 0:
            raise ValueError("No frames could be read from this video.")
        # Browser-friendly copy; fall back to the raw mp4v file (downloadable,
        # but most browsers won't play it inline) if ffmpeg isn't available.
        if _to_browser_mp4(self.raw_path, self.output_path):
            os.remove(self.raw_path)
        else:
            os.replace(self.raw_path, self.output_path)


def video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the video file - is it a valid video?")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def detect_players(video_path, progress=None):
    """
    Run v6 detection + tracking on video_path. progress(done_frames,
    total_frames) is called periodically. Returns the detections for every
    frame: [x1, y1, x2, y2, bucket, track_id, conf], where bucket is v6's
    colour name (teams.assign_teams replaces it with a proper team later).
    """
    write_custom_tracker_config(CUSTOM_TRACKER_YAML)

    # Fresh model per job so tracker IDs from a previous upload don't leak in
    # (persist=True keeps tracker state on the model object).
    model = YOLO(MODEL_PATH)

    fps, total_frames = video_info(video_path)
    all_detections = []

    results = model.track(
        source=video_path,
        classes=[PERSON_CLASS, BALL_CLASS],
        tracker=CUSTOM_TRACKER_YAML,
        conf=BALL_CONF_MIN,
        imgsz=1280,
        stream=True,
        persist=True,
        verbose=False,
    )

    for result in results:
        frame = result.orig_img
        detections = []

        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            track_ids = (
                result.boxes.id.cpu().numpy().astype(int)
                if result.boxes.id is not None
                else [None] * len(boxes)
            )

            for box, cls, conf, track_id in zip(boxes, classes, confs, track_ids):
                x1, y1, x2, y2 = (int(v) for v in box)

                if cls == PERSON_CLASS:
                    if conf < PERSON_CONF_MIN:
                        continue
                    bucket, _ = classify_team(frame, x1, y1, x2, y2)
                elif cls == BALL_CLASS:
                    bucket = "ball"
                else:
                    continue

                tid = int(track_id) if track_id is not None else None
                detections.append([x1, y1, x2, y2, bucket, tid, round(float(conf), 2)])

        all_detections.append(detections)
        if progress and len(all_detections) % 10 == 0:
            progress(len(all_detections), total_frames)

    if not all_detections:
        raise ValueError("No frames could be read from this video.")
    if progress:
        progress(len(all_detections), total_frames)
    return all_detections


def render_video(video_path, all_detections, output_path, hidden=(), colors=None, progress=None):
    """Draw the saved detections onto the video, leaving out hidden buckets.
    Much faster than detection - no model involved."""
    fps, total_frames = video_info(video_path)
    total = min(total_frames, len(all_detections)) if total_frames else len(all_detections)
    out = _VideoOut(output_path, fps)
    hidden = set(hidden)

    cap = cv2.VideoCapture(video_path)
    try:
        for detections in all_detections:
            ok, frame = cap.read()
            if not ok:
                break
            draw_detections(frame, detections, hidden, colors)
            out.write(frame)
            if progress and out.frames % 10 == 0:
                progress(out.frames, total)
    finally:
        cap.release()
        out.release()

    if progress:
        progress(out.frames, total)
    out.finish()
