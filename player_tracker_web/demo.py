"""
What the front page shows about a real analysed video: one frame from it,
crops of that frame, and numbers measured on it. Nothing here is made up -
if there's no analysed video, the page says so instead.

Used by app.py for the logged-in coach's own latest video. To put a fixed
example on the front page for every account, run once:

    python demo.py JOB_ID

which writes static/demo/raw.jpg, tracked.jpg and demo.json from that job.
"""

import json
import os
import sys
from statistics import median

TEAM_SKIP = ("ball", "other", "unsure", "unknown")
MAX_HIDDEN = 90        # frames; matches track_buffer in tracker.py
TOUCH = 0.6            # feet closer than this many box heights = one group
MIN_GROUP = 4          # players in one group = a breakdown worth cropping
GREY = (136, 136, 136) # box colour for a bucket without a kit colour
PER_TEAM = 15          # players a side on the pitch (rugby union)


def frame_index(frame_count):
    """The frame shown on the front page: a third in, past any kick-off close-ups."""
    return frame_count // 3


def _hex(bgr):
    return "#%02x%02x%02x" % tuple(int(c) for c in bgr[::-1])


def _ink(bgr):
    """Label text colour on a kit colour - the rule tracker.py draws with."""
    return "#ffffff" if sum(bgr) < 380 else "#0a0a0a"


def _dark(bgr):
    """Dark enough that tracker.draw_box gives the box a light edge."""
    return 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2] < 70


def facts(detections, fps, colors=None):
    """Numbers measured on one analysed video. `detections` is the saved
    per-frame list of [x1, y1, x2, y2, bucket, track_id, conf]."""
    seen = {}      # track id -> frames it was found in
    bucket_of = {}
    in_view = []
    for f, frame in enumerate(detections):
        n = {}
        for x1, y1, x2, y2, bucket, tid, conf in frame:
            if tid is None or bucket == "ball":
                continue
            seen.setdefault(tid, []).append(f)
            bucket_of[tid] = bucket
            if bucket not in TEAM_SKIP:
                n[bucket] = n.get(bucket, 0) + 1
        # 15 a side: anyone past that is a replacement, staff or crowd, not a player.
        in_view.append(sum(min(v, PER_TEAM) for v in n.values()))

    per_team, officials = {}, 0
    longest_track, hidden_runs = 0, []
    for tid, frames in seen.items():
        b = bucket_of[tid]
        if b == "other":
            officials += 1
        elif b not in TEAM_SKIP:
            per_team[b] = per_team.get(b, 0) + 1
        longest_track = max(longest_track, frames[-1] - frames[0] + 1)
        for a, c in zip(frames, frames[1:]):
            if 1 < c - a <= MAX_HIDDEN + 1:
                hidden_runs.append(c - a - 1)

    teams = sorted(per_team, key=per_team.get, reverse=True)[:2]
    kit = lambda t: (colors or {}).get(t, GREY)
    return {
        "frames": len(detections),
        "fps": round(fps, 2),
        "seconds": round(len(detections) / fps, 1) if fps else 0,
        "teams": [{"name": t, "tracks": per_team[t], "color": _hex(kit(t)), "ink": _ink(kit(t)), "dark": _dark(kit(t))}
                  for t in teams],
        "officials": officials,
        "median_in_view": int(median(in_view)) if in_view else 0,
        "max_in_view": max(in_view, default=0),
        "longest_track": longest_track,
        "recovered": len(hidden_runs),
        "longest_hidden": max(hidden_runs, default=0),
    }


def _pad(box, size, pad, aspect=None):
    """Box grown by `pad` (share of its size) and widened to `aspect`, kept
    inside the picture, as fractions of the frame."""
    W, H = size
    x1, y1, x2, y2 = box
    w, h = (x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad)
    if aspect:
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
    w, h = min(w, W), min(h, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(0, cx - w / 2), W - w)
    top = min(max(0, cy - h / 2), H - h)
    return {"x": round(left / W, 4), "y": round(top / H, 4), "w": round(w / W, 4), "h": round(h / H, 4)}


def crops(frame, size, hidden=(), colors=None):
    """Parts of one frame worth a close look: the biggest group of players
    (usually a breakdown), the nearest player of each team, and an official
    if one is in view (they aren't boxed in the tracked video)."""
    people = [d for d in frame if d[4] != "ball" and d[5] is not None]
    out = []

    # Group players whose feet touch (same idea as analysis.py's contact groups).
    team_people = [d for d in people if d[4] not in TEAM_SKIP and d[4] not in hidden]
    parent = list(range(len(team_people)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(team_people):
        for j in range(i + 1, len(team_people)):
            b = team_people[j]
            reach = TOUCH * max(a[3] - a[1], b[3] - b[1])
            if abs((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2) < reach * 1.5 and abs(a[3] - b[3]) < reach:
                parent[root(i)] = root(j)
    groups = {}
    for i in range(len(team_people)):
        groups.setdefault(root(i), []).append(team_people[i])
    biggest = max(groups.values(), key=len, default=[])
    if len(biggest) >= MIN_GROUP:
        box = (min(d[0] for d in biggest), min(d[1] for d in biggest), max(d[2] for d in biggest), max(d[3] for d in biggest))
        teams_in = sorted({d[4] for d in biggest})
        out.append({"kind": "group", "image": "tracked", "players": len(biggest), "teams": teams_in,
                    **_pad(box, size, 0.25, aspect=4 / 3)})

    # The nearest (tallest) player of each team, with the label above the box.
    for team in sorted({d[4] for d in team_people}, key=lambda t: -sum(1 for d in team_people if d[4] == t))[:2]:
        d = max((p for p in team_people if p[4] == team), key=lambda p: p[3] - p[1])
        h = d[3] - d[1]
        out.append({"kind": "player", "image": "tracked", "team": team, "id": d[5],
                    "color": _hex((colors or {}).get(team, GREY)), "ink": _ink((colors or {}).get(team, GREY)),
                    **_pad((d[0], d[1] - max(0.3 * h, 30), d[2], d[3]), size, 0.35, aspect=3 / 4)})

    refs = [d for d in people if d[4] == "other"]
    if refs:
        d = max(refs, key=lambda p: p[3] - p[1])
        out.append({"kind": "official", "image": "raw" if "other" in hidden else "tracked",
                    **_pad(d[:4], size, 0.6, aspect=3 / 4)})
    return out


def summary(detections, fps, size, hidden=(), colors=None, name=""):
    """Everything the front page needs about one analysed video."""
    f = frame_index(len(detections))
    return {
        "name": name,
        "frame": f,
        "size": list(size) if size else None,
        "facts": facts(detections, fps, colors),
        "crops": crops(detections[f], size, hidden, colors) if detections and size else [],
    }


def _export(job_id):
    import cv2
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir, demo_dir = os.path.join(here, "outputs"), os.path.join(here, "static", "demo")
    with open(os.path.join(out_dir, f"{job_id}.json")) as fh:
        job = json.load(fh)
    with open(os.path.join(out_dir, f"{job_id}.detections.json")) as fh:
        detections = json.load(fh)
    cap = cv2.VideoCapture(job["input"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    f = frame_index(len(detections))
    os.makedirs(demo_dir, exist_ok=True)
    size = None
    for kind, path in (("raw", job["input"]), ("tracked", job["output"])):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Couldn't read frame {f} of {path}")
        size = (img.shape[1], img.shape[0])
        cv2.imwrite(os.path.join(demo_dir, f"{kind}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    info = summary(detections, fps, size, job.get("hidden") or [], job.get("colors"),
                   job.get("original_name") or "")
    with open(os.path.join(demo_dir, "demo.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(f"Front page example written to {demo_dir} (frame {f}).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python demo.py JOB_ID   (the id is in the address: /upload#job=JOB_ID)")
    _export(sys.argv[1])
