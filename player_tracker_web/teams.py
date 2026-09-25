"""
Sort detected players into teams by shirt colour, learned from each video.

v6 names each box from fixed hue ranges, one frame at a time. That breaks on
small / far-away players, whose shirt sample is mostly grass - they all come
out "green". This module instead:
  1. ignores pixels that look like the pitch when reading a shirt colour
  2. finds the two most common kit colours in this video (plus "other" for
     referees and anything that fits neither)
  3. lets every shirt pixel vote for the kit colour it's closest to, and
     gives each tracked player the team that won the vote over all frames
It runs on the saved boxes, reading the video twice - no detection needed.
"""

import cv2
import numpy as np

PITCH_DISTANCE = 12     # colour (a/b) distance: closer than this to the pitch = grass
                        # (brightness is ignored so mowing stripes still count as grass)
MIN_SHIRT_SHARE = 0.1   # need this share of non-grass pixels to trust a sample
L_WEIGHT = 0.3          # brightness matters less than hue (shade, floodlights)
MERGE_DISTANCE = 14     # a 3rd colour this close to a team is that team
PIXEL_MATCH = 22        # a shirt pixel this close to a kit colour votes for it...
CLEAR_RATIO = 0.6       # ...if it's also this much closer to it than to the next one
CLEAR_WIN = 0.65        # a player needs this share of the votes, else "unsure"
OTHER = "other"
UNSURE = "unsure"


def _shirt_pixels(frame, detections):
    """Non-grass pixels (Lab) from each person's shirt area in this frame, or None."""
    H, W = frame.shape[:2]
    # Pitch colour from the lower part of the picture (the top is often crowd).
    small = cv2.resize(frame[int(H * 0.4):], (64, 36), interpolation=cv2.INTER_AREA)
    pitch = np.median(cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32), axis=0)

    out = []
    for x1, y1, x2, y2, bucket, track_id, conf in detections:
        if bucket == "ball":
            out.append(None)
            continue
        w, h = x2 - x1, y2 - y1
        tx1, tx2 = max(0, int(x1 + w * 0.30)), min(W, int(x1 + w * 0.70))
        ty1, ty2 = max(0, int(y1 + h * 0.15)), min(H, int(y1 + h * 0.50))
        if tx2 - tx1 < 2 or ty2 - ty1 < 2:
            out.append(None)
            continue
        patch = cv2.cvtColor(frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        shirt = patch[np.linalg.norm(patch[:, 1:] - pitch[1:], axis=1) > PITCH_DISTANCE]
        if len(shirt) < max(4, MIN_SHIRT_SHARE * len(patch)):
            out.append(None)
            continue
        out.append(shirt)
    return out


def _each_frame(video_path, all_detections, progress, offset, total):
    cap = cv2.VideoCapture(video_path)
    try:
        for fi, detections in enumerate(all_detections):
            ok, frame = cap.read()
            if not ok:
                break
            yield fi, _shirt_pixels(frame, detections)
            if progress and fi % 20 == 0:
                progress(offset + fi, total)
    finally:
        cap.release()


def _weighted_kmeans(points, weights, k, iters=30):
    rng = np.random.default_rng(0)
    # k-means++ style start, weighted
    centers = [points[rng.choice(len(points), p=weights / weights.sum())]]
    for _ in range(1, k):
        d = np.min([np.sum((points - c) ** 2, axis=1) for c in centers], axis=0) * weights
        if d.sum() == 0:
            break
        centers.append(points[rng.choice(len(points), p=d / d.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        labels = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([
            np.average(points[labels == j], axis=0, weights=weights[labels == j])
            if np.any(labels == j) else centers[j] for j in range(len(centers))
        ])
        if np.allclose(new, centers):
            break
        centers = new
    labels = np.argmin(((points[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    return centers, labels


def _lab_to_bgr(lab):
    px = np.uint8([[np.clip(lab, 0, 255)]])
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])


def _box_color(bgr):
    """Kit colour made bright enough to see as a box outline on the pitch."""
    h, s, v = (int(x) for x in cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0])
    if s >= 25:
        s, v = max(s, 170), max(v, 230)
    else:
        v = 245 if v > 110 else 40  # white / black kits
    return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _color_name(bgr):
    """Same colour families as v6, for one colour."""
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    if s < 25:
        return "white" if v > 140 else ("black" if v < 80 else "grey")
    if h < 10 or h >= 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "purple/pink"


def assign_teams(video_path, all_detections, progress=None):
    """
    Returns (new_detections, colors): detections with the bucket replaced by
    a team name, and {team name: BGR box colour}.
    """
    # The video is read twice: once to learn the kit colours, once to vote.
    total = 2 * len(all_detections)
    key_of = lambda fi, di: (("t", all_detections[fi][di][5]) if all_detections[fi][di][5] is not None
                             else ("d", fi, di))

    # Pass 1: one typical shirt colour per detection.
    per_det = []  # (track key, feature, weight)
    for fi, pixels in _each_frame(video_path, all_detections, progress, 0, total):
        for di, shirt in enumerate(pixels):
            if shirt is not None:
                per_det.append((key_of(fi, di), np.median(shirt, axis=0), float(len(shirt))))

    new = [[list(d) for d in dets] for dets in all_detections]
    if len(per_det) < 10:
        return new, {}

    # One colour per tracked player (weighted by how much shirt was visible).
    tracks = {}
    for key, feat, w in per_det:
        tracks.setdefault(key, []).append((feat, w))
    keys = list(tracks)
    feats = np.array([np.average([f for f, _ in tracks[k]], axis=0, weights=[w for _, w in tracks[k]]) for k in keys])
    weights = np.array([sum(w for _, w in tracks[k]) for k in keys], dtype=np.float64)
    scaled = feats * np.array([L_WEIGHT, 1, 1], dtype=np.float32)

    centers, labels = _weighted_kmeans(scaled, weights, k=min(3, len(keys)))
    size = np.array([weights[labels == j].sum() for j in range(len(centers))])
    order = list(np.argsort(-size))
    teams = order[:2]
    # A small 3rd group close to a team is just that team in shade/light.
    remap = {j: j for j in range(len(centers))}
    for j in order[2:]:
        dists = [np.linalg.norm(centers[j] - centers[t]) for t in teams]
        if min(dists) < MERGE_DISTANCE:
            remap[j] = teams[int(np.argmin(dists))]

    # Name the teams after their kit colour.
    unscale = np.array([1 / L_WEIGHT, 1, 1])
    names, colors = {}, {}
    team_bgr = {t: _lab_to_bgr(centers[t] * unscale) for t in teams}
    base = {t: _color_name(team_bgr[t]) for t in teams}
    if len(teams) == 2 and base[teams[0]] == base[teams[1]]:
        light = max(teams, key=lambda t: centers[t][0])
        for t in teams:
            base[t] = ("light " if t == light else "dark ") + base[t]
    for t in teams:
        names[t] = base[t]
        colors[base[t]] = _box_color(team_bgr[t])
    colors[OTHER] = (160, 160, 160)
    colors[UNSURE] = (0, 215, 255)

    # Pass 2: every shirt pixel votes for the kit colour it is closest to
    # (pixels close to neither don't vote), and a tracked player's team is
    # the majority over all its frames. Far-away players have only a few
    # clean shirt pixels among blurred edges; averaging would lose them.
    weight = np.array([L_WEIGHT, 1, 1], dtype=np.float32)
    votes = {}
    for fi, pixels in _each_frame(video_path, all_detections, progress, len(all_detections), total):
        for di, shirt in enumerate(pixels):
            if shirt is None:
                continue
            d = np.sqrt((((shirt * weight)[:, None, :] - centers[None]) ** 2).sum(-1))
            nearest = np.argmin(d, axis=1)
            ranked = np.sort(d, axis=1)
            # Only clear pixels vote: close to a kit colour, and clearly closer
            # to it than to the other one (blurred shirt/grass edges don't).
            close = (ranked[:, 0] < PIXEL_MATCH) & (ranked[:, 0] < CLEAR_RATIO * ranked[:, 1])
            tally = votes.setdefault(key_of(fi, di), {})
            for j, n in zip(*np.unique(nearest[close], return_counts=True)):
                j = remap[int(j)]
                tally[j] = tally.get(j, 0) + int(n)
    # A player whose vote is close (e.g. a far-away dark shirt blurred into
    # the grass) is marked unsure rather than guessed - a wrong team would
    # mislead the coaching report more than a missing player does.
    track_team = {}
    for k, v in votes.items():
        if v:
            best = max(v, key=v.get)
            track_team[k] = best if v[best] >= CLEAR_WIN * sum(v.values()) else UNSURE
    for fi, dets in enumerate(new):
        for di, d in enumerate(dets):
            if d[4] == "ball":
                continue
            tid = d[5]
            key = ("t", tid) if tid is not None else ("d", fi, di)
            t = track_team.get(key)
            d[4] = UNSURE if t == UNSURE else names.get(t, OTHER)
    return new, colors
