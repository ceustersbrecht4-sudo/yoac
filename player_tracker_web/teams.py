"""
Sort detected players into teams by shirt colour, learned from each video.

v6 names each box from fixed hue ranges, one frame at a time. That breaks on
small / far-away players, whose shirt sample is mostly grass - they all come
out "green". This module instead:
  1. ignores pixels that look like the pitch when reading a shirt colour
  2. finds the two kit colours in this video - a rugby match only ever has
     two teams. Other colour groups are either one of those teams in shade or
     floodlight (merged in) or, if small and clearly different, the match
     officials ("other")
  3. lets every shirt pixel vote for the kit colour it's closest to, and
     gives each tracked player the team that won the vote over all frames.
     A close vote still picks the likelier team rather than giving up.
It runs on the saved boxes, reading the video twice - no detection needed.
"""

import cv2
import numpy as np

PITCH_DISTANCE = 12     # colour (a/b) distance: closer than this to the pitch = grass
                        # (brightness is ignored so mowing stripes still count as grass)
MIN_SHIRT_SHARE = 0.1   # need this share of non-grass pixels to trust a sample
GREY_SAT = 40           # HSV saturation below this = white / grey / black kit
                        # (a white shirt in shade picks up a slight tint)
L_WEIGHT = 0.3          # brightness matters less than hue (shade, floodlights)
MERGE_DISTANCE = 14     # colour groups this close together are the same kit
CLUSTERS = 4            # colour groups looked for: 2 teams + shade/light + officials
REF_MAX_SHARE = 0.15    # a group bigger than this share of all players can't be
                        # the officials (referee + 2 touch judges of ~33 people),
                        # so it's a team seen in different light
REF_DISTANCE = 30       # officials wear a clearly different colour: a small group
                        # closer than this to a team is that team (e.g. in shade)
PIXEL_MATCH = 22        # a shirt pixel this close to a kit colour votes for it...
CLEAR_RATIO = 0.6       # ...if it's also this much closer to it than to the next one
REF_WIN = 0.65          # a player needs this share of the votes to count as an official
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
    if s >= GREY_SAT:
        s, v = max(s, 170), max(v, 230)
    else:
        v = 245 if v > 110 else 40  # white / black kits
    return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _color_name(bgr):
    """Same colour families as v6, for one colour."""
    h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    if s < GREY_SAT:
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

    centers, labels = _weighted_kmeans(scaled, weights, k=min(CLUSTERS, len(keys)))
    k = len(centers)
    size = np.array([weights[labels == j].sum() for j in range(k)])

    # Colour groups that are close together are one kit (shade, floodlight,
    # a muddy shirt) - join them first so a team isn't counted twice.
    group = list(range(k))

    def root(j):
        while group[j] != j:
            j = group[j]
        return j
    for a in range(k):
        for b in range(a + 1, k):
            if np.linalg.norm(centers[a] - centers[b]) < MERGE_DISTANCE:
                group[root(b)] = root(a)
    members = {}
    for j in range(k):
        members.setdefault(root(j), []).append(j)
    group_size = {g: size[m].sum() for g, m in members.items()}

    # The two biggest kits are the teams.
    ranked = sorted(members, key=lambda g: -group_size[g])
    team_groups = ranked[:2]
    teams = [max(members[g], key=lambda j: size[j]) for g in team_groups]
    remap = {j: None for j in range(k)}
    for g, t in zip(team_groups, teams):
        for j in members[g]:
            remap[j] = t
    # Any other kit is a team in different light unless it's small and far
    # from both teams - that's the officials, who go to "other".
    total_size = size.sum()
    for g in ranked[2:]:
        dists = [min(np.linalg.norm(centers[j] - centers[t]) for j in members[g]) for t in teams]
        if group_size[g] > REF_MAX_SHARE * total_size or min(dists) < REF_DISTANCE:
            for j in members[g]:
                remap[j] = teams[int(np.argmin(dists))]
        else:
            for j in members[g]:
                remap[j] = OTHER
    # A team's kit colour is the average over everything merged into it, so a
    # white kit half in shade still reads as white.
    kit = {t: np.average([centers[j] for j in range(k) if remap[j] == t], axis=0,
                         weights=[size[j] for j in range(k) if remap[j] == t])
           for t in teams}

    # Name the teams after their kit colour.
    unscale = np.array([1 / L_WEIGHT, 1, 1])
    names, colors = {}, {}
    team_bgr = {t: _lab_to_bgr(kit[t] * unscale) for t in teams}
    base = {t: _color_name(team_bgr[t]) for t in teams}
    if len(teams) == 2 and base[teams[0]] == base[teams[1]]:
        light = max(teams, key=lambda t: kit[t][0])
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
    # Every player gets one of the two teams: there are only two on the pitch.
    # Only someone who clearly wears the officials' colour is left out, and a
    # close vote goes to the likelier team rather than "unsure". Players with
    # no clear shirt pixels at all go to the team their average colour is
    # nearest to.
    nearest_team = {key: teams[int(np.argmin([np.linalg.norm(f - centers[t]) for t in teams]))]
                    for key, f in zip(keys, scaled)}
    track_team = {}
    for key in set(votes) | set(nearest_team):
        v = votes.get(key, {})
        total_votes = sum(v.values())
        ref_votes = v.get(OTHER, 0)
        team_votes = {t: v.get(t, 0) for t in teams}
        if total_votes and ref_votes >= REF_WIN * total_votes:
            track_team[key] = OTHER
        elif any(team_votes.values()):
            track_team[key] = max(team_votes, key=team_votes.get)
        elif key in nearest_team:
            track_team[key] = nearest_team[key]
        else:
            track_team[key] = UNSURE  # never seen a clean shirt sample
    for fi, dets in enumerate(new):
        for di, d in enumerate(dets):
            if d[4] == "ball":
                continue
            tid = d[5]
            key = ("t", tid) if tid is not None else ("d", fi, di)
            t = track_team.get(key)
            d[4] = names.get(t, t or UNSURE)
    return new, colors
