"""
Carries into contact: how the ball carrier goes into each tackle.

Close-ups are fine here (better, even): everything is measured relative to
the players themselves - body lengths and body angles - so zoom doesn't
matter. For each contact between an attacker and a defender:

- approach, in the last half second: running at space (the gap between two
  defenders, or a defender's edge) or straight at a defender's chest;
  footwork (a change of direction just before contact); speeding up or
  slowing down into contact.
- body at contact, from a pose model (shoulders, hips, knees, ankles):
  low (shoulders well below standing height) or upright; and whether the
  carrier leads with the shoulder (side-on) or goes in square.
- leg drive: the carrier's feet still moving forward in the second after
  contact.

Coaching points behind it: a low, strong body position; contact with the
shoulder and arms; footwork to get to the edge of the defender; driving the
legs through contact.

One camera sees everything in the picture plane, so angles are
approximate. "Side-on or square" can only be judged when the carrier runs
across the picture or towards/away from the camera; otherwise it's left
out. Everything here is a pointer to moments to rewatch.
"""

import math

TEAM_SKIP = ("ball", "other", "unsure", "unknown")
TOUCH = 0.5            # feet this many body lengths apart (or boxes overlapping) = contact
FRESH = 0.5            # s the pair must have been apart before, for a new contact
APPROACH = 0.5         # s before contact that the run-in is measured over
MERGE = 1.0            # s: a second tackler joining within this is the same contact
STRAIGHT = 12.0        # degrees: heading within this of the defender = straight at them
EDGE = 20.0            # degrees: heading this far off the defender = at their edge
FOOTWORK = 20.0        # degrees of change in direction in the run-in = footwork
FASTER, SLOWER = 1.1, 0.8   # speed ratio late vs early in the run-in
LOW, UPRIGHT = 0.68, 0.76   # shoulder height at contact, share of standing height
SHOULDER_WIDE, SHOULDER_NARROW = 0.55, 0.35  # shoulder span / torso length
DRIVE = 0.5            # body lengths the feet keep moving forward after contact
KP_CONF = 0.4          # keypoint confidence to use a joint


def _people(frame):
    return [d for d in frame if d[4] not in TEAM_SKIP and d[5] is not None]


def _foot(d):
    return ((d[0] + d[2]) / 2, d[3])


def _h(d):
    return max(1.0, d[3] - d[1])


def _touching(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    overlap = ix > 0 and iy > 0 and ix * iy > 0.1 * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    (ax, ay), (bx, by) = _foot(a), _foot(b)
    return overlap or math.hypot(ax - bx, ay - by) < TOUCH * (_h(a) + _h(b)) / 2


def find_contacts(detections, fps, carrier_team):
    """Moments a `carrier_team` player first touches an opponent after being
    apart. Returns [{f, carrier, tackler}] - track ids."""
    fresh = max(2, int(FRESH * fps))
    last_touch = {}   # (carrier, tackler) -> last frame they touched
    last_start = {}   # carrier -> frame of their last contact start
    out = []
    for f, frame in enumerate(detections):
        people = _people(frame)
        mine = [d for d in people if d[4] == carrier_team]
        theirs = [d for d in people if d[4] != carrier_team]
        for c in mine:
            for t in theirs:
                if not _touching(c, t):
                    continue
                key = (c[5], t[5])
                new = f - last_touch.get(key, -10 ** 9) > fresh
                last_touch[key] = f
                if new and f - last_start.get(c[5], -10 ** 9) > MERGE * fps:
                    last_start[c[5]] = f
                    out.append({"f": f, "carrier": c[5], "tackler": t[5]})
    return out


def _track(detections, tid, f0, f1):
    """{frame: detection} for one track id between f0 and f1."""
    out = {}
    for f in range(max(0, f0), min(len(detections), f1 + 1)):
        for d in detections[f]:
            if d[5] == tid:
                out[f] = d
                break
    return out


def _angle(ax, ay, bx, by):
    """Angle between two vectors, degrees."""
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if not na or not nb:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))))


def approach(detections, fps, c):
    """Run-in facts for one contact: heading vs the defenders, footwork, speed."""
    f0, n = c["f"], max(4, int(APPROACH * fps))
    run = _track(detections, c["carrier"], f0 - n, f0)
    if len(run) < n * 0.6:
        return None
    fs = sorted(run)
    h = sum(_h(run[f]) for f in fs) / len(fs)
    p = {f: _foot(run[f]) for f in fs}
    early, mid, late = fs[0], fs[len(fs) // 2], fs[-1]
    vx, vy = p[late][0] - p[mid][0], p[late][1] - p[mid][1]          # heading just before contact
    ex, ey = p[mid][0] - p[early][0], p[mid][1] - p[early][1]
    t_late, t_early = max(1, late - mid) / fps, max(1, mid - early) / fps
    speed_late, speed_early = math.hypot(vx, vy) / h / t_late, math.hypot(ex, ey) / h / t_early
    turn = _angle(ex, ey, vx, vy)
    # Defenders ahead of the carrier at the start of the run-in's second half.
    cx, cy = p[mid]
    frame = detections[mid]
    ahead = []
    for d in _people(frame):
        if d[4] == run[mid][4]:
            continue
        dx, dy = _foot(d)[0] - cx, _foot(d)[1] - cy
        if math.hypot(dx, dy) <= 3 * h and dx * vx + dy * vy > 0:
            ahead.append((math.hypot(dx, dy), dx, dy, d[5]))
    ahead.sort()
    heading = None
    if ahead and math.hypot(vx, vy) > 0.05 * h:
        _, dx, dy, _ = ahead[0]
        a1 = _angle(vx, vy, dx, dy)
        ag = None
        if len(ahead) > 1:
            gx, gy = (ahead[0][1] + ahead[1][1]) / 2, (ahead[0][2] + ahead[1][2]) / 2
            ag = _angle(vx, vy, gx, gy)
        if a1 is not None:
            if a1 < STRAIGHT and (ag is None or ag >= a1):
                heading = "straight"
            elif ag is not None and ag < a1:
                heading = "gap"
            elif a1 >= EDGE:
                heading = "edge"
            else:
                heading = "straight"
    ratio = speed_late / speed_early if speed_early > 0.2 else None
    return {
        "heading": heading,
        "footwork": turn is not None and turn >= FOOTWORK and speed_early > 0.5,
        "speed": None if ratio is None else ("faster" if ratio >= FASTER else "slower" if ratio <= SLOWER else "same"),
        "dir": (vx, vy), "standing_h": max(_h(run[f]) for f in fs), "box": run[late][:4],
    }


def drive(detections, fps, c, dir_xy, h):
    """Body lengths the carrier's feet moved forward in the second after contact."""
    after = _track(detections, c["carrier"], c["f"], c["f"] + int(fps))
    if len(after) < fps * 0.4:
        return None
    fs = sorted(after)
    (x0, y0), (x1, y1) = _foot(after[fs[0]]), _foot(after[fs[-1]])
    n = math.hypot(*dir_xy)
    if not n:
        return None
    return ((x1 - x0) * dir_xy[0] + (y1 - y0) * dir_xy[1]) / n / h


def body(keypoints, box, standing_h, dir_xy):
    """Body height and turn from COCO keypoints [(x, y, conf) x 17] in frame
    pixels. Returns {height: low/mid/upright, turned: shoulder/square} (None
    where the joints aren't visible enough)."""
    k = lambda i: keypoints[i] if keypoints and keypoints[i][2] >= KP_CONF else None
    ls, rs, lh, rh, la, ra = k(5), k(6), k(11), k(12), k(15), k(16)
    out = {"height": None, "turned": None}
    if not (ls and rs):
        return out
    sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
    feet = [a for a in (la, ra) if a]
    ground = max(a[1] for a in feet) if feet else box[3]
    share = (ground - sy) / standing_h if standing_h else None
    if share is not None:
        out["height"] = "low" if share < LOW else "upright" if share > UPRIGHT else "mid"
    if lh and rh:
        hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
        torso = math.hypot(sx - hx, sy - hy)
        span = math.hypot(ls[0] - rs[0], ls[1] - rs[1]) / torso if torso else None
        vx, vy = dir_xy
        if span is not None and (abs(vx) > 2 * abs(vy) or abs(vy) > 2 * abs(vx)):
            across = abs(vx) > 2 * abs(vy)   # running across the picture: we see their side
            if span >= SHOULDER_WIDE:
                out["turned"] = "shoulder" if across else "square"
            elif span <= SHOULDER_NARROW:
                out["turned"] = "square" if across else "shoulder"
    return out


def analyse(detections, fps, carrier_team, pose=None, limit=200):
    """Every carry into contact by `carrier_team`. pose(frame_no, box) ->
    COCO keypoints or None (see pose_reader); without it, body height and
    turn are left out."""
    out = []
    for c in find_contacts(detections, fps, carrier_team)[:limit]:
        a = approach(detections, fps, c)
        if not a:
            continue
        h = a["standing_h"]
        rec = {"f": c["f"], "t": round(c["f"] / fps, 2), "carrier": c["carrier"], "tackler": c["tackler"],
               "heading": a["heading"], "footwork": a["footwork"], "speed": a["speed"],
               "drive": None, "height": None, "turned": None, "box": [int(v) for v in a["box"]]}
        d = drive(detections, fps, c, a["dir"], h)
        rec["drive"] = None if d is None else round(d, 2)
        if pose:
            kp = pose(c["f"], a["box"])
            rec.update(body(kp, a["box"], h, a["dir"]))
        out.append(rec)
    return out


def pose_reader(video_path, model_name="yolov8m-pose.pt"):
    """A pose(frame_no, box) function reading frames from the video and
    running the pose model on a crop around the carrier. None if the model
    can't be loaded (then body height and turn are skipped)."""
    try:
        import cv2
        from ultralytics import YOLO
        model = YOLO(model_name)
    except Exception:
        return None
    cap = cv2.VideoCapture(video_path)

    def pose(frame_no, box):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, img = cap.read()
        if not ok:
            return None
        x1, y1, x2, y2 = box
        pad_x, pad_y = (x2 - x1) * 0.35, (y2 - y1) * 0.2
        X1, Y1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
        X2, Y2 = int(min(img.shape[1], x2 + pad_x)), int(min(img.shape[0], y2 + pad_y))
        crop = img[Y1:Y2, X1:X2]
        if crop.size == 0:
            return None
        res = model.predict(crop, verbose=False, imgsz=320)[0]
        if res.keypoints is None or not len(res.keypoints):
            return None
        # The person whose box sits most centrally in the crop is the carrier.
        boxes = res.boxes.xyxy.cpu().numpy()
        cx, cy = (x1 + x2) / 2 - X1, (y1 + y2) / 2 - Y1
        best = min(range(len(boxes)), key=lambda i: math.hypot((boxes[i][0] + boxes[i][2]) / 2 - cx,
                                                                (boxes[i][1] + boxes[i][3]) / 2 - cy))
        xy = res.keypoints.xy[best].cpu().numpy()
        conf = res.keypoints.conf[best].cpu().numpy() if res.keypoints.conf is not None else [1.0] * len(xy)
        return [(float(x) + X1, float(y) + Y1, float(p)) for (x, y), p in zip(xy, conf)]

    return pose


def summary(carries, team):
    """Report points about `team`'s carries into contact."""
    T = team.capitalize()
    pts = []

    def moment(r, label):
        from rugby import _fmt_t
        return {"frame": r["f"], "t": r["t"], "time": _fmt_t(r["t"]), "label": label,
                "hl": "box:" + ",".join(str(v) for v in r["box"])}

    head = [r for r in carries if r["heading"]]
    if len(head) >= 3:
        space = [r for r in head if r["heading"] in ("gap", "edge")]
        straight = [r for r in head if r["heading"] == "straight"]
        share = len(space) / len(head)
        fw = sum(1 for r in carries if r["footwork"])
        pts.append({
            "kind": "good" if share >= 0.6 else "issue" if share < 0.4 else "info",
            "title": f"Running at space: {share:.0%} of carries",
            "detail": f"Of {len(head)} carries into contact, {len(space)} went at a gap or a defender's edge and {len(straight)} "
                      f"straight at a defender. {fw} had footwork (a change of direction) in the last half second.",
            "why": "Getting to the edge of the tackler, instead of their chest, is what wins the collision and keeps the "
                   "ball on the far side. Look at the straight ones: was there space, and did the carrier see it?",
            "moments": [moment(r, "Straight at the defender") for r in straight[:4]]})
    tall = [r for r in carries if r["height"]]
    if len(tall) >= 3:
        low = sum(1 for r in tall if r["height"] == "low")
        up = [r for r in tall if r["height"] == "upright"]
        pts.append({
            "kind": "good" if low / len(tall) >= 0.6 else "issue" if len(up) / len(tall) >= 0.4 else "info",
            "title": f"Body height into contact: low in {low / len(tall):.0%}",
            "detail": f"Of {len(tall)} carries where the body could be seen, {low} went in low (shoulders well below standing "
                      f"height) and {len(up)} upright. Measured from shoulders, hips and ankles, so it works in close-ups too.",
            "why": "A low, strong body position with the head up wins the collision and keeps the carrier on their feet. "
                   "Upright carriers get driven back and are easy to hold up.",
            "moments": [moment(r, "Upright into contact") for r in up[:4]]})
    turned = [r for r in carries if r["turned"]]
    if len(turned) >= 3:
        sh = sum(1 for r in turned if r["turned"] == "shoulder")
        sq = [r for r in turned if r["turned"] == "square"]
        pts.append({
            "kind": "good" if sh / len(turned) >= 0.6 else "info",
            "title": f"Leading with the shoulder: {sh / len(turned):.0%} of carries",
            "detail": f"Of {len(turned)} carries where the turn of the body could be judged, {sh} led with the shoulder "
                      f"(side-on) and {len(sq)} went in square.",
            "why": "Leading with the shoulder protects the ball, which should be in the arm away from the contact. "
                   "Square carries expose the ball to the tackler.",
            "moments": [moment(r, "Square into contact") for r in sq[:4]]})
    dr = [r for r in carries if r["drive"] is not None]
    if len(dr) >= 3:
        good = sum(1 for r in dr if r["drive"] >= DRIVE)
        back = [r for r in dr if r["drive"] < 0]
        pts.append({
            "kind": "good" if good / len(dr) >= 0.5 else "issue" if len(back) / len(dr) >= 0.4 else "info",
            "title": f"Leg drive after contact: {good / len(dr):.0%} of carries",
            "detail": f"In {good} of {len(dr)} carries {T}'s carrier kept going forward after contact; {len(back)} were "
                      "pushed back.",
            "why": "Driving the legs through contact wins the gain line and gives the ruck quick ball.",
            "moments": [moment(r, "Pushed back after contact") for r in back[:4]]})
    if not pts and carries:
        pts.append({"kind": "info", "title": "Carries into contact: too few to judge",
                    "detail": f"Only {len(carries)} carries into contact could be followed.", "why": "", "moments": []})
    return pts
