"""
1v1 duels in soccer: a player on the ball with one defender in front of
them and nobody else close. The soccer counterpart of rugby's carries into
contact (contact.py), and built the same way: everything is measured
relative to the players (body lengths, body angles), so close-ups count
too, and the pose model reads the body.

For the defender (the coaching points of 1v1 defending, "jockeying"):
  - side-on (one foot forward, showing the attacker one way) or square
  - low (knees bent, weight forward) or upright
  - distance: about an arm's length (1-2 m), not diving in, not backing off
  - outcome: won the ball, or got beaten
For the player on the ball:
  - running at the defender's side (to go past) or straight at them
  - a change of pace in the take-on
  - outcome: kept the ball / went past, or lost it

One camera sees everything in the picture plane, so "side-on" can only be
judged when the two players face across the picture or along it.
"""

import math

import contact

BALL_REACH = 0.6     # body lengths from the feet: the ball is at this player's feet
DUEL_R = 2.0         # body lengths: the defender is this close...
ALONE_R = 3.0        # ...and no other defender within this
MIN_DUEL = 0.4       # s the 1v1 must last
DUEL_GAP = 0.3       # s it may break off and still be the same duel
APART = 1.5          # s between two duels of the same pair
DUEL_END = 2.5       # s after the start to see who has the ball
OWN_REACH = 0.8      # body lengths from the ball = has it (for the outcome)
STANCE_AT = 0.25     # s into the duel to read the defender's body
ARM_NEAR, ARM_FAR = 0.45, 1.3   # body lengths: about an arm's length (1-2 m) = good jockeying distance
LIMIT = 150          # duels read with the pose model per team


def _ball(frame):
    """Where the ball touches the ground in this frame (bottom middle), or None."""
    balls = [d for d in frame if d[4] == "ball"]
    if not balls:
        return None
    b = max(balls, key=lambda d: d[6])
    return ((b[0] + b[2]) / 2, b[3])


def _on_ball(frame, ball, team=None):
    """The player at the ball (optionally of `team`), or None."""
    best, who = None, None
    for d in contact._people(frame):
        if team and d[4] != team:
            continue
        fx, fy = contact._foot(d)
        r = math.hypot(fx - ball[0], fy - ball[1]) / contact._h(d)
        if r <= BALL_REACH and (best is None or r < best):
            best, who = r, d
    return who


def find_duels(detections, fps, team):
    """1v1s with a `team` player on the ball: [{f, carrier, defender}]."""
    live = {}     # (carrier, defender) -> [first frame, last frame]
    done = {}     # (carrier, defender) -> last frame of the last duel
    out = []

    def close(key):
        f0, f1 = live.pop(key)
        if f1 - f0 + 1 >= MIN_DUEL * fps and f0 - done.get(key, -10 ** 9) > APART * fps:
            out.append({"f": f0, "carrier": key[0], "defender": key[1]})
        done[key] = f1

    for f, frame in enumerate(detections):
        ball = _ball(frame)
        c = _on_ball(frame, ball, team) if ball else None
        pair = None
        if c:
            cx, cy = contact._foot(c)
            h = contact._h(c)
            opp = sorted((math.hypot(contact._foot(d)[0] - cx, contact._foot(d)[1] - cy) / h, d[5])
                         for d in contact._people(frame) if d[4] != team and d[4] != "other")
            if opp and opp[0][0] <= DUEL_R and (len(opp) == 1 or opp[1][0] > ALONE_R):
                pair = (c[5], opp[0][1])
        for key in list(live):
            if key != pair and f - live[key][1] > DUEL_GAP * fps:
                close(key)
        if pair:
            if pair in live:
                live[pair][1] = f
            else:
                live[pair] = [f, f]
    for key in list(live):
        close(key)
    return sorted(out, key=lambda d: d["f"])


def _outcome(detections, fps, d, team):
    """'won' (the defending team got the ball), 'kept' or None."""
    owners = []
    for f in range(d["f"], min(len(detections), d["f"] + int(DUEL_END * fps) + 1)):
        ball = _ball(detections[f])
        if not ball:
            continue
        best, who = None, None
        for p in contact._people(detections[f]):
            r = math.hypot(contact._foot(p)[0] - ball[0], contact._foot(p)[1] - ball[1]) / contact._h(p)
            if r <= OWN_REACH and (best is None or r < best):
                best, who = r, p[4]
        if who:
            owners.append(who)
    if not owners:
        return None
    theirs = sum(1 for o in owners if o != team)
    if theirs >= 3:
        return "won"
    return "kept" if owners[-1] == team else None


def analyse(detections, fps, team, pose=None):
    """Every 1v1 with a `team` player on the ball. pose(frame, box) ->
    keypoints (contact.pose_reader); without it body shape is left out."""
    out = []
    for d in find_duels(detections, fps, team)[:LIMIT]:
        f0 = d["f"]
        g = min(len(detections) - 1, f0 + max(1, int(STANCE_AT * fps)))
        run = contact._track(detections, d["carrier"], f0, f0 + int(0.5 * fps))
        dfn = contact._track(detections, d["defender"], f0 - int(fps), g)
        if g not in dfn or f0 not in run or f0 not in dfn:
            continue
        c0, de = run[f0], dfn[g]
        h = max(contact._h(x) for x in dfn.values())
        (cx, cy), (dx, dy) = contact._foot(c0), contact._foot(dfn[f0])
        to_def = (dx - cx, dy - cy)
        rec = {"f": f0, "t": round(f0 / fps, 2), "carrier": d["carrier"], "defender": d["defender"],
               "box": [int(v) for v in de[:4]], "carrier_box": [int(v) for v in c0[:4]],
               "outcome": _outcome(detections, fps, d, team), "heading": None, "burst": None,
               "distance": None, "height": None, "turned": None}
        fs = sorted(run)
        if len(fs) >= 3:
            p = {f: contact._foot(run[f]) for f in fs}
            vx, vy = p[fs[-1]][0] - p[fs[0]][0], p[fs[-1]][1] - p[fs[0]][1]
            if math.hypot(vx, vy) > 0.1 * h:
                a = contact._angle(vx, vy, *to_def)
                if a is not None:
                    rec["heading"] = "straight" if a < contact.STRAIGHT else "side" if a >= contact.EDGE else None
            mid = fs[len(fs) // 2]
            early = math.hypot(p[mid][0] - p[fs[0]][0], p[mid][1] - p[fs[0]][1]) / max(1, mid - fs[0])
            late = math.hypot(p[fs[-1]][0] - p[mid][0], p[fs[-1]][1] - p[mid][1]) / max(1, fs[-1] - mid)
            rec["burst"] = early > 0 and late / early >= 1.3
        gx, gy = contact._foot(de)
        cg = contact._track(detections, d["carrier"], g, g).get(g, c0)
        rec["distance"] = round(math.hypot(gx - contact._foot(cg)[0], gy - contact._foot(cg)[1]) / h, 2)
        if pose:
            kp = pose(g, de[:4])
            # The defender faces the attacker: "shoulder" here means side-on.
            rec.update(contact.body(kp, de[:4], h, (-to_def[0], -to_def[1])))
        out.append(rec)
    return out


def _moment(r, label, box="box"):
    from analysis import _fmt_t
    return {"frame": r["f"], "t": r["t"], "time": _fmt_t(r["t"]), "label": label,
            "hl": "box:" + ",".join(str(v) for v in r["box"]) + ";box:" + ",".join(str(v) for v in r["carrier_box"])}


def defending_points(duels, team):
    """Points for `team` defending 1v1s (duels = the other team on the ball)."""
    T = team.capitalize()
    pts = []
    judged = [r for r in duels if r["outcome"]]
    if len(judged) >= 3:
        won = sum(1 for r in judged if r["outcome"] == "won")
        lost = [r for r in judged if r["outcome"] == "kept"]
        pts.append({
            "kind": "good" if won / len(judged) >= 0.5 else "issue" if won / len(judged) < 0.3 else "info",
            "title": f"1v1 defending: won the ball in {won / len(judged):.0%}",
            "detail": f"Of {len(judged)} 1v1s against {T} where the ball could be followed, {T} won it {won} times and the "
                      f"attacker kept it {len(lost)} times.",
            "why": "In a 1v1 the defender's job is first to stop the attacker going forward, then to win it at the right moment. "
                   "Look at the lost ones: did the defender dive in, or stand off?",
            "moments": [_moment(r, "Attacker kept it") for r in lost[:4]]})
    side = [r for r in duels if r["turned"]]
    if len(side) >= 3:
        on = sum(1 for r in side if r["turned"] == "shoulder")
        sq = [r for r in side if r["turned"] == "square"]
        pts.append({
            "kind": "good" if on / len(side) >= 0.6 else "issue" if on / len(side) < 0.4 else "info",
            "title": f"Side-on in 1v1s: {on / len(side):.0%}",
            "detail": f"Of {len(side)} 1v1s where the defender's body could be judged, {on} were side-on and {len(sq)} square "
                      "to the attacker (from shoulders and hips, so close-ups count too).",
            "why": "Side-on, one foot in front, the defender can turn and run with the attacker and shows them one way "
                   "(usually away from goal or onto the weaker foot). Square-on, the attacker can go either side.",
            "moments": [_moment(r, "Square to the attacker") for r in sq[:4]]})
    tall = [r for r in duels if r["height"]]
    if len(tall) >= 3:
        low = sum(1 for r in tall if r["height"] in ("low", "mid"))
        up = [r for r in tall if r["height"] == "upright"]
        pts.append({
            "kind": "good" if low / len(tall) >= 0.6 else "issue" if len(up) / len(tall) >= 0.5 else "info",
            "title": f"Knees bent in 1v1s: {low / len(tall):.0%}",
            "detail": f"Of {len(tall)} 1v1s, the defender was low (knees bent, shoulders below standing height) in {low} "
                      f"and upright in {len(up)}.",
            "why": "A low centre of gravity, weight on the front of the feet, lets the defender react to the touch. "
                   "Upright defenders get turned.",
            "moments": [_moment(r, "Upright") for r in up[:4]]})
    dist = [r for r in duels if r["distance"] is not None]
    if len(dist) >= 3:
        ok = sum(1 for r in dist if ARM_NEAR <= r["distance"] <= ARM_FAR)
        far = [r for r in dist if r["distance"] > ARM_FAR]
        near = [r for r in dist if r["distance"] < ARM_NEAR]
        pts.append({
            "kind": "good" if ok / len(dist) >= 0.6 else "info",
            "title": f"Distance in 1v1s: about an arm's length in {ok / len(dist):.0%}",
            "detail": f"Of {len(dist)} 1v1s, the defender stood about an arm's length away (1-2 m) in {ok}, further off in "
                      f"{len(far)} and tight in {len(near)}.",
            "why": "Too far gives the attacker time and space to run; too tight and one touch past you beats you. "
                   "About an arm's length is close enough to put a foot in.",
            "moments": [_moment(r, "Standing off") for r in far[:2]] + [_moment(r, "Too tight") for r in near[:2]]})
    return pts


def attacking_points(duels, team):
    """Points for `team` taking players on (duels = `team` on the ball)."""
    T = team.capitalize()
    pts = []
    judged = [r for r in duels if r["outcome"]]
    if len(judged) >= 3:
        kept = sum(1 for r in judged if r["outcome"] == "kept")
        lost = [r for r in judged if r["outcome"] == "won"]
        pts.append({
            "kind": "good" if kept / len(judged) >= 0.6 else "issue" if kept / len(judged) < 0.4 else "info",
            "title": f"Taking players on: kept the ball in {kept / len(judged):.0%}",
            "detail": f"In {len(judged)} 1v1s, {T}'s player kept the ball {kept} times and lost it {len(lost)} times.",
            "why": "Check the lost ones: was there a pass on, or was the take-on in a dangerous place?",
            "moments": [_moment(r, "Lost it") for r in lost[:4]]})
    head = [r for r in duels if r["heading"]]
    if len(head) >= 3:
        side = sum(1 for r in head if r["heading"] == "side")
        straight = [r for r in head if r["heading"] == "straight"]
        burst = sum(1 for r in duels if r["burst"])
        pts.append({
            "kind": "good" if side / len(head) >= 0.5 else "info",
            "title": f"Running at the defender's side: {side / len(head):.0%}",
            "detail": f"In {len(head)} take-ons, {side} went at the space beside the defender and {len(straight)} straight at "
                      f"them. {burst} had a change of pace.",
            "why": "Running at the defender's front foot side and changing pace commits them, and that's when the gap opens.",
            "moments": [_moment(r, "Straight at the defender") for r in straight[:4]]})
    return pts
