"""
Rule-of-thumb coaching analysis built from the saved detections.

Everything is measured in the video picture, not in metres on the pitch -
we don't know where the pitch lines are. Distances are expressed in
"body lengths" (a player's box height at that spot on screen), which
roughly cancels out camera zoom and distance from the camera. Only players
the camera can see are counted.

So treat the output as pointers to moments worth reviewing on video, not as
exact measurements. The report says the same thing to the user.
"""

import math
from statistics import median

MIN_BOX_HEIGHT = 12   # px; smaller boxes are usually crowd or noise
CONTACT_TOUCH = 0.6   # feet closer than this (body lengths) = in contact
CONTACT_MIN = 4       # this many players clumped together = ruck/maul/scrum
MIN_LINE = 4          # players needed outside contact to judge a line
DETACHED = 3.5        # body lengths from every teammate = not part of the line
                      # (touch judges, sideline staff, a lone winger or full-back).
                      # A "hole" is then a gap between two connected groups.
CLOSE_UP = 0.22       # typical player taller than this share of the picture
                      # = close-up / replay; team shape can't be judged there
MIN_WIDE_PLAYERS = 6  # fewer players in view = too tight a shot to judge shape
MAX_UNSURE = 0.15     # more than this share of "unsure" players in a frame =
                      # a gap in the line might just be a player we couldn't
                      # place, so line shape isn't judged in that frame


# ---------------------------------------------------------------- per frame

def _players(detections, size):
    players = []
    for x1, y1, x2, y2, bucket, track_id, conf in detections:
        if bucket == "ball" or (y2 - y1) < MIN_BOX_HEIGHT:
            continue
        # Players cut off by the edge of the picture, or bent over / on the
        # ground, have a misleading box height - count them as in view, but
        # don't use them to measure the line.
        cut = bool(size) and (x1 <= 3 or y1 <= 3 or x2 >= size[0] - 3 or y2 >= size[1] - 3)
        low = (x2 - x1) > 0.9 * (y2 - y1)
        players.append({
            "x": (x1 + x2) / 2, "y": float(y2), "h": float(y2 - y1),
            "bucket": bucket, "box": (x1, y1, x2, y2), "measurable": not (cut or low), "id": track_id,
        })
    return players


def _contact_groups(players):
    """Clumps of CONTACT_MIN+ players standing on top of each other."""
    n = len(players)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = players[i]
        for j in range(i + 1, n):
            b = players[j]
            scale = (a["h"] + b["h"]) / 2
            if abs(a["x"] - b["x"]) > scale * CONTACT_TOUCH:
                continue
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) / scale < CONTACT_TOUCH:
                parent[find(i)] = find(j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # A ruck/maul/scrum has both teams in it; one team bunched together
    # (e.g. waiting for a lineout, or overlapping in a close-up) isn't one.
    return [g for g in groups.values()
            if len(g) >= CONTACT_MIN and len({players[i]["bucket"] for i in g} - {"other", "unsure"}) >= 2]


def prepare(all_detections, size=None):
    """Team-independent per-frame data; cache this per video.
    size = (width, height) of the video picture."""
    frames = []
    for detections in all_detections:
        players = _players(detections, size)
        wide = len(players) >= MIN_WIDE_PLAYERS and (
            not size or median(p["h"] for p in players) < CLOSE_UP * size[1])
        if not wide:
            players = []  # close-up / replay: skip for structure analysis
        groups = _contact_groups(players)
        unsure = sum(1 for p in players if p["bucket"] == "unsure")
        clear = bool(players) and unsure <= MAX_UNSURE * len(players)
        frames.append((players, groups, {i for g in groups for i in g}, clear))
    return frames


def _line_shape(pts):
    """Shape of a set of players standing outside contact."""
    n = len(pts)
    mx = sum(p["x"] for p in pts) / n
    my = sum(p["y"] for p in pts) / n
    sxx = sum((p["x"] - mx) ** 2 for p in pts)
    syy = sum((p["y"] - my) ** 2 for p in pts)
    sxy = sum((p["x"] - mx) * (p["y"] - my) for p in pts)
    # Main direction the players are spread along = the line.
    angle = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux

    along = [(p["x"] - mx) * ux + (p["y"] - my) * uy for p in pts]
    across = [((p["x"] - mx) * vx + (p["y"] - my) * vy) / p["h"] for p in pts]

    # Neighbours along the line; the gap is the real distance between them.
    order = sorted(range(n), key=lambda k: along[k])
    gaps = []
    for a, b in zip(order, order[1:]):
        pa, pb = pts[a], pts[b]
        scale = (pa["h"] + pb["h"]) / 2
        gaps.append((math.hypot(pa["x"] - pb["x"], pa["y"] - pb["y"]) / scale, pa, pb))
    widest = max(gaps, key=lambda g: g[0])

    mid = median(across)
    devs = [abs(c - mid) for c in across]
    out_of_line = max(range(n), key=lambda k: devs[k])
    mean_across = sum(across) / n

    return {
        "gaps": [g[0] for g in gaps],
        "max_gap": widest[0],
        "gap_pair": (widest[1], widest[2]),
        "dogleg": devs[out_of_line],
        "dogleg_player": pts[out_of_line],
        "depth": math.sqrt(sum((c - mean_across) ** 2 for c in across) / n),
        "width": (max(along) - min(along)) / median(p["h"] for p in pts),
        "line": pts,
    }


def _team_frame(frame, team):
    players, groups, in_contact, clear = frame
    mine = [i for i, p in enumerate(players) if p["bucket"] == team]
    line = [i for i in mine if i not in in_contact]
    measurable = [i for i in line if players[i]["measurable"]]

    def near_teammate(i):
        a = players[i]
        return any(math.hypot(a["x"] - players[j]["x"], a["y"] - players[j]["y"]) /
                   ((a["h"] + players[j]["h"]) / 2) <= DETACHED
                   for j in measurable if j != i)
    measurable = [i for i in measurable if near_teammate(i)]
    commits = []
    for g in groups:
        c = sum(1 for i in g if players[i]["bucket"] == team)
        if c:
            commits.append((c, g))
    unsure_line = sum(1 for i, p in enumerate(players) if p["bucket"] == "unsure" and i not in in_contact)
    r = {"n": len(mine), "n_line": len(line), "unsure_line": unsure_line, "commits": commits, "players": players}
    if clear and len(measurable) >= MIN_LINE:
        r.update(_line_shape([players[i] for i in measurable]))
    return r


# ------------------------------------------------------------- highlights

def _center(p):
    x1, y1, x2, y2 = p["box"]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _hl_gap(a, b):
    (ax, ay), (bx, by) = _center(a), _center(b)
    return f"line:{ax},{ay},{bx},{by};box:{','.join(map(str, a['box']))};box:{','.join(map(str, b['box']))}"


def _hl_box(box):
    return "box:" + ",".join(str(int(v)) for v in box)


def _hl_bounds(players):
    return _hl_box((
        min(p["box"][0] for p in players), min(p["box"][1] for p in players),
        max(p["box"][2] for p in players), max(p["box"][3] for p in players),
    ))


# ----------------------------------------------------------------- events

def _events(flags, fps):
    """Turn per-frame flags {frame: (severity, highlight)} into distinct
    moments, ignoring flickers shorter than ~0.3 s."""
    tol = max(2, int(fps * 0.5))
    min_len = max(3, int(fps * 0.3))
    events, run = [], []

    def close():
        if len(run) >= min_len:
            best = max(run, key=lambda f: flags[f][0])
            events.append({
                "frame": best, "t": round(best / fps, 2),
                "start": round(run[0] / fps, 1), "end": round(run[-1] / fps, 1),
                "severity": flags[best][0], "hl": flags[best][1],
            })

    for f in sorted(flags):
        if run and f - run[-1] > tol:
            close()
            run = []
        run.append(f)
    close()
    events.sort(key=lambda e: -e["severity"])
    return events


def _moments(events, label, limit=4):
    return [dict(e, label=label(e)) for e in events[:limit]]


def _fmt_t(t):
    t = int(t)
    return f"{t // 60}:{t % 60:02d}"


# ------------------------------------------------------------------ report

def analyse_team(frames, fps, team, opp, phase):
    """phase: 'attack', 'defence' or 'mixed' - for `team`."""
    T = team.capitalize()
    O = opp.capitalize() if opp else None
    per = [_team_frame(f, team) for f in frames]
    per_opp = [_team_frame(f, opp) for f in frames] if opp else None
    total = len(frames)
    seconds = total / fps if fps else 0
    minutes = max(seconds / 60, 0.25)  # don't let short clips inflate rates

    seen = [r for r in per if r["n"]]
    line_frames = [i for i, r in enumerate(per) if "gaps" in r]
    all_gaps = [g for i in line_frames for g in per[i]["gaps"]]
    typical_gap = median(all_gaps) if all_gaps else None
    commits = [c for r in per for c, _ in r["commits"]]
    avg_commit = sum(commits) / len(commits) if commits else None

    stats = {
        "players_in_view": round(sum(r["n"] for r in seen) / len(seen), 1) if seen else 0,
        "players_in_line": round(sum(r["n_line"] for r in seen) / len(seen), 1) if seen else 0,
        "per_breakdown": round(avg_commit, 1) if avg_commit else None,
        "usual_spacing": round(typical_gap, 1) if typical_gap else None,
    }
    points = []
    notes = []

    wide = sum(1 for f in frames if f[0])
    all_seen = sum(len(f[0]) for f in frames)
    unsure_seen = sum(1 for f in frames for p in f[0] if p["bucket"] == "unsure")
    if all_seen and unsure_seen > 0.05 * all_seen:
        notes.append(f"{unsure_seen / all_seen:.0%} of player sightings couldn't be confidently put in a team "
                     "(\"unsure\" - usually far from the camera). Moments with many unsure players are left "
                     "out of the line measurements, so some real moments may be missed.")
    if wide < total * 0.9:
        notes.append(f"Used {wide / fps:.0f}s of wide-angle footage out of {seconds:.0f}s - "
                     "close-ups and replays are skipped because team shape can't be judged from them.")
    if len(seen) < wide * 0.05 or not seen:
        notes.append(f"{T} was hardly visible in the wide shots, so there's not enough to analyse.")
        return {"team": team, "opponent": opp, "phase": phase, "stats": stats, "points": points, "notes": notes}

    enough_line = len(line_frames) >= fps * 2
    if not enough_line:
        notes.append(f"{T} rarely had {MIN_LINE}+ players standing in a line on screen, "
                     "so line shape (gaps, depth, width) couldn't be judged.")

    # --- 1. Gaps between neighbouring players in the line ---------------
    if enough_line and typical_gap:
        limit = max(typical_gap * (2.2 if phase != "attack" else 3.0), typical_gap + 1.5)
        flags = {}
        for i in line_frames:
            r = per[i]
            if r["max_gap"] > limit:
                flags[i] = (r["max_gap"] / typical_gap, _hl_gap(*r["gap_pair"]))
        ev = _events(flags, fps)
        rate = len(ev) / minutes
        label = lambda e: f"Gap about {e['severity']:.1f}× the usual spacing"
        if phase == "defence":
            if ev and rate >= 0.5:
                points.append({
                    "kind": "issue", "title": "Holes in the defensive line",
                    "detail": f"{len(ev)} moment(s) where the gap between two {T} defenders was more than "
                              f"{limit / typical_gap:.1f}× their usual spacing (usual ≈ {typical_gap:.1f} body lengths).",
                    "why": "Gaps like these are where line breaks come from. Check whether a player was slow to "
                           "fold round from the ruck, drifted too early, or stopped talking to the player next to them.",
                    "moments": _moments(ev, label),
                })
            else:
                points.append({
                    "kind": "good", "title": "Defensive line stayed connected",
                    "detail": f"No lasting big gaps between {T} defenders in view.",
                    "why": "Keep it up - a connected line forces the attack to go through contact.",
                    "moments": [],
                })
        elif phase == "attack":
            if ev and rate >= 0.5:
                points.append({
                    "kind": "issue", "title": "Attackers too far apart to support each other",
                    "detail": f"{len(ev)} moment(s) where two neighbouring {T} attackers were more than "
                              f"{limit / typical_gap:.1f}× the usual spacing apart.",
                    "why": "Width is good, but a player too far away can't take a pass or support a line break. "
                           "Check whether support runners were late or the shape had split.",
                    "moments": _moments(ev, label),
                })
        else:
            if ev:
                points.append({
                    "kind": "info", "title": "Big gaps between neighbouring players",
                    "detail": f"{len(ev)} moment(s) with a gap over {limit / typical_gap:.1f}× {T}'s usual spacing.",
                    "why": "In defence these are holes to close; in attack check the support could still reach. "
                           "Pick Attacking or Defending above for more specific points.",
                    "moments": _moments(ev, label),
                })

    # --- 2. Players out of line (defence) / flat attack --------------------
    if enough_line and phase in ("defence", "mixed"):
        devs = [per[i]["dogleg"] for i in line_frames]
        limit = max(1.5, median(devs) * 2.5)
        flags = {i: (per[i]["dogleg"], _hl_box(per[i]["dogleg_player"]["box"]))
                 for i in line_frames if per[i]["dogleg"] > limit}
        ev = _events(flags, fps)
        if ev and len(ev) / minutes >= 0.5:
            points.append({
                "kind": "issue" if phase == "defence" else "info",
                "title": "Players out of line (dog-legs)",
                "detail": f"{len(ev)} moment(s) where one {T} player stood about {limit:.1f}+ body lengths "
                          f"in front of or behind the rest of the line.",
                "why": "A defender behind the line gives the attack space on the inside; one who shoots up "
                       "alone can be stepped or passed around. The line should move up together.",
                "moments": _moments(ev, lambda e: f"Player ~{e['severity']:.1f} body lengths out of line"),
            })
        elif phase == "defence":
            points.append({
                "kind": "good", "title": "Line kept its shape",
                "detail": f"{T} defenders mostly stayed level with each other.",
                "why": "A flat, level line is what lets the defence move up together.",
                "moments": [],
            })

    if enough_line and phase == "attack":
        flat = {}
        compared = 0
        for i in line_frames:
            r, o = per[i], per_opp[i] if per_opp else None
            if o and "depth" in o:
                compared += 1
                if r["depth"] <= o["depth"] * 1.1:
                    flat[i] = (o["depth"] - r["depth"] + 1, _hl_bounds(r["line"]))
            elif r["depth"] < 0.5:
                compared += 1
                flat[i] = (1 - r["depth"], _hl_bounds(r["line"]))
        share = len(flat) / compared if compared else 0
        ev = _events(flat, fps)
        vs = f"as flat as {O}'s defence" if per_opp else "very flat"
        if share >= 0.5:
            points.append({
                "kind": "issue", "title": "Attack too flat",
                "detail": f"In about {share:.0%} of the time {T}'s attacking line was {vs}.",
                "why": "Without depth, receivers catch the ball standing still and right on the defence. "
                       "Starting a few metres deeper lets them run onto the ball and hit the line at pace.",
                "moments": _moments(ev, lambda e: "Flat attacking line"),
            })
        elif compared:
            points.append({
                "kind": "good", "title": "Attack had depth",
                "detail": f"{T}'s attacking line was flat only about {share:.0%} of the time.",
                "why": "Depth lets receivers run onto the ball and choose their line.",
                "moments": [],
            })

    # --- 3. Numbers in the line (needs the other team) ----------------------
    if per_opp:
        flags = {}
        for i, (r, o) in enumerate(zip(per, per_opp)):
            diff = (o["n_line"] - r["n_line"]) if phase != "attack" else (r["n_line"] - o["n_line"])
            # Unsure players could belong to either side, so they have to be
            # outnumbered too before we call it.
            diff -= r["unsure_line"]
            if diff >= 2 and min(r["n_line"], o["n_line"]) >= 2:
                group = [p for p in r["players"] if p["bucket"] in (team, opp)]
                flags[i] = (diff, _hl_bounds(group) if group else "")
        ev = _events(flags, fps)
        if phase == "defence":
            if ev:
                points.append({
                    "kind": "issue", "title": f"Outnumbered by {O}'s attack",
                    "detail": f"{len(ev)} moment(s) where {O} had 2+ more players standing in the line than {T} had defenders in view.",
                    "why": "That's where overlaps come from. Look at why: players still on the floor at the ruck, "
                           "slow to get back onside, or too many committed to the breakdown.",
                    "moments": _moments(ev, lambda e: f"{O} +{e['severity']} in the line"),
                })
            else:
                points.append({
                    "kind": "good", "title": "Numbers matched in defence",
                    "detail": f"{T} had enough defenders in the line to match {O}'s attackers in view.",
                    "why": "Getting back onside and into the line quickly is what stops overlaps.",
                    "moments": [],
                })
        elif phase == "attack" and ev:
            points.append({
                "kind": "info", "title": "Overlap chances",
                "detail": f"{len(ev)} moment(s) where {T} had 2+ more players in the line than {O} had defenders in view.",
                "why": "Check the video: did the ball get to the extra player, or did the attack go back into contact?",
                "moments": _moments(ev, lambda e: f"{T} +{e['severity']} in the line"),
            })

    # --- 4. Players committed to rucks / mauls ------------------------------
    if avg_commit:
        big = {}
        for i, r in enumerate(per):
            if r["commits"]:
                c, g = max(r["commits"], key=lambda cg: cg[0])
                if c >= 4:
                    big[i] = (c, _hl_bounds([r["players"][k] for k in g]))
        ev = _events(big, fps)
        label = lambda e: f"{e['severity']} {T} players in one ruck/maul"
        detail = f"On average {avg_commit:.1f} {T} player(s) were in each ruck/maul/scrum on screen."
        if phase == "defence" and avg_commit >= 3:
            points.append({
                "kind": "issue", "title": "Committing too many to the breakdown",
                "detail": detail,
                "why": "Every extra defender in the ruck is one fewer standing in the line. Unless you're "
                       "competing for a turnover, 1-2 at the ruck is usually enough.",
                "moments": _moments(ev, label),
            })
        elif phase == "attack" and avg_commit >= 4:
            points.append({
                "kind": "issue", "title": "Too many attackers in the ruck",
                "detail": detail,
                "why": "Over-committing to protect the ball leaves fewer runners for the next phase. "
                       "Aim for 2-3 quick cleaners and get everyone else back in shape.",
                "moments": _moments(ev, label),
            })
        else:
            points.append({
                "kind": "good" if phase != "mixed" else "info",
                "title": "Players at the breakdown",
                "detail": detail,
                "why": "Numbers at the ruck look reasonable. (Players buried in a ruck can be missed, so the real number may be a bit higher.)",
                "moments": [],
            })

    # --- 5. Width compared with the other team -----------------------------
    if enough_line and per_opp:
        ratios, flags = [], {}
        for i in line_frames:
            o = per_opp[i]
            if "width" in o and o["width"] > 0:
                ratio = per[i]["width"] / o["width"]
                ratios.append(ratio)
                if ratio < 0.7:
                    flags[i] = (1 / max(ratio, 0.05), _hl_bounds(per[i]["line"]))
        if len(ratios) >= fps * 2:
            ratio = median(ratios)
            ev = _events(flags, fps)
            label = lambda e: f"{T} line ~{100 / e['severity']:.0f}% as wide as {O}'s"
            if phase == "attack":
                if ratio < 0.8:
                    points.append({
                        "kind": "issue", "title": "Not using the width",
                        "detail": f"{T}'s attacking line typically covered about {ratio:.0%} of the width {O}'s defence did.",
                        "why": "Attacking narrower than the defence lets defenders drift across without being stretched. "
                               "Get width early so the defence has to spread and gaps open.",
                        "moments": _moments(ev, label),
                    })
                elif ratio >= 1.05:
                    points.append({
                        "kind": "good", "title": "Good width in attack",
                        "detail": f"{T} spread wider than {O}'s defence (about {ratio:.0%} of its width).",
                        "why": "Width stretches the defence and creates space in the middle.",
                        "moments": [],
                    })
            elif phase == "defence":
                if ratio < 0.8:
                    points.append({
                        "kind": "issue", "title": "Defence too narrow - edges exposed",
                        "detail": f"{T}'s defensive line typically covered about {ratio:.0%} of the width {O}'s attack did.",
                        "why": "When the defence is narrower than the attack, the outside attackers have space. "
                               "Check the edge defenders' starting positions and whether they get pulled in.",
                        "moments": _moments(ev, label),
                    })
                elif ratio >= 0.95:
                    points.append({
                        "kind": "good", "title": "Defence covered the width",
                        "detail": f"{T}'s line was about as wide as {O}'s attack ({ratio:.0%}).",
                        "why": "Covering the attack's width takes away easy space on the edges.",
                        "moments": [],
                    })

    order = {"issue": 0, "info": 1, "good": 2}
    points.sort(key=lambda p: order[p["kind"]])
    for p in points:
        for m in p["moments"]:
            m["time"] = _fmt_t(m["t"])
    return {"team": team, "opponent": opp, "phase": phase, "stats": stats, "points": points, "notes": notes}
