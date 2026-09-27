"""
Rugby measurements on top of the tracking: breakdowns (rucks, mauls,
scrums), ruck speed and defensive line speed.

Works from analysis.prepare() frames: a "contact group" there is 4+ players
of both teams standing on top of each other. Here those groups are followed
over time, so each ruck becomes one event with a start and an end.

Ruck speed is timed the way coaches time it: from the tackle (the ball
carrier going to ground with a tackler) until the ball is out. The tackle is
found by looking back from the moment the ruck forms for the opposing pair
at that spot; the ball out is the first frame the ball is seen clear of the
ruck, or, when the ball isn't seen, the ruck breaking up. Quick ball is
under 3 s; over 4 s is slow ball that lets the defence reorganise.

With the pitch marked (pitch.py), positions are in metres, which adds:
- mauls: a group that moves 3 m or more;
- defensive line speed: how fast the defending line moves up in the
  first 1.5 s after the ball comes out. The defending team is the one whose
  line stands closest to the ruck (the attack stands deeper).
- possible offside at the ruck (Law 15): everyone not in the ruck must stay
  behind the offside line through the hindmost foot of the ruck on their
  side. Defenders standing in front of it just before the ball comes out
  are shown as moments to check.

Thresholds are rules of thumb; each is a named constant below.
"""

import math
from statistics import median

import numpy as np

import pitch
from analysis import CONTACT_TOUCH

JOIN_GAP = 0.6         # s a group may disappear (hidden, missed) and still be the same ruck
MIN_BREAKDOWN = 1.0    # s a group must last to count as a ruck/maul/scrum
MAX_BREAKDOWN = 60.0   # s longer than this = not a breakdown we can time
SCRUM_SIZE = 10        # players seen in one group = a scrum
MAUL_MOVE = 3.0        # m a group travels = a maul
QUICK_RUCK = 3.0       # s: quick ball, the usual coaching benchmark
SLOW_RUCK = 4.0        # s: slow ball, the defence has time to reorganise
TACKLE_LOOKBACK = 2.0  # s before the ruck forms to look for the tackle that started it
BALL_CLEAR = 0.4       # body lengths outside the ruck = the ball is out
BALL_REACH = 4.0       # ... but within this many body lengths (not a ball elsewhere)
OFFSIDE_WINDOW = 1.0   # s before the ball comes out that players are checked for offside
OFFSIDE_MARGIN = 1.0   # m in front of the offside line before it counts (one camera isn't exact)
OFFSIDE_SIDE = 1.5     # m across the pitch from the ruck's middle: closer = could be in the ruck
OFFSIDE_SHARE = 0.5    # share of the window a player must be offside for
LINE_WINDOW = 1.5      # s after the ball comes out that line speed is measured over
LINE_REACH = 30.0      # m either side of the ruck (across the pitch) that counts as "the line"
MIN_LINE_PLAYERS = 3
PASSIVE_LINE = 1.5     # m/s: slower than this = a passive defensive line
FAST_LINE = 2.5        # m/s: faster than this = good line speed
MAX_SPEED = 9.0        # m/s: faster than this = a measuring error, ignored


def _feet(players, idx):
    return [(players[i]["x"], players[i]["y"]) for i in idx]


def _group_facts(frame, g, size):
    players = frame[0]
    xs = [players[i]["x"] for i in g]
    ys = [players[i]["y"] for i in g]
    box = (min(players[i]["box"][0] for i in g), min(players[i]["box"][1] for i in g),
           max(players[i]["box"][2] for i in g), max(players[i]["box"][3] for i in g))
    edge = bool(size) and (box[0] <= 4 or box[2] >= size[0] - 4)
    return {
        "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys), "h": median(players[i]["h"] for i in g),
        "ids": {players[i]["id"] for i in g if players[i].get("id") is not None},
        "n": len(g), "box": box, "edge": edge, "idx": g,
    }


def breakdowns(frames, fps, maps=None, size=None):
    """Every ruck / maul / scrum seen, as events with start/end times.
    maps: per-frame pixel -> metres mappings (or None)."""
    gap = max(2, int(JOIN_GAP * fps))
    active, done = [], []
    for f, frame in enumerate(frames):
        groups = [_group_facts(frame, g, size) for g in frame[1]]
        wide = bool(frame[0])
        used = set()
        for ev in active:
            last = ev["seen"][-1][1]
            best, best_d = None, None
            for j, g in enumerate(groups):
                if j in used:
                    continue
                shared = len(last["ids"] & g["ids"])
                d = math.hypot(last["cx"] - g["cx"], last["cy"] - g["cy"]) / max(last["h"], g["h"], 1)
                if shared >= 2 or d < 1.5:
                    if best is None or d < best_d:
                        best, best_d = j, d
            if best is not None:
                used.add(best)
                ev["seen"].append((f, groups[best]))
        for j, g in enumerate(groups):
            if j not in used:
                # Did the view start just before this (so the ruck may have
                # formed off camera)?
                started_blind = f == 0 or not all(frames[k][0] for k in range(max(0, f - gap), f))
                active.append({"seen": [(f, g)], "blind_start": started_blind or g["edge"]})
        still = []
        for ev in active:
            if f - ev["seen"][-1][0] > gap:
                done.append(ev)
            else:
                still.append(ev)
        active = still
        if not wide:
            # Close-up / replay: we can't see how the rucks on screen end.
            for ev in active:
                ev["blind_end"] = True
    done.extend(active)
    for ev in active:
        ev["blind_end"] = True

    out = []
    for ev in done:
        f0, f1 = ev["seen"][0][0], ev["seen"][-1][0]
        secs = (f1 - f0 + 1) / fps
        if secs < MIN_BREAKDOWN or secs > MAX_BREAKDOWN:
            continue
        biggest = max(n for _, n in ((f, g["n"]) for f, g in ev["seen"]))
        last = ev["seen"][-1][1]
        # A ruck that ends against the edge of the picture or right before a
        # cut may have ended off camera: count it, but don't time it.
        after = range(f1 + 1, min(len(frames), f1 + gap + 1))
        blind_end = ev.get("blind_end") or last["edge"] or not all(frames[k][0] for k in after)
        kind = "scrum" if biggest >= SCRUM_SIZE else "ruck"
        where = travel = None
        if maps:
            pts = []
            for f, g in ev["seen"]:
                H = maps[f] if f < len(maps) else None
                if H is not None:
                    xy = pitch.project(H, _feet(frames[f][0], g["idx"]))
                    pts.append((f, float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))))
            if len(pts) >= 2:
                where = pts[-1][1:]
                travel = math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])
                if kind == "ruck" and travel >= MAUL_MOVE and secs >= 2:
                    kind = "maul"
        mid_f, mid_g = ev["seen"][len(ev["seen"]) // 2]
        group_f0 = f0
        if kind == "ruck":
            f0 = _tackle_start(frames, f0, ev["seen"][0][1], fps)
            out_f = _ball_out(frames, ev["seen"], fps)
            if out_f is not None:
                f1 = out_f
            secs = (f1 - f0 + 1) / fps
        else:
            out_f = None
        out.append({
            "kind": kind, "start_f": f0, "group_f": group_f0, "mid_f": mid_f, "mid_box": mid_g["box"], "end_f": f1,
            "start": round(f0 / fps, 2), "end": round(f1 / fps, 2), "ball_seen": out_f is not None,
            "seconds": round(secs, 1), "players": biggest, "timed": not (ev["blind_start"] or blind_end),
            "box": last["box"], "where": where, "travel": round(travel, 1) if travel is not None else None,
            "idx": last["idx"],
        })
    out.sort(key=lambda e: e["start_f"])
    return out


def _tackle_start(frames, f0, first, fps):
    """The frame the tackle happened: going back from the ruck forming, the
    last frame in a row with players of both teams touching at that spot."""
    start = f0
    for f in range(f0 - 1, max(-1, f0 - int(TACKLE_LOOKBACK * fps) - 1), -1):
        near = [p for p in frames[f][0] if p["bucket"] not in ("other", "unsure")
                and math.hypot(p["x"] - first["cx"], p["y"] - first["cy"]) < 1.5 * first["h"]]
        if not any(a["bucket"] != b["bucket"]
                   and math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < CONTACT_TOUCH * (a["h"] + b["h"]) / 2
                   for i, a in enumerate(near) for b in near[i + 1:]):
            break
        start = f
    return start


def _ball_out(frames, seen, fps):
    """The first frame, from halfway through the ruck on, where the ball is
    seen clear of it (passed or picked up from the base), in two frames close
    together so a single wrong detection doesn't end the ruck. None if the
    ball isn't seen."""
    first_seen = None
    for f, g in seen[len(seen) // 2:]:
        x1, y1, x2, y2 = g["box"]
        pad, reach = BALL_CLEAR * g["h"], BALL_REACH * g["h"]
        clear = [b for b in (frames[f][4] if len(frames[f]) > 4 else [])
                 if not (x1 - pad <= b[0] <= x2 + pad and y1 - pad <= b[1] <= y2 + pad)
                 and math.hypot(b[0] - g["cx"], b[1] - g["cy"]) < reach]
        if not clear:
            continue
        if first_seen is not None and f - first_seen <= max(2, int(0.3 * fps)):
            return first_seen
        first_seen = f
    return None


def _team_line(frame, H, team, ruck_x, ruck_y):
    """Pitch x of each `team` player near the ruck, standing outside contact."""
    players, _, in_contact = frame[:3]
    idx = [i for i, p in enumerate(players) if p["bucket"] == team and i not in in_contact]
    if len(idx) < MIN_LINE_PLAYERS:
        return None, []
    xy = pitch.project(H, _feet(players, idx))
    keep = [k for k in range(len(idx)) if abs(xy[k, 1] - ruck_y) <= LINE_REACH]
    if len(keep) < MIN_LINE_PLAYERS:
        return None, []
    return float(np.median(xy[keep, 0])), [players[idx[k]] for k in keep]


def line_speeds(frames, fps, maps, events, teams):
    """Defensive line speed after each timed ruck (needs the pitch marked).
    Returns [{team, speed, frame, t, ruck, players}] - team = the defenders."""
    if not maps or len(teams) < 2:
        return []
    a, b = teams
    win = max(2, int(LINE_WINDOW * fps))
    edge = max(1, int(0.3 * fps))
    out = []
    for n, ev in enumerate(events):
        if ev["kind"] != "ruck" or not ev["timed"] or not ev["where"]:
            continue
        rx, ry = ev["where"]
        start = ev["end_f"] + 1
        if start + win >= len(frames):
            continue

        def mean_line(team, fs):
            vals, shown = [], []
            for f in fs:
                H = maps[f]
                if H is None:
                    continue
                x, pl = _team_line(frames[f], H, team, rx, ry)
                if x is not None:
                    vals.append(x)
                    shown = shown or pl
            return (sum(vals) / len(vals), shown) if len(vals) >= max(1, len(fs) // 2) else (None, [])

        first = range(start, start + edge)
        last = range(start + win - edge, start + win)
        xa0, pa = mean_line(a, first)
        xb0, pb = mean_line(b, first)
        if xa0 is None or xb0 is None:
            continue
        side_a, side_b = math.copysign(1, xa0 - rx), math.copysign(1, xb0 - rx)
        if side_a == side_b:
            continue  # both teams on the same side of the ruck: can't tell who defends
        # The defence stands flat, close to the ruck; the attack stands deeper.
        if abs(xa0 - rx) <= abs(xb0 - rx):
            team, x0, side, shown = a, xa0, side_a, pa
        else:
            team, x0, side, shown = b, xb0, side_b, pb
        x1, _ = mean_line(team, last)
        if x1 is None:
            continue
        dt = (win - edge) / fps
        speed = -side * (x1 - x0) / dt  # moving towards the ruck / the attack = positive
        if abs(speed) > MAX_SPEED:
            continue
        out.append({"team": team, "speed": round(speed, 1), "frame": start, "t": round(start / fps, 2),
                    "ruck": n, "players": shown})
    return out


def offside_at_rucks(frames, fps, maps, events, teams):
    """Rucks where a defender stood in front of the offside line (Law 15)
    just before the ball came out. Needs the pitch marked. Returns
    [{team, frame, t, players, metres}] - team = the defenders checked."""
    if not maps or len(teams) < 2:
        return []
    a, b = teams
    win = max(2, int(OFFSIDE_WINDOW * fps))
    out = []
    for ev in events:
        if ev["kind"] != "ruck" or not ev["timed"] or not ev["where"]:
            continue
        rx, ry = ev["where"]
        fs = [f for f in range(max(ev["group_f"], ev["end_f"] - win), ev["end_f"] + 1) if maps[f] is not None]
        if len(fs) < win // 2:
            continue
        # Who defends: the team whose line stands closest to the ruck (as for line speed).
        f_mid = fs[len(fs) // 2]
        xa, _ = _team_line(frames[f_mid], maps[f_mid], a, rx, ry)
        xb, _ = _team_line(frames[f_mid], maps[f_mid], b, rx, ry)
        if xa is None or xb is None or math.copysign(1, xa - rx) == math.copysign(1, xb - rx):
            continue
        team, side = (a, math.copysign(1, xa - rx)) if abs(xa - rx) <= abs(xb - rx) else (b, math.copysign(1, xb - rx))
        count, where = {}, {}
        for f in fs:
            players, groups, in_contact = frames[f][:3]
            ruck = min(groups, key=lambda g: abs(sum(players[i]["x"] for i in g) / len(g) - ev["box"][0] / 2 - ev["box"][2] / 2),
                       default=None)
            if not ruck:
                continue
            xy = pitch.project(maps[f], _feet(players, ruck))
            line = side * max(side * xy[:, 0])  # hindmost foot of the ruck on the defenders' side
            idx = [i for i, p in enumerate(players) if p["bucket"] == team and i not in in_contact]
            if not idx:
                continue
            pxy = pitch.project(maps[f], _feet(players, idx))
            for k, i in enumerate(idx):
                ahead = side * (line - pxy[k, 0])  # metres in front of the line, towards the ruck
                if ahead > OFFSIDE_MARGIN and OFFSIDE_SIDE < abs(pxy[k, 1] - ry) <= LINE_REACH:
                    key = players[i].get("id") or ("n", i)
                    count[key] = count.get(key, 0) + 1
                    where[key] = (f, players[i], round(float(ahead), 1))
        caught = [where[k] for k, n in count.items() if n >= OFFSIDE_SHARE * len(fs)]
        if caught:
            f = max(w[0] for w in caught)
            out.append({"team": team, "frame": f, "t": round(f / fps, 2), "players": [w[1] for w in caught],
                        "metres": max(w[2] for w in caught)})
        else:
            out.append({"team": team, "frame": None, "t": None, "players": [], "metres": 0})
    return out


def offside_point(checks, team, phase):
    """A coach-report point about offside at the ruck for `team`'s defence, or None."""
    if phase == "attack":
        return None
    mine = [c for c in checks if c["team"] == team]
    if len(mine) < 3:
        return None
    caught = [c for c in mine if c["frame"] is not None]
    T = team.capitalize()
    if not caught:
        return {"kind": "good", "title": "Onside at the rucks",
                "detail": f"At {len(mine)} rucks where {T} defended, nobody stood in front of the offside line "
                          f"in the last {OFFSIDE_WINDOW:.0f} s before the ball came out.",
                "why": "Staying onside keeps the penalty count down, and a line that starts level can move up together.",
                "moments": []}
    moments = [{"frame": c["frame"], "t": c["t"], "time": _fmt_t(c["t"]), "hl": _hl_players(c["players"]),
                "label": f"{len(c['players'])} player{'s' if len(c['players']) > 1 else ''} about {c['metres']:.0f} m in front of the offside line"}
               for c in sorted(caught, key=lambda c: -c["metres"])[:4]]
    return {"kind": "issue", "title": "Possible offside at the ruck",
            "detail": f"At {len(caught)} of {len(mine)} rucks, a {T} defender stood more than {OFFSIDE_MARGIN:.0f} m in front "
                      f"of the offside line (the hindmost foot of the ruck) just before the ball came out.",
            "why": "Referees penalise defenders who are in front of the hindmost foot, and it's where a lot of "
                   "breakdown penalties come from. Check each moment: from one camera, a player lying in the ruck can "
                   "look like the hindmost foot is further back than it is.",
            "moments": moments}


def _hl_players(players):
    if not players:
        return ""
    return "box:" + ",".join(str(int(v)) for v in (
        min(p["box"][0] for p in players), min(p["box"][1] for p in players),
        max(p["box"][2] for p in players), max(p["box"][3] for p in players)))


def _fmt_t(t):
    t = int(t)
    return f"{t // 60}:{t % 60:02d}"


def breakdown_summary(events, fps):
    """The report block about all breakdowns (both teams together)."""
    rucks = [e for e in events if e["kind"] == "ruck"]
    timed = [e for e in rucks if e["timed"]]
    times = [e["seconds"] for e in timed]
    s = {
        "rucks": len(rucks), "mauls": sum(1 for e in events if e["kind"] == "maul"),
        "scrums": sum(1 for e in events if e["kind"] == "scrum"),
        "timed": len(timed), "ball_seen": sum(1 for e in timed if e.get("ball_seen")),
        "median": round(median(times), 1) if times else None,
        "quick": sum(1 for t in times if t < QUICK_RUCK),
        "medium": sum(1 for t in times if QUICK_RUCK <= t <= SLOW_RUCK),
        "slow": sum(1 for t in times if t > SLOW_RUCK),
        "quick_limit": QUICK_RUCK, "slow_limit": SLOW_RUCK,
    }
    slowest = sorted((e for e in timed if e["seconds"] > QUICK_RUCK), key=lambda e: -e["seconds"])[:4]
    s["moments"] = [{
        "frame": e["mid_f"], "t": e["start"],
        "time": _fmt_t(e["start"]), "hl": "box:" + ",".join(str(int(v)) for v in e["mid_box"]),
        "label": f"Ruck lasted {e['seconds']:.1f} s",
    } for e in slowest]
    scrums = [e["seconds"] for e in events if e["kind"] == "scrum" and e["timed"]]
    s["scrum_median"] = round(median(scrums), 1) if scrums else None
    return s


def line_speed_point(speeds, team, opp, phase, fps):
    """A coach-report point about line speed for `team`'s section, or None."""
    T, O = team.capitalize(), (opp or "").capitalize()
    if phase == "attack":
        theirs = [s for s in speeds if s["team"] == opp]
        if len(theirs) < 2:
            return None
        med = median(s["speed"] for s in theirs)
        return {
            "kind": "info", "title": f"{O}'s defence line speed: {med:.1f} m/s",
            "detail": f"After {len(theirs)} rucks, {O}'s defensive line came up at {med:.1f} m/s on average "
                      f"in the first {LINE_WINDOW:.1f} s.",
            "why": "Against a fast line, receivers need to stand deeper or play flatter and earlier; "
                   "against a passive one, you can hold the ball longer and pick your pass.",
            "moments": [], "speed": round(med, 1),
        }
    mine = [s for s in speeds if s["team"] == team]
    if len(mine) < 2:
        return None
    med = median(s["speed"] for s in mine)
    slow = sorted(mine, key=lambda s: s["speed"])[:4]
    moments = [{"frame": s["frame"], "t": s["t"], "time": _fmt_t(s["t"]), "hl": _hl_players(s["players"]),
                "label": f"Line came up at {s['speed']:.1f} m/s" if s["speed"] > 0 else
                         f"Line went backwards ({s['speed']:.1f} m/s)"} for s in slow if s["speed"] < FAST_LINE]
    common = f"After {len(mine)} rucks where {T} defended, their line moved up at {med:.1f} m/s on average " \
             f"in the first {LINE_WINDOW:.1f} s after the ball came out."
    if med < PASSIVE_LINE:
        return {"kind": "issue", "title": "Passive defensive line speed", "detail": common,
                "why": "A line that waits gives the attack time and space to choose. Moving up together "
                       "as the ball leaves the ruck takes time away from the first receiver.",
                "moments": moments, "speed": round(med, 1)}
    if med >= FAST_LINE:
        return {"kind": "good", "title": "Good defensive line speed", "detail": common,
                "why": "Getting off the line quickly puts the attack under pressure. Keep checking the line "
                       "stays connected while it moves up.",
                "moments": [], "speed": round(med, 1)}
    return {"kind": "info", "title": "Steady defensive line speed", "detail": common,
            "why": f"Faster than {FAST_LINE:.1f} m/s puts more pressure on the attack; look at the slowest "
                   "moments to see who holds the line back.",
            "moments": moments, "speed": round(med, 1)}
