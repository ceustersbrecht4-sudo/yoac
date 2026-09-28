"""
Tactics a coach can pick for their team. For each one the report checks two
things from the tracking: did the team play it (in what share of the rucks),
and did it work (the success check that goes with it).

Everything is built on rugby.py's rucks. With the pitch marked, each ruck
has a place in metres, a defending side and a "next ruck" in the same
attack, which gives the gain line and where the ball went. Without the pitch
marked only the tactics that count players at the ruck or time it can be
checked, and then the coach's "attacking" / "defending" choice says which
team had the ball.

The thresholds come from coaching material where it gives numbers (quick
ball under 3 s, the 1 m / 5 m / 10 m law distances) and are otherwise rules
of thumb, named below so they can be tuned on real footage.
"""

import math
from statistics import median

import numpy as np

import pitch
import rugby

NEXT_RUCK = 15.0      # s: a ruck this soon after the last one is the next phase of the same attack
REACH = 30.0          # m across the pitch from the ruck that counts as "the line"
QUICK = rugby.QUICK_RUCK

# What a coach can pick. needs_metres: only with the pitch marked.
CATALOGUE = [
    # defence: how the line moves after the ruck (pick one)
    {"id": "blitz", "side": "defence", "group": "line", "name": "Blitz (rush)", "needs_metres": True,
     "about": "The whole line shoots up together as the ball leaves the ruck.",
     "played": "line comes up at 2.5 m/s or more in the first 1.5 s, without drifting",
     "worked": "the next ruck is no more than 1 m over the gain line"},
    {"id": "drift", "side": "defence", "group": "line", "name": "Drift", "needs_metres": True,
     "about": "The line slides towards touch and uses the touchline as an extra defender.",
     "played": "line spreads towards touch at 1 m/s or more, faster than it comes up",
     "worked": "the next ruck is 10 m or more wider and less than 3 m over the gain line"},
    {"id": "hold", "side": "defence", "group": "line", "name": "Passive hold", "needs_metres": True,
     "about": "A flat line that holds its ground (often in the 22 or when outnumbered).",
     "played": "line comes up slower than 1.5 m/s and doesn't drift",
     "worked": "the next ruck is less than 3 m over the gain line"},
    # defence: around the ruck (any)
    {"id": "gap_trap", "side": "defence", "group": "ruck", "name": "Gap trap", "needs_metres": True,
     "about": "Leave a gap next to the ruck on purpose, to invite a carry into a prepared tackle.",
     "played": "a gap over 2.2x the usual spacing 3-8 m from the ruck, with tight defenders either side of it",
     "worked": "the next ruck forms within 5 m of the last one, within 3 s, and at most 1 m over the gain line"},
    {"id": "fan", "side": "defence", "group": "ruck", "name": "Fan (don't contest the ruck)", "needs_metres": False,
     "about": "Only the tackler stays; everyone else fans out into the line.",
     "played": "1 defender or none in the ruck",
     "worked": "the defending line has at least as many players as the attack in view"},
    {"id": "compete", "side": "defence", "group": "ruck", "name": "Compete at every ruck", "needs_metres": False,
     "about": "The tackler or the first defender in attacks the ball (jackal or counter-ruck).",
     "played": "2 or more defenders in the ruck",
     "worked": "the ruck isn't quick ball (3 s or more), so the attack is slowed down"},
    {"id": "three_back", "side": "defence", "group": "back", "name": "Three in the backfield", "needs_metres": True,
     "about": "Three players drop back to cover kicks (back three / pendulum).",
     "played": "3 or more defenders 10 m or more behind the line at ball out",
     "worked": None},
    {"id": "two_back", "side": "defence", "group": "back", "name": "Two in the backfield", "needs_metres": True,
     "about": "Two players cover the backfield, one more in the line.",
     "played": "exactly 2 defenders 10 m or more behind the line at ball out",
     "worked": None},
    # attack
    {"id": "quick_tempo", "side": "attack", "group": "tempo", "name": "Quick tempo", "needs_metres": False,
     "about": "Recycle fast and play before the defence is set.",
     "played": "the ruck is quick ball, under 3 s",
     "worked": "with the pitch marked: the next ruck is over the gain line"},
    {"id": "pick_and_go", "side": "attack", "group": "carry", "name": "Pick and go", "needs_metres": True,
     "about": "Forwards pick from the base and drive close to the ruck, phase after phase.",
     "played": "3+ rucks in a row, each within 5 m across of the last, 0-5 m forward, within 3 s",
     "worked": "the run of rucks gains 5 m or more in total"},
    {"id": "off_9", "side": "attack", "group": "carry", "name": "Play off 9 (one-out runners)", "needs_metres": True,
     "about": "Direct carries one pass from the ruck.",
     "played": "the next ruck is 2-7 m across from the last, within 3 s",
     "worked": "the next ruck is 1 m or more over the gain line"},
    {"id": "off_10", "side": "attack", "group": "carry", "name": "Play off 10", "needs_metres": True,
     "about": "The ball goes to the first receiver and the attack starts wider.",
     "played": "the next ruck is 7-15 m across from the last",
     "worked": "the next ruck is over the gain line"},
    {"id": "wide", "side": "attack", "group": "carry", "name": "Wide / expansive", "needs_metres": True,
     "about": "Move the ball across the pitch to stretch the defence.",
     "played": "the next ruck is more than 20 m across from the last",
     "worked": "the next ruck is over the gain line"},
    {"id": "overload", "side": "attack", "group": "shape", "name": "Overload one side", "needs_metres": True,
     "about": "Stack attackers on one side to outnumber the defence there.",
     "played": "2+ more attackers than defenders on one side of the ruck at ball out",
     "worked": "the next ruck is on that side and over the gain line"},
    {"id": "flat", "side": "attack", "group": "depth", "name": "Flat attack", "needs_metres": True,
     "about": "Receivers close to the gain line, attacking the defence early.",
     "played": "the attackers stand less than 3 m behind the ruck, on average",
     "worked": "the next ruck is over the gain line"},
    {"id": "depth", "side": "attack", "group": "depth", "name": "Attack with depth", "needs_metres": True,
     "about": "Receivers stand deep and run onto the ball.",
     "played": "the attackers stand 5-10 m behind the ruck, on average",
     "worked": "the next ruck is over the gain line"},
    {"id": "pods_1331", "side": "attack", "group": "shape", "name": "1-3-3-1 pods", "needs_metres": True,
     "about": "Two pods of three in midfield and one player on each edge. Rough: the camera rarely sees the full width.",
     "played": "two groups of 2-4 attackers in the middle 40 m and one player alone within 15 m of a touchline",
     "worked": "the next ruck is over the gain line"},
    {"id": "pods_242", "side": "attack", "group": "shape", "name": "2-4-2 pods", "needs_metres": True,
     "about": "Two players on each edge and four in midfield. Rough: the camera rarely sees the full width.",
     "played": "a pair within 15 m of a touchline and a group of 3-5 in the middle",
     "worked": "the next ruck is over the gain line"},
]
BY_ID = {t["id"]: t for t in CATALOGUE}
PLAYED_ENOUGH = 0.6   # share of rucks that must show a tactic to call it "played"
PLAYED_ENOUGH_RARE = {"pick_and_go": 0.3, "wide": 0.25, "overload": 0.3, "gap_trap": 0.5, "pods_1331": 0.4, "pods_242": 0.4}
MIN_RUCKS = 3


def catalogue():
    """For the tactic picker: every tactic, with what it checks."""
    return CATALOGUE


# ------------------------------------------------------------ ruck contexts

def _mapped(maps, f, fps):
    """The pitch mapping for frame f, or the nearest one within 0.3 s."""
    for d in range(0, max(1, int(0.3 * fps)) + 1):
        for g in (f - d, f + d):
            if maps and 0 <= g < len(maps) and maps[g] is not None:
                return g, maps[g]
    return f, None


def _ruck_group(frame, ev):
    players, groups = frame[0], frame[1]
    cx = (ev["box"][0] + ev["box"][2]) / 2
    return min(groups, key=lambda g: abs(sum(players[i]["x"] for i in g) / len(g) - cx), default=None)


def _in_ruck(frames, ev, team):
    """Median number of `team` players in the ruck over its middle half."""
    n = ev["end_f"] - ev["group_f"] + 1
    counts = []
    for f in range(ev["group_f"] + n // 4, ev["end_f"] - n // 4 + 1):
        g = _ruck_group(frames[f], ev)
        if g:
            counts.append(sum(1 for i in g if frames[f][0][i]["bucket"] == team))
    return median(counts) if counts else None


def _positions(frame, H, team):
    """Pitch x, y of `team`'s players outside contact, as an array."""
    players, _, in_contact = frame[:3]
    idx = [i for i, p in enumerate(players) if p["bucket"] == team and i not in in_contact]
    if not idx or H is None:
        return np.zeros((0, 2)), idx
    return pitch.project(H, [(players[i]["x"], players[i]["y"]) for i in idx]), idx


def contexts(frames, fps, maps, events, teams, phase_team=None, phase=None):
    """One entry per ruck: who defended, where, and the next ruck of the same
    attack (with the gain line and how far across the ball went)."""
    a, b = (teams + [None, None])[:2]
    out = []
    for ev in events:
        if ev["kind"] != "ruck":
            continue
        c = {"ev": ev, "defender": None, "attacker": None, "side": None, "next": None}
        f, H = _mapped(maps, ev["end_f"], fps)
        if H is not None and ev["where"] and a and b:
            rx, ry = ev["where"]
            xa, _ = rugby._team_line(frames[f], H, a, rx, ry)
            xb, _ = rugby._team_line(frames[f], H, b, rx, ry)
            if xa is not None and xb is not None and math.copysign(1, xa - rx) != math.copysign(1, xb - rx):
                d = a if abs(xa - rx) <= abs(xb - rx) else b
                c.update(defender=d, attacker=b if d == a else a, side=math.copysign(1, (xa if d == a else xb) - rx),
                         f=f, H=H, rx=rx, ry=ry)
        if c["defender"] is None and phase_team and phase in ("attack", "defence"):
            other = b if phase_team == a else a
            c["defender"], c["attacker"] = (other, phase_team) if phase == "attack" else (phase_team, other)
        out.append(c)
    # Link each ruck to the next one of the same attack.
    for c, n in zip(out, out[1:]):
        if c["side"] is None or n["side"] is None or n["attacker"] != c["attacker"]:
            continue
        gap = (n["ev"]["start_f"] - c["ev"]["end_f"]) / fps
        if 0 <= gap <= NEXT_RUCK:
            c["next"] = n
            c["gain"] = c["side"] * (n["rx"] - c["rx"])     # metres forward for the attack
            c["across"] = n["ry"] - c["ry"]
            c["gap_s"] = gap
    return out


# ------------------------------------------------------------ per tactic

def _defence_line(c, frames):
    xy, _ = _positions(frames[c["f"]], c["H"], c["defender"])
    keep = [k for k in range(len(xy)) if abs(xy[k, 1] - c["ry"]) <= REACH and c["side"] * (xy[k, 0] - c["rx"]) > -1]
    return xy[keep] if keep else np.zeros((0, 2))


def _check(tid, c, frames, fps, speeds, width):
    """(played, worked) for one ruck; None where it can't be judged."""
    ev = c["ev"]
    gain = c.get("gain")
    if tid in ("blitz", "drift", "hold"):
        s = next((s for s in speeds if s["ruck_ev"] is ev), None) if speeds else None
        if not s:
            return None, None
        played = s["style"] == tid
        if gain is None:
            return played, None
        if tid == "blitz":
            return played, gain <= 1.0
        if tid == "drift":
            return played, abs(c["across"]) >= 10 and gain < 3
        return played, gain < 3
    if tid in ("fan", "compete"):
        n = _in_ruck(frames, ev, c["defender"])
        if n is None:
            return None, None
        if tid == "fan":
            played = n <= 1
            worked = None
            if c["side"] is not None:
                d = len(_defence_line(c, frames))
                att, _ = _positions(frames[c["f"]], c["H"], c["attacker"])
                att = [p for p in att if abs(p[1] - c["ry"]) <= REACH]
                worked = d >= len(att) if att else None
            return played, worked
        return n >= 2, (ev["seconds"] >= QUICK) if ev["timed"] else None
    if tid == "quick_tempo":
        if not ev["timed"]:
            return None, None
        return ev["seconds"] < QUICK, (gain > 0) if gain is not None else None
    if c["side"] is None:
        return None, None  # everything below needs the pitch marked
    if tid in ("three_back", "two_back"):
        line = _defence_line(c, frames)
        if len(line) < 3:
            return None, None
        depth = c["side"] * (line[:, 0] - c["rx"])
        front = np.median(np.sort(depth)[:max(3, len(depth) // 2)])
        back = int(np.sum(depth - front >= 10))
        return (back >= 3) if tid == "three_back" else (back == 2), None
    if tid == "gap_trap":
        line = _defence_line(c, frames)
        if len(line) < 4:
            return None, None
        front = np.median(c["side"] * (line[:, 0] - c["rx"]))
        line = line[np.abs(c["side"] * (line[:, 0] - c["rx"]) - front) <= 5]  # the front line, not the backfield
        played = False
        for sign in (-1, 1):
            ys = sorted(sign * (y - c["ry"]) for y in line[:, 1] if sign * (y - c["ry"]) > 0)
            if len(ys) < 3:
                continue
            gaps = [ys[0]] + [q - p for p, q in zip(ys, ys[1:])]  # from the ruck out
            usual = median(gaps[1:]) if len(gaps) > 2 else median(gaps)
            for k, g in enumerate(gaps):
                start = ys[k - 1] if k else 0.0
                mid = start + g / 2
                inner_ok = k == 0 or gaps[k - 1] <= 1.2 * usual
                outer_ok = k + 1 >= len(gaps) or gaps[k + 1] <= 1.2 * usual
                if g >= max(2.2 * usual, 3.0) and 3 <= mid <= 8 and inner_ok and outer_ok:
                    played = True
        worked = None
        if played and c["next"] is not None:
            n = c["next"]
            worked = (math.hypot(n["rx"] - c["rx"], n["ry"] - c["ry"]) <= 5 and c["gap_s"] <= 3 and gain <= 1.0)
        return played, worked
    if tid == "pick_and_go":
        return None, None  # judged over runs of rucks, see below
    if tid in ("off_9", "off_10", "wide"):
        if c["next"] is None:
            return None, None
        a = abs(c["across"])
        played = (2 <= a <= 7 and c["gap_s"] <= 3) if tid == "off_9" else (7 < a <= 15) if tid == "off_10" else a > 20
        return played, (gain >= 1.0 if tid == "off_9" else gain > 0)
    att, _ = _positions(frames[c["f"]], c["H"], c["attacker"])
    att = att[np.abs(att[:, 1] - c["ry"]) <= REACH] if len(att) else att
    if tid in ("flat", "depth"):
        behind = [-c["side"] * (x - c["rx"]) for x, _ in att if 0 <= -c["side"] * (x - c["rx"]) <= 25]
        if len(behind) < 3:
            return None, None
        d = median(behind)
        played = d < 3 if tid == "flat" else 5 <= d <= 10
        return played, (gain > 0) if gain is not None else None
    if tid == "overload":
        dfd = _defence_line(c, frames)
        best = None
        for sign in (-1, 1):
            na = sum(1 for _, y in att if sign * (y - c["ry"]) > 1)
            nd = sum(1 for _, y in dfd if sign * (y - c["ry"]) > 1)
            if na - nd >= 2:
                best = sign
        if best is None:
            return False, None
        if c["next"] is None:
            return True, None
        return True, (math.copysign(1, c["across"]) == best and gain > 0)
    if tid in ("pods_1331", "pods_242"):
        xy, _ = _positions(frames[c["f"]], c["H"], c["attacker"])
        ys = sorted(float(y) for x, y in xy if -c["side"] * (x - c["rx"]) >= -1)
        if len(ys) < 5:
            return None, None
        groups, cur = [], [ys[0]]
        for p, q in zip(ys, ys[1:]):
            if q - p > 5:
                groups.append(cur)
                cur = []
            cur.append(q)
        groups.append(cur)
        W = width or 70.0
        mid = [g for g in groups if 15 <= median(g) <= W - 15]
        edge = [g for g in groups if median(g) < 15 or median(g) > W - 15]
        if tid == "pods_1331":
            played = sum(1 for g in mid if 2 <= len(g) <= 4) >= 2 and any(len(g) == 1 for g in edge)
        else:
            played = any(len(g) == 2 for g in edge) and any(3 <= len(g) <= 5 for g in mid)
        return played, (gain > 0) if gain is not None else None
    return None, None


def _pick_and_go(ctx, team):
    """Runs of 3+ close-in rucks in a row by `team`: (rucks in runs, rucks judged, runs, runs that gained 5 m+)."""
    mine = [c for c in ctx if c["attacker"] == team and c["side"] is not None]
    in_run, runs, worked, run = set(), 0, 0, []
    for c in mine + [None]:
        close = c is not None and c["next"] is not None and abs(c["across"]) <= 5 and 0 <= c["gain"] <= 5 and c["gap_s"] <= 3
        if close:
            run.append(c)
            continue
        if run and c is not None:
            run.append(c)  # the ruck the last close carry ended in
        if len(run) >= 3:
            runs += 1
            in_run.update(id(r) for r in run)
            if sum(r.get("gain", 0) for r in run if r.get("gain") is not None) >= 5:
                worked += 1
        run = []
    return len(in_run), len(mine), runs, worked


def evaluate(chosen, frames, fps, maps, events, speeds, team, opp, phase, width=None):
    """Report points for the tactics `team`'s coach picked."""
    if not chosen:
        return []
    for s in speeds or []:
        s.setdefault("ruck_ev", None)
    for s in speeds or []:
        if s.get("ruck") is not None and s["ruck"] < len(events):
            s["ruck_ev"] = events[s["ruck"]]
    ctx = contexts(frames, fps, maps, events, [team, opp] if opp else [team], team, phase)
    T = team.capitalize()
    points = []
    for tid in chosen:
        t = BY_ID.get(tid)
        if not t:
            continue
        mine = [c for c in ctx if (c["defender"] if t["side"] == "defence" else c["attacker"]) == team]
        if t["needs_metres"] and not maps:
            points.append(_point(t, "info", f"{t['name']}: mark the pitch to check it",
                                 "This tactic is measured in metres. Mark the pitch in the Pitch tab and make the report again.", []))
            continue
        if tid == "pick_and_go":
            n_in, n, runs, ok = _pick_and_go(ctx, team)
            if n < MIN_RUCKS:
                points.append(_few(t, T, n))
                continue
            share = n_in / n
            target = PLAYED_ENOUGH_RARE.get(tid, PLAYED_ENOUGH)
            kind = "good" if share >= target and (not runs or ok / runs >= 0.5) else "issue"
            points.append(_point(t, kind, f"{t['name']}: {share:.0%} of rucks in a pick-and-go run",
                                 f"{runs} run{'s' if runs != 1 else ''} of 3+ close-in rucks, {ok} gained 5 m or more. "
                                 f"Played means: {t['played']}.", []))
            continue
        judged, played, worked, tried, misses = 0, 0, 0, 0, []
        for c in mine:
            p, w = _check(tid, c, frames, fps, speeds, width)
            if p is None:
                continue
            judged += 1
            ev = c["ev"]
            hl = "box:" + ",".join(str(int(v)) for v in ev["mid_box"])
            if p:
                played += 1
                if w is not None:
                    tried += 1
                    worked += bool(w)
                    if not w:
                        misses.append({"frame": ev["end_f"], "t": ev["end"], "time": rugby._fmt_t(ev["end"]), "hl": hl,
                                       "label": f"{t['name']} played, didn't work"})
            else:
                misses.append({"frame": ev["end_f"], "t": ev["end"], "time": rugby._fmt_t(ev["end"]), "hl": hl,
                               "label": f"{t['name']} not played"})
        if judged < MIN_RUCKS:
            points.append(_few(t, T, judged))
            continue
        share = played / judged
        target = PLAYED_ENOUGH_RARE.get(tid, PLAYED_ENOUGH)
        works = worked / tried if tried else None
        kind = "good" if share >= target and (works is None or works >= 0.5) else "issue"
        detail = f"Played at {played} of {judged} rucks ({share:.0%}); played means {t['played']}."
        if works is not None:
            detail += f" It worked {worked} of {tried} times: {t['worked']}."
        if share < target:
            title = f"{t['name']}: only played at {share:.0%} of rucks"
        elif works is not None and works < 0.5:
            title = f"{t['name']}: played, but worked only {works:.0%} of the time"
        else:
            title = f"{t['name']}: stuck at {share:.0%} of rucks" + (f", worked {works:.0%}" if works is not None else "")
        points.append(_point(t, kind, title, detail, misses[:4]))
    return points


def _few(t, T, n):
    return _point(t, "info", f"{t['name']}: not enough rucks to judge",
                  f"Only {n} ruck{'s' if n != 1 else ''} where {T} {'defended' if t['side'] == 'defence' else 'attacked'} "
                  f"could be checked (at least {MIN_RUCKS} needed).", [])


def _point(t, kind, title, detail, moments):
    return {"kind": kind, "title": title, "detail": detail, "moments": moments, "tactic": t["id"],
            "why": f"Your plan: {t['about']}" + (" Checked from the video; confirm each moment before acting on it."
                                                  if kind != "info" else "")}
