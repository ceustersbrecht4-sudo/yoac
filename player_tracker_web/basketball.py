"""
Basketball layer of the coach report.

Tracking, teams and court marking are shared with rugby and soccer; this
module turns them into basketball, in metres on the court:

  - who has the ball and which basket each team attacks
  - half-court sets and transition (fast breaks, getting back)
  - offence: spacing, the corners, players parked in the key (3 seconds),
    getting over halfway (8 seconds), using the shot clock (24 seconds)
  - defence: goal-side (between your player and the basket), help in the
    paint, pressure on the ball, man-to-man or zone (and which zone)
  - screens (pick-and-roll) and how they were defended: switch, hedge or
    trap, drop, or fight over
  - free throws
  - the tactics a coach can pick

Numbers come from the rules (FIBA / NBA: 3 s in the key, 8 s to cross
halfway, 24 s shot clock; the court sizes in pitch.py) and coaching rules
of thumb (spacing of 4.5-5.5 m, 15-18 ft), named below so they can be
tuned on real footage. One camera from the side sees most of the court, but
players hide behind each other, so every count is "of the players in view".
"""

import math
from statistics import median

import cv2
import numpy as np

import analysis
import pitch

TEAM_SKIP = ("other", "unsure", "unknown", "ball")

OWN_REACH = 2.0         # m from the ball (seen in the air or in hands, so not exact) = has it
OWN_FRAMES = 5          # frames in a row before possession changes
OWN_HOLD = 2.0          # s without anyone on the ball before falling back on where the players stand
HALF_COURT = 8          # of the players in view, this many in one half = a half-court set
MIN_PLAYERS = 6         # players in view before a frame counts
SPACING_BAD, SPACING_GOOD = 3.5, 4.5   # m to the nearest teammate on offence (good spacing 4.5-5.5 m)
CROWDED = 2.5           # m: two attackers this close for CROWDED_TIME (a screen is shorter)
CROWDED_TIME = 2.0
CORNER_X, CORNER_Y = 4.2, 3.0   # m from the baseline / sideline = in the corner
THREE_SECONDS = 3.0     # s an attacker may stay in the key
EIGHT_SECONDS = 8.0     # s to get the ball over halfway
SHOT_CLOCK = 24.0
LATE_CLOCK = 20.0       # s into a possession = late in the shot clock
FAST_BREAK = 6.0        # s from gaining the ball to a shot-ready spot in the frontcourt = pushed the pace
GOAL_SIDE_BAD, GOAL_SIDE_GOOD = 0.6, 0.8
HELP_R = 1.0            # m around the key that still counts as help in the paint
BALL_AWAY = 6.0         # m from the basket: the ball is outside, help should be in the paint
ON_BALL = 2.0           # m: a defender this close is on the ball
SCREEN_R = 1.3          # m: a screener this close to the ball handler's defender...
SCREEN_STILL = 0.5      # m moved in SCREEN_TIME = standing still (a legal screen)
SCREEN_TIME = 0.5       # s
SCREEN_AFTER = 1.5      # s after the handler comes off the screen to read the coverage
SCREEN_USE = 1.5        # m the handler moves = they've used the screen...
SCREEN_WAIT = 3.0       # s ...within this of it being set (else it wasn't used)
SCREEN_WORKED = 3.0     # s to get the ball (or the handler) to the key
MARK_R = 2.5            # m from one attacker, following them = man-to-man
MAN_SHARE, ZONE_SHARE = 0.6, 0.3
ZONE_SPLIT = 1.5        # m between defenders' distances from the baseline = a new line of the zone
FT_STILL = 0.4          # m moved in 1 s = standing for a free throw
FT_TIME = 2.0           # s the lane lines stand still
MIN_EVENTS = 3
PLAYED_ENOUGH = 0.6


# ================================================================ context

def _view(H, size):
    if size is None:
        return None
    w, h = size
    c = pitch.project(H, [(0, 0), (w, 0), (w, h), (0, h)])
    return c.astype(np.float32).reshape(-1, 1, 2) if np.all(np.isfinite(c)) else None


def _ball(frame, H, L, W):
    balls = sorted(frame[4], key=lambda b: -b[2])
    if not balls:
        return None
    for x, y in pitch.project(H, [(b[0], b[1]) for b in balls]):
        if -1 <= x <= L + 1 and -1 <= y <= W + 1:
            return float(x), float(y)
    return None


def context(frames, fps, maps, calib, teams, size=None):
    dims = pitch.SIZES["basketball"]
    L = float((calib or {}).get("length") or dims["length"])
    W = float((calib or {}).get("width") or dims["width"])
    n = len(frames)
    c = {"fps": fps, "L": L, "W": W, "n": n, "teams": list(teams), "mapped": [], "pl": [None] * n,
         "ball": [None] * n, "owner": [None] * n, "off": [None] * n, "basket": {t: None for t in teams},
         "half": [False] * n, "spells": [], "ball_share": 0}
    if not maps or len(teams) < 2:
        return c
    for f, H in enumerate(maps):
        if H is None or not frames[f][0]:
            continue
        players = frames[f][0]
        xy = pitch.project(H, [(p["x"], p["y"]) for p in players])
        c["pl"][f] = [(p["id"], p["bucket"], float(x), float(y), i) for i, (p, (x, y)) in enumerate(zip(players, xy))
                      if p["bucket"] in teams and -1.5 <= x <= L + 1.5 and -1.5 <= y <= W + 1.5]
        c["ball"][f] = _ball(frames[f], H, L, W)
        if len(c["pl"][f]) >= MIN_PLAYERS:
            c["mapped"].append(f)
        b = c["ball"][f]
        if b is not None:
            near = min(((math.hypot(x - b[0], y - b[1]), tid, t) for tid, t, x, y, _ in c["pl"][f]), default=None)
            if near and near[0] <= OWN_REACH:
                c["owner"][f] = (near[1], near[2])
    _offence(c)
    return c


def _offence(c):
    """Which team is attacking, frame by frame, and which basket each team
    attacks. The ball says who has it; without the ball, defenders stand
    between their player and the basket, so the team nearer the basket of
    the half the players are in is defending."""
    fps, L, W, teams = c["fps"], c["L"], c["W"], c["teams"]
    struct = [None] * c["n"]
    votes = {t: [0, 0] for t in teams}   # [attacks left, attacks right]
    for f in c["mapped"]:
        pl = c["pl"][f]
        mx = median(x for _, _, x, _, _ in pl)
        right = mx > L / 2
        bx = L - pitch.BASKET if right else pitch.BASKET
        if sum(1 for _, _, x, _, _ in pl if (x > L / 2) == right) < HALF_COURT * len(pl) / 10:
            continue
        dist = {t: [math.hypot(x - bx, y - W / 2) for _, tm, x, y, _ in pl if tm == t] for t in teams}
        if all(dist[t] for t in teams):
            d = min(teams, key=lambda t: median(dist[t]))
            o = next(t for t in teams if t != d)
            struct[f] = o
            votes[o][int(right)] += 1
    # Possession from the ball, held through short gaps (as in soccer.py).
    cur, cand, streak, last = None, None, 0, -10 ** 9
    seen = 0
    for f in range(c["n"]):
        o = c["owner"][f][1] if c["owner"][f] else None
        if o:
            seen += 1
            last = f
            if o == cur:
                cand, streak = None, 0
            else:
                streak = streak + 1 if o == cand else 1
                cand = o
                if streak >= OWN_FRAMES:
                    cur, cand, streak = o, None, 0
        elif f - last > OWN_HOLD * fps:
            cur = None
        c["off"][f] = cur or struct[f]
    c["ball_share"] = seen / len(c["mapped"]) if c["mapped"] else 0
    for t in teams:
        if sum(votes[t]):
            c["basket"][t] = (L - pitch.BASKET if votes[t][1] > votes[t][0] else pitch.BASKET, W / 2)
    a, b = teams
    # Each team attacks the other end: fill in one from the other if needed.
    if c["basket"][a] and not c["basket"][b]:
        c["basket"][b] = (L - c["basket"][a][0], W / 2)
    if c["basket"][b] and not c["basket"][a]:
        c["basket"][a] = (L - c["basket"][b][0], W / 2)
    for f in c["mapped"]:
        o = c["off"][f]
        if o and c["basket"][o]:
            right = c["basket"][o][0] > L / 2
            c["half"][f] = sum(1 for _, _, x, _, _ in c["pl"][f] if (x > L / 2) == right) >= HALF_COURT * len(c["pl"][f]) / 10
    # Spells of possession: [(first, last, team)]
    start, team = None, None
    for f in range(c["n"] + 1):
        o = c["off"][f] if f < c["n"] else None
        if o != team:
            if team is not None and f - start >= fps:
                c["spells"].append((start, f - 1, team))
            start, team = f, o


def _base(c, team):
    """From the baseline `team` attacks: (metres from that baseline, metres from the middle line across)."""
    b = c["basket"].get(team)
    if not b:
        return None
    right = b[0] > c["L"] / 2
    return lambda x, y: (c["L"] - x if right else x, y - c["W"] / 2)


def _in_key(bx, dy, pad=0.0):
    return bx <= pitch.KEY_DEPTH + pad and abs(dy) <= pitch.KEY_HALF + pad


def _hl(frames, f, idx):
    ps = [frames[f][0][i] for i in idx if i < len(frames[f][0])]
    return analysis._hl_bounds(ps) if ps else ""


def _m(frames, fps, f, idx, label):
    return {"frame": f, "t": round(f / fps, 2), "time": analysis._fmt_t(f / fps), "hl": _hl(frames, f, idx), "label": label}


def _pt(kind, title, detail, why, moments=()):
    return {"kind": kind, "title": title, "detail": detail, "why": why, "moments": list(moments)}


# ================================================================== measures

def offence_frames(c, team):
    return [f for f in c["mapped"] if c["off"][f] == team and c["half"][f]]


def defence_frames(c, team):
    return [f for f in c["mapped"] if c["off"][f] and c["off"][f] != team and c["half"][f]]


def _spacing(c, f, team):
    pl = [(x, y) for _, t, x, y, _ in c["pl"][f] if t == team]
    if len(pl) < 4:
        return None
    return median(min(math.hypot(x - a, y - b) for a, b in pl if (a, b) != (x, y)) for x, y in pl)


def _goal_side(c, f, team, opp):
    """Share of the attackers in view who have a defender between them and the basket."""
    base = _base(c, opp)
    if not base:
        return None
    att = [base(x, y) for _, t, x, y, _ in c["pl"][f] if t == opp]
    dfn = [base(x, y) for _, t, x, y, _ in c["pl"][f] if t == team]
    if len(att) < 3 or not dfn:
        return None
    bx = pitch.BASKET
    ok = 0
    for ax, ay in att:
        near = min(dfn, key=lambda d: math.hypot(d[0] - ax, d[1] - ay))
        ok += math.hypot(near[0] - bx, near[1]) < math.hypot(ax - bx, ay)
    return ok / len(att)


def screens(c, frames):
    """Ball screens: an attacker standing still right next to the defender of
    the player with the ball. With how the defence played it."""
    fps = c["fps"]
    out, last = [], {}
    k = max(1, int(SCREEN_TIME * fps))
    pos = lambda f, tid: next(((x, y, t) for i, t, x, y, _ in (c["pl"][f] or []) if i == tid), None)
    for f in range(k, c["n"]):
        o = c["owner"][f]
        if not o or not c["pl"][f]:
            continue
        handler, team = o
        if c["off"][f] != team:
            continue
        h = pos(f, handler)
        dfn = [(math.hypot(x - h[0], y - h[1]), i) for i, t, x, y, _ in c["pl"][f] if t != team]
        if not dfn:
            continue
        dd, d = min(dfn)
        if dd > MARK_R:
            continue
        dp = pos(f, d)
        for i, t, x, y, _ in c["pl"][f]:
            if t != team or i == handler or math.hypot(x - dp[0], y - dp[1]) > SCREEN_R:
                continue
            before = pos(f - k, i)
            if not before or math.hypot(x - before[0], y - before[1]) > SCREEN_STILL:
                continue
            if f - last.get(team, -10 ** 9) < 2 * fps:
                break
            last[team] = f
            # The screener's own defender, before the screen.
            sd = min(((math.hypot(a - x, b - y), j) for j, tm, a, b, _ in c["pl"][f] if tm != team and j != d), default=None)
            sd = sd[1] if sd and sd[0] <= MARK_R + 1 else None
            out.append(_coverage(c, frames, f, team, handler, i, d, sd))
            break
    return [s for s in out if s]


def _coverage(c, frames, f, team, handler, screener, d, sd):
    fps = c["fps"]
    base = _base(c, team)
    if not base:
        return None
    pos = lambda g, tid: next(((x, y) for i, t, x, y, _ in (c["pl"][g] or []) if i == tid), None)
    h0 = pos(f, handler)
    use = next((g for g in range(f, min(c["n"], f + int(SCREEN_WAIT * fps) + 1))
                if pos(g, handler) and math.hypot(pos(g, handler)[0] - h0[0], pos(g, handler)[1] - h0[1]) >= SCREEN_USE), None)
    if use is None:
        return None
    two, swapped, deep, seen = 0, 0, 0, 0
    reached = False
    for g in range(use, min(c["n"], use + int(SCREEN_WORKED * fps) + 1)):
        h = pos(g, handler)
        if not h:
            continue
        hb = base(*h)
        b = c["ball"][g]
        if (b is not None and _in_key(*base(*b))) or _in_key(*hb):
            reached = True
        if g > use + SCREEN_AFTER * fps:
            continue
        seen += 1
        near = sorted((math.hypot(x - h[0], y - h[1]), i) for i, t, x, y, _ in (c["pl"][g] or []) if t != team)
        if len([1 for dist, _ in near if dist <= ON_BALL + 0.5]) >= 2:
            two += 1
        if sd is not None and near and near[0][1] == sd and pos(g, d) and math.hypot(pos(g, d)[0] - h[0], pos(g, d)[1] - h[1]) > MARK_R:
            swapped += 1
        sp = pos(g, sd) if sd is not None else None
        if sp:
            sb = base(*sp)
            if sb[0] < hb[0] - 1.5 and math.hypot(sp[0] - h[0], sp[1] - h[1]) > 3.0:
                deep += 1
    if not seen:
        return None
    if two >= 0.3 * seen:
        cov = "hedge"
    elif swapped >= 0.5 * seen:
        cov = "switch"
    elif deep >= 0.5 * seen:
        cov = "drop"
    else:
        cov = "over"
    idx = [i for tid, t, x, y, i in c["pl"][f] if tid in (handler, screener, d)]
    return {"f": f, "t": round(f / fps, 2), "team": team, "coverage": cov, "worked": reached, "hl": _hl(frames, f, idx)}


def _zone_shape(c, f, team, opp):
    """Lines of the defence from the baseline (e.g. '2-3'), read from back to front reversed (front first, like coaches)."""
    base = _base(c, opp)
    if not base:
        return None
    xs = sorted(base(x, y)[0] for _, t, x, y, _ in c["pl"][f] if t == team)
    if len(xs) != 5:
        return None
    lines, cur = [], [xs[0]]
    for a, b in zip(xs, xs[1:]):
        if b - a > ZONE_SPLIT:
            lines.append(cur)
            cur = []
        cur.append(b)
    lines.append(cur)
    return "-".join(str(len(l)) for l in reversed(lines)) if 2 <= len(lines) <= 3 else None


def _follow(c, f, team, opp):
    """Share of `team`'s players marking the same opponent as a second earlier."""
    g = f - int(c["fps"])
    if g < 0 or not c["pl"][g]:
        return None

    def marks(h):
        theirs = [(i, x, y) for i, t, x, y, _ in c["pl"][h] if t == opp]
        out = {}
        for i, t, x, y, _ in c["pl"][h]:
            if t == team and theirs:
                d, j = min((math.hypot(x - a, y - b), j) for j, a, b in theirs)
                out[i] = j if d <= MARK_R else None
        return out
    now, before = marks(f), marks(g)
    if len(now) < 3:
        return None
    return sum(1 for k, v in now.items() if v is not None and before.get(k) == v) / len(now)


def free_throws(c, frames):
    """Free throws: players standing still along both sides of the key, one at the line."""
    fps = c["fps"]
    out, run = [], []
    k = max(1, int(fps))
    for f in range(k, c["n"]):
        pl, before = c["pl"][f], c["pl"][f - k]
        hit = None
        if pl and before:
            prev = {i: (x, y) for i, t, x, y, _ in before}
            for side in (pitch.BASKET, c["L"] - pitch.BASKET):
                right = side > c["L"] / 2
                bx = lambda x: c["L"] - x if right else x
                still = [(i, t, x, y) for i, t, x, y, _ in pl if i in prev and math.hypot(x - prev[i][0], y - prev[i][1]) <= FT_STILL]
                lane = [s for s in still if 1.5 <= bx(s[2]) <= 5.5 and 2.0 <= abs(s[3] - c["W"] / 2) <= 3.6]
                line = [s for s in still if 5.0 <= bx(s[2]) <= 6.8 and abs(s[3] - c["W"] / 2) <= 1.2]
                if len(lane) >= 4 and line:
                    hit = (line[0][1], side)
                    break
        if hit:
            run.append((f, hit))
        else:
            if len(run) >= FT_TIME * fps:
                f0, (team, side) = run[0]
                out.append({"f": f0, "t": round(f0 / fps, 2), "team": team, "hl": _hl(frames, f0, [i for _, _, _, _, i in c["pl"][f0]])})
            run = []
    if len(run) >= FT_TIME * fps:
        f0, (team, side) = run[0]
        out.append({"f": f0, "t": round(f0 / fps, 2), "team": team, "hl": ""})
    return out


# ================================================================== report

def analyse_team(c, frames, team, opp, phase, extra):
    fps = c["fps"]
    T, O = team.capitalize(), (opp or "").capitalize()
    seen = [sum(1 for p in fr[0] if p["bucket"] == team) for fr in frames if fr[0]]
    seen = [n for n in seen if n]
    stats = {"players_in_view": round(sum(seen) / len(seen), 1) if seen else 0}
    points, notes = [], []
    base = {"team": team, "opponent": opp, "phase": phase, "stats": stats, "points": points, "notes": notes}
    if not c["mapped"]:
        notes.append("Mark the court in the Pitch tab to get the basketball report: it measures in metres where the "
                     "players stand against the basket - spacing, the key, help defence, screens and transition.")
        return base
    wide = sum(1 for fr in frames if fr[0])
    if wide:
        notes.append(f"{len(c['mapped']) / wide:.0%} of the footage could be measured on the court.")
    if c["ball_share"] < 0.3:
        notes.append("The ball was rarely seen with a player, so who was attacking comes mostly from where the players "
                     "stood (defenders stand between their player and the basket).")
    off_f, def_f = offence_frames(c, team), defence_frames(c, team)
    mine = [s for s in c["spells"] if s[2] == team]
    if mine:
        lengths = [(b - a + 1) / fps for a, b, _ in mine]
        stats["possession_length"] = round(median(lengths), 1)
    if phase in ("attack", "mixed") and off_f:
        sp = [(f, _spacing(c, f, team)) for f in off_f]
        sp = [(f, v) for f, v in sp if v is not None]
        if len(sp) >= 2 * fps:
            ms = median(v for _, v in sp)
            stats["spacing"] = round(ms, 1)
            if ms < SPACING_BAD:
                points.append(_pt("issue", "Crowded spacing", f"In the half-court {T}'s players were typically {ms:.1f} m from the "
                                  "nearest teammate.", "Good spacing is about 4.5-5.5 m (15-18 ft): one defender can't guard two "
                                  "attackers, and drives and passes have room. Fill the spots, then move.",
                                  [_m(frames, fps, f, [], "Crowded") for f, v in sorted(sp, key=lambda t: t[1])[:1]]))
            elif ms >= SPACING_GOOD:
                points.append(_pt("good", "Good spacing", f"In the half-court {T}'s players were typically {ms:.1f} m apart.",
                                  "4.5-5.5 m between players stretches the defence and opens driving lanes."))
        corners = [f for f in off_f if any(_base(c, team)(x, y)[0] <= CORNER_X and (y <= CORNER_Y or y >= c["W"] - CORNER_Y)
                                           for _, t, x, y, _ in c["pl"][f] if t == team)]
        if len(off_f) >= 2 * fps:
            points.append(_pt("info", f"A player in the corner {len(corners) / len(off_f):.0%} of the time",
                              f"In the half-court {T} had someone in a corner (within {CORNER_X:.0f} m of the baseline and "
                              f"{CORNER_Y:.0f} m of the sideline) {len(corners) / len(off_f):.0%} of the time.",
                              "The corner three is the shortest three, and a player there pulls a help defender out of the paint."))
        # 3 seconds in the key
        stay, flags = {}, []
        for f in off_f:
            bs = _base(c, team)
            here = {i for i, t, x, y, _ in c["pl"][f] if t == team and _in_key(*bs(x, y))}
            for i in list(stay):
                if i not in here and f - stay[i][1] > 0.3 * fps:
                    s0, s1 = stay.pop(i)
                    if s1 - s0 > THREE_SECONDS * fps:
                        flags.append((s0, s1, i))
            for i in here:
                stay[i] = [stay[i][0], f] if i in stay else [f, f]
        flags += [(a, b, i) for i, (a, b) in stay.items() if b - a > THREE_SECONDS * fps]
        if flags:
            def idx(f, tid):
                return [k for i, t, x, y, k in (c["pl"][f] or []) if i == tid]
            points.append(_pt("issue", f"Possible 3 seconds in the key: {len(flags)}",
                              f"{len(flags)} time(s) a {T} player stayed in the key for more than {THREE_SECONDS:.0f} s "
                              f"(up to {max(b - a for a, b, _ in flags) / fps:.1f} s) while {T} attacked in the half-court.",
                              "An attacker may not stay in the key for more than 3 s. Beyond the rule, a player parked in "
                              "the paint brings their defender with them and clogs the drives.",
                              [_m(frames, fps, a + int(THREE_SECONDS * fps), idx(a, i), f"{(b - a) / fps:.1f} s in the key")
                               for a, b, i in flags[:4]]))
        # 8 seconds and the shot clock
        slow, late = [], []
        bs = _base(c, team)
        for a, b, _ in mine:
            start = c["pl"][a]
            if start and bs:
                xs = [bs(x, y)[0] for _, t, x, y, _ in start if t == team]
                if xs and median(xs) > c["L"] / 2:   # started in their own backcourt
                    over = next((g for g in range(a, b + 1) if c["pl"][g] and median(
                        [bs(x, y)[0] for _, t, x, y, _ in c["pl"][g] if t == team] or [99]) < c["L"] / 2), None)
                    if over is None or over - a > EIGHT_SECONDS * fps:
                        if (over or b) - a > EIGHT_SECONDS * fps:
                            slow.append(a)
            if (b - a) / fps >= LATE_CLOCK:
                late.append(a)
        if slow:
            points.append(_pt("issue", f"Slow over halfway: {len(slow)} possession(s)",
                              f"{len(slow)} time(s) {T} took more than {EIGHT_SECONDS:.0f} s to get over halfway.",
                              "The ball must be in the frontcourt within 8 s. Against pressure, get the ball to a guard "
                              "early and give them a target to pass to.",
                              [_m(frames, fps, a, [], "Over 8 s in the backcourt") for a in slow[:4]]))
        if len(mine) >= 3:
            points.append(_pt("info", f"Typical possession: {stats['possession_length']:.0f} s",
                              f"{T} had {len(mine)} possessions in the measured footage; {len(late)} went past {LATE_CLOCK:.0f} s "
                              f"of the {SHOT_CLOCK:.0f} s shot clock.",
                              "Possessions that run deep into the clock end in hurried shots. Check whether the first action "
                              "came too late.", [_m(frames, fps, a, [], "Long possession") for a in late[:4]]))
    if phase in ("defence", "mixed") and def_f:
        gs = [(f, _goal_side(c, f, team, opp)) for f in def_f]
        gs = [(f, v) for f, v in gs if v is not None]
        if len(gs) >= 2 * fps:
            mg = median(v for _, v in gs)
            stats["goal_side"] = round(100 * mg)
            flags = {f: (1 - v, _hl(frames, f, [i for _, _, _, _, i in c["pl"][f]])) for f, v in gs if v < 0.5}
            if mg < GOAL_SIDE_BAD:
                points.append(_pt("issue", "Not between the player and the basket",
                                  f"Typically only {mg:.0%} of {O}'s players in view had a {T} defender nearer the basket than them.",
                                  "Defenders stay between their player and the basket (goal-side, 'ball-you-man'). Losing that "
                                  "gives up straight drives and back-door cuts.",
                                  analysis._moments(analysis._events(flags, fps), lambda e: "Attackers goal-side")))
            elif mg >= GOAL_SIDE_GOOD:
                points.append(_pt("good", "Goal-side defence", f"Typically {mg:.0%} of {O}'s players had a {T} defender "
                                  "between them and the basket.", "That's what keeps drives in front of you."))
        help_f, none = 0, {}
        bs = _base(c, opp)
        for f in def_f:
            b = c["ball"][f]
            if b is None or not bs:
                continue
            bx, by = bs(*b)
            if math.hypot(bx - pitch.BASKET, by) < BALL_AWAY:
                continue
            inside = [i for _, t, x, y, i in c["pl"][f] if t == team and _in_key(*bs(x, y), HELP_R)]
            help_f += 1
            if not inside:
                none[f] = (1, _hl(frames, f, [i for _, t, _, _, i in c["pl"][f] if t == team]))
        if help_f >= 2 * fps:
            share = 1 - len(none) / help_f
            points.append(_pt("good" if share >= 0.7 else "issue" if share < 0.4 else "info",
                              f"Help in the paint: {share:.0%} of the time",
                              f"With the ball outside ({BALL_AWAY:.0f} m or more from the basket), {T} had a defender in or "
                              f"next to the key {share:.0%} of the time.",
                              "The help defender in the paint stops the drive and the roll to the basket. Off the ball, sink "
                              "towards the paint and see both your player and the ball.",
                              analysis._moments(analysis._events(none, fps), lambda e: "Nobody in the paint")))
        onb = []
        for f in def_f:
            o, b = c["owner"][f], c["ball"][f]
            if not o or o[1] != opp or b is None:
                continue
            d = min((math.hypot(x - b[0], y - b[1]) for _, t, x, y, _ in c["pl"][f] if t == team), default=None)
            if d is not None:
                onb.append(d <= ON_BALL)
        if len(onb) >= 2 * fps:
            share = sum(onb) / len(onb)
            points.append(_pt("good" if share >= 0.6 else "issue" if share < 0.3 else "info",
                              f"Pressure on the ball: {share:.0%}",
                              f"{O}'s player with the ball had a {T} defender within {ON_BALL:.0f} m {share:.0%} of the time.",
                              "Pressure on the ball makes passes harder to see and slows the offence down."))
        shapes = [_zone_shape(c, f, team, opp) for f in def_f]
        follow = [v for v in (_follow(c, f, team, opp) for f in def_f) if v is not None]
        if len(follow) >= 2 * fps:
            mf = median(follow)
            shp = [s for s in shapes if s]
            style = "man-to-man" if mf >= MAN_SHARE else "a zone" if mf <= ZONE_SHARE else "a mix of man and zone"
            best = max(set(shp), key=shp.count) if shp else None
            if style == "a zone" and best:
                style = f"a {best} zone"
            stats["defence"] = style
            points.append(_pt("info", f"Defended in {style}", f"Typically {mf:.0%} of {T}'s defenders stayed with the same "
                              "attacker from one second to the next.", "Check it's the defence you called."))
    sc = [s for s in extra.get("screens", [])]
    if phase in ("defence", "mixed"):
        against = [s for s in sc if s["team"] == opp]
        if len(against) >= MIN_EVENTS:
            cov = {k: sum(1 for s in against if s["coverage"] == k) for k in ("switch", "hedge", "drop", "over")}
            beat = [s for s in against if s["worked"]]
            names = {"switch": "switched", "hedge": "hedged or trapped", "drop": "dropped", "over": "went over the screen"}
            points.append(_pt("issue" if len(beat) / len(against) >= 0.5 else "info",
                              f"Ball screens defended: {len(against) - len(beat)} of {len(against)} kept out of the key",
                              "Coverage: " + ", ".join(f"{names[k]} {v}" for k, v in cov.items() if v) + ".",
                              "Check each coverage is the one you called, and the help on the roller.",
                              [{**s, "time": analysis._fmt_t(s["t"]), "frame": s["f"], "label": f"{names[s['coverage']]}, got to the key"} for s in beat[:4]]))
    if phase in ("attack", "mixed"):
        ours = [s for s in sc if s["team"] == team]
        if len(ours) >= MIN_EVENTS:
            ok = sum(1 for s in ours if s["worked"])
            points.append(_pt("good" if ok / len(ours) >= 0.5 else "info", f"Ball screens: {len(ours)}, {ok} got to the key",
                              f"{T} set {len(ours)} ball screens; the ball or the handler got into the key within "
                              f"{SCREEN_WORKED:.0f} s after {ok} of them.",
                              "The pick-and-roll is about the two-on-one it creates. Watch the ones that didn't get in: "
                              "flat screen angle, or the roller too slow?",
                              [{**s, "time": analysis._fmt_t(s["t"]), "frame": s["f"], "label": "Didn't get to the key"}
                               for s in ours if not s["worked"]][:4]))
    fts = extra.get("free_throws", [])
    if fts:
        mine_ft = [f for f in fts if f["team"] == team]
        if mine_ft:
            points.append(_pt("info", f"Free throws: {len(mine_ft)}", f"{T} went to the line {len(mine_ft)} time(s) in the footage.",
                              "Found from the players lining up along the key.",
                              [{**f, "time": analysis._fmt_t(f["t"]), "frame": f["f"], "label": "Free throw"} for f in mine_ft[:4]]))
    order = {"issue": 0, "info": 1, "good": 2}
    points.sort(key=lambda p: order[p["kind"]])
    for p in points:
        for m in p["moments"]:
            m["time"] = analysis._fmt_t(m["t"])
    return base


def summary(c, extra):
    if not c["mapped"]:
        return None
    fps = c["fps"]
    had = {t: sum(1 for f in c["mapped"] if c["off"][f] == t) for t in c["teams"]}
    known = sum(had.values())
    return {
        "measured": round(len(c["mapped"]) / fps, 1),
        "ball_seen": round(100 * c["ball_share"]),
        "possession": {t: round(100 * n / known) for t, n in had.items()} if known else None,
        "possessions": {t: sum(1 for s in c["spells"] if s[2] == t) for t in c["teams"]},
        "screens": len(extra.get("screens", [])),
        "free_throws": len(extra.get("free_throws", [])),
        "attacks": {t: (None if not c["basket"][t] else "right" if c["basket"][t][0] > c["L"] / 2 else "left") for t in c["teams"]},
    }


# ================================================================== tactics

CATALOGUE = [
    {"id": "man", "side": "defence", "group": "defence", "name": "Man-to-man", "needs_metres": True,
     "about": "Every defender guards one attacker.",
     "played": "60% or more of the defenders stay with the same attacker from one second to the next",
     "worked": "the defender is between their player and the basket"},
    {"id": "zone_23", "side": "defence", "group": "defence", "name": "2-3 zone", "needs_metres": True,
     "about": "Two up top, three along the baseline, guarding areas instead of players.",
     "played": "a zone (30% or fewer follow one player) standing in two lines, two at the front and three at the back",
     "worked": "a defender in or next to the key when the ball is outside"},
    {"id": "zone_32", "side": "defence", "group": "defence", "name": "3-2 zone", "needs_metres": True,
     "about": "Three up top, two low.",
     "played": "a zone standing three at the front and two at the back",
     "worked": "a defender in or next to the key when the ball is outside"},
    {"id": "zone_131", "side": "defence", "group": "defence", "name": "1-3-1 zone", "needs_metres": True,
     "about": "One at the top, three across the middle, one at the back.",
     "played": "a zone standing in three lines: one, three, one",
     "worked": "a defender in or next to the key when the ball is outside"},
    {"id": "press", "side": "defence", "group": "pressure", "name": "Full-court press", "needs_metres": True,
     "about": "Pressure the ball in the backcourt.",
     "played": "their player with the ball in their own backcourt has a defender within 2 m (needs the ball)",
     "worked": "they take more than 8 s to get over halfway, or lose the ball before it"},
    {"id": "pack_paint", "side": "defence", "group": "pressure", "name": "Pack the paint", "needs_metres": True,
     "about": "Help defenders sink into the key and make them shoot from outside.",
     "played": "two or more defenders in or next to the key with the ball outside (needs the ball)",
     "worked": "the ball doesn't get into the key"},
    {"id": "switch_all", "side": "defence", "group": "screens", "name": "Switch screens", "needs_metres": True,
     "about": "Defenders swap players on every ball screen.",
     "played": "the screener's defender takes the ball handler", "worked": "the ball doesn't get into the key within 3 s"},
    {"id": "hedge", "side": "defence", "group": "screens", "name": "Hedge / trap screens", "needs_metres": True,
     "about": "The screener's defender jumps out at the ball handler.",
     "played": "two defenders on the ball handler after the screen", "worked": "the ball doesn't get into the key within 3 s"},
    {"id": "drop", "side": "defence", "group": "screens", "name": "Drop coverage", "needs_metres": True,
     "about": "The big sits back near the basket and the guard fights over the screen.",
     "played": "the screener's defender stays 1.5 m or more nearer the basket than the handler, 3 m or more away",
     "worked": "the ball doesn't get into the key within 3 s"},
    {"id": "get_back", "side": "defence", "group": "pressure", "name": "Get back in transition", "needs_metres": True,
     "about": "Sprint back after a lost ball or a shot so they never have numbers.",
     "played": "when they get the ball, at least as many of yours as theirs between the ball and your basket within 3 s",
     "worked": None},
    {"id": "fast_break", "side": "attack", "group": "pace", "name": "Push the pace / fast break", "needs_metres": True,
     "about": "Run after every defensive rebound or steal, before the defence is set.",
     "played": "the team is set in the frontcourt within 6 s of getting the ball", "worked": "the ball gets into the key"},
    {"id": "slow_pace", "side": "attack", "group": "pace", "name": "Use the clock", "needs_metres": True,
     "about": "Long possessions to control the game.",
     "played": "possessions of 15 s or more", "worked": "no possession goes past 24 s"},
    {"id": "pick_roll", "side": "attack", "group": "sets", "name": "Pick-and-roll", "needs_metres": True,
     "about": "A ball screen for the handler, the screener rolls to the basket.",
     "played": "a ball screen in the possession (needs the ball)", "worked": "the ball or the handler gets into the key within 3 s"},
    {"id": "five_out", "side": "attack", "group": "sets", "name": "5-out", "needs_metres": True,
     "about": "All five outside the key to open driving lanes.",
     "played": "all attackers in view 5 m or more from the basket", "worked": "spacing of 4.5 m or more"},
    {"id": "four_out", "side": "attack", "group": "sets", "name": "4-out 1-in", "needs_metres": True,
     "about": "Four on the perimeter, one in the post.",
     "played": "exactly one attacker within 4 m of the basket", "worked": "spacing of 4.5 m or more"},
    {"id": "corners", "side": "attack", "group": "sets", "name": "Fill the corners", "needs_metres": True,
     "about": "Keep shooters in both corners.",
     "played": "a player in each corner", "worked": None},
    {"id": "post_up", "side": "attack", "group": "sets", "name": "Post-ups", "needs_metres": True,
     "about": "Get the ball to a player with their back to the basket on the block.",
     "played": "the ball with a player 1-4 m from the basket beside the key (needs the ball)", "worked": None},
]
BY_ID = {t["id"]: t for t in CATALOGUE}
NEEDS_BALL = {"press", "pack_paint", "pick_roll", "post_up", "switch_all", "hedge", "drop"}


def catalogue():
    return CATALOGUE


def _frame_check(tid, c, frames, f, team, opp):
    W = c["W"]
    if tid in ("man", "zone_23", "zone_32", "zone_131"):
        fl = _follow(c, f, team, opp)
        if fl is None:
            return None, None
        if tid == "man":
            if fl < MAN_SHARE:
                return False, None
            g = _goal_side(c, f, team, opp)
            return True, None if g is None else g >= GOAL_SIDE_GOOD
        shape = _zone_shape(c, f, team, opp)
        if shape is None:
            return None, None
        played = fl <= ZONE_SHARE and shape == tid[5:].replace("", "-").strip("-")
        if not played:
            return False, None
        bs, b = _base(c, opp), c["ball"][f]
        if b is None or math.hypot(bs(*b)[0] - pitch.BASKET, bs(*b)[1]) < BALL_AWAY:
            return True, None
        return True, any(_in_key(*bs(x, y), HELP_R) for _, t, x, y, _ in c["pl"][f] if t == team)
    if tid == "pack_paint":
        bs, b = _base(c, opp), c["ball"][f]
        if b is None or not bs or math.hypot(bs(*b)[0] - pitch.BASKET, bs(*b)[1]) < BALL_AWAY:
            return None, None
        n = sum(1 for _, t, x, y, _ in c["pl"][f] if t == team and _in_key(*bs(x, y), HELP_R))
        return n >= 2, (not _in_key(*bs(*b))) if n >= 2 else None
    if tid in ("five_out", "four_out", "corners"):
        bs = _base(c, team)
        pts = [bs(x, y) for _, t, x, y, _ in c["pl"][f] if t == team]
        if len(pts) < 4:
            return None, None
        inside = sum(1 for x, y in pts if math.hypot(x - pitch.BASKET, y) < (5.0 if tid == "five_out" else 4.0))
        if tid == "corners":
            left = any(x <= CORNER_X and y <= -W / 2 + CORNER_Y for x, y in pts)
            right = any(x <= CORNER_X and y >= W / 2 - CORNER_Y for x, y in pts)
            return left and right, None
        played = inside == 0 if tid == "five_out" else inside == 1
        sp = _spacing(c, f, team)
        return played, (None if not played or sp is None else sp >= SPACING_GOOD)
    if tid == "post_up":
        o, b = c["owner"][f], c["ball"][f]
        if not o or o[1] != team or b is None:
            return None, None
        x, y = _base(c, team)(*b)
        return 1.0 <= math.hypot(x - pitch.BASKET, y) <= 4.0 and abs(y) >= pitch.KEY_HALF - 0.5, None
    return None, None


def _events(tid, c, frames, team, opp, extra):
    fps = c["fps"]
    out = []
    if tid in ("switch_all", "hedge", "drop"):
        want = {"switch_all": "switch", "hedge": "hedge", "drop": "drop"}[tid]
        for s in extra.get("screens", []):
            if s["team"] == opp:
                played = s["coverage"] == want
                out.append((s["f"], played, (not s["worked"]) if played else None))
        return out
    if tid == "pick_roll":
        sc = [s for s in extra.get("screens", []) if s["team"] == team]
        for a, b, t in c["spells"]:
            if t != team or b - a < 3 * fps:
                continue
            mine = [s for s in sc if a <= s["f"] <= b]
            out.append((a, bool(mine), any(s["worked"] for s in mine) if mine else None))
        return out
    for a, b, t in c["spells"]:
        if tid in ("fast_break", "slow_pace") and t == team:
            if tid == "slow_pace":
                out.append((a, (b - a) / fps >= 15.0, (b - a) / fps <= SHOT_CLOCK if (b - a) / fps >= 15.0 else None))
                continue
            set_at = next((g for g in range(a, b + 1) if c["half"][g]), None)
            played = set_at is not None and (set_at - a) / fps <= FAST_BREAK and not c["half"][a]
            bs = _base(c, team)
            worked = any(c["ball"][g] is not None and _in_key(*bs(*c["ball"][g])) for g in range(a, b + 1)) if played else None
            out.append((a, played, worked))
        elif tid == "get_back" and t == opp:
            bs = _base(c, opp)
            g = min(b, a + int(3 * fps))
            if not c["pl"][g] or c["ball"][g] is None or not bs:
                continue
            bx = bs(*c["ball"][g])[0]
            mine = sum(1 for _, tm, x, y, _ in c["pl"][g] if tm == team and bs(x, y)[0] < bx)
            theirs = sum(1 for _, tm, x, y, _ in c["pl"][g] if tm == opp and bs(x, y)[0] < bx)
            out.append((a, mine >= theirs, None))
        elif tid == "press" and t == opp:
            bs = _base(c, opp)
            back = [g for g in range(a, b + 1) if c["owner"][g] and c["owner"][g][1] == opp and c["ball"][g] is not None
                    and bs(*c["ball"][g])[0] > c["L"] / 2]
            if not back:
                continue
            played = any(min((math.hypot(x - c["ball"][g][0], y - c["ball"][g][1]) for _, tm, x, y, _ in c["pl"][g] if tm == team),
                             default=99) <= ON_BALL for g in back)
            out.append((a, played, ((back[-1] - a) / fps > EIGHT_SECONDS or back[-1] == b) if played else None))
    return out


EVENTS = ("switch_all", "hedge", "drop", "pick_roll", "fast_break", "slow_pace", "get_back", "press")
UNITS = {"switch_all": "screens against you", "hedge": "screens against you", "drop": "screens against you",
         "pick_roll": "possessions of 3 s or more", "fast_break": "possessions", "slow_pace": "possessions",
         "get_back": "their possessions", "press": "their possessions starting in the backcourt"}
RARE = {"pick_roll": 0.4, "fast_break": 0.3, "slow_pace": 0.4, "post_up": 0.1, "corners": 0.4, "press": 0.5}


def _tp(t, kind, title, detail, moments=()):
    return {"kind": kind, "title": title, "detail": detail, "moments": list(moments), "tactic": t["id"],
            "why": f"Your plan: {t['about']}" + (" Checked from the video; confirm each moment before acting on it." if kind != "info" else "")}


def evaluate(chosen, c, frames, team, opp, phase, extra):
    fps = c["fps"]
    T = team.capitalize()
    out = []
    for tid in chosen:
        t = BY_ID.get(tid)
        if not t:
            continue
        if not c["mapped"]:
            out.append(_tp(t, "info", f"{t['name']}: mark the court to check it",
                           "This tactic is measured in metres. Mark the court in the Pitch tab and make the report again."))
            continue
        if tid in NEEDS_BALL and c["ball_share"] < 0.1:
            out.append(_tp(t, "info", f"{t['name']}: the ball wasn't seen enough",
                           f"This check needs the ball, which was seen with a player in only {c['ball_share']:.0%} of the footage."))
            continue
        target = RARE.get(tid, PLAYED_ENOUGH)
        if tid in EVENTS:
            ch = _events(tid, c, frames, team, opp, extra)
            if len(ch) < MIN_EVENTS:
                out.append(_tp(t, "info", f"{t['name']}: not enough to judge",
                               f"Only {len(ch)} {UNITS[tid]} could be checked (at least {MIN_EVENTS} needed)."))
                continue
            played = [x for x in ch if x[1]]
            tried = [x for x in played if x[2] is not None]
            worked = [x for x in tried if x[2]]
            share = len(played) / len(ch)
            works = len(worked) / len(tried) if tried else None
            detail = f"Played in {len(played)} of {len(ch)} {UNITS[tid]} ({share:.0%}); played means {t['played']}."
            if works is not None and t["worked"]:
                detail += f" It worked {len(worked)} of {len(tried)} times ({works:.0%}): {t['worked']}."
            misses = [_m(frames, fps, f, [], f"{t['name']} played, didn't work") for f, p, w in ch if p and w is False] + \
                     [_m(frames, fps, f, [], f"{t['name']} not played") for f, p, _ in ch if not p]
        else:
            fs = offence_frames(c, team) if t["side"] == "attack" else defence_frames(c, team)
            res = [(f,) + _frame_check(tid, c, frames, f, team, opp) for f in fs]
            res = [r for r in res if r[1] is not None]
            if len(res) < 2 * fps:
                out.append(_tp(t, "info", f"{t['name']}: not enough to judge",
                               f"{T} was seen {'attacking' if t['side'] == 'attack' else 'defending'} in the half-court with "
                               f"enough players in view for only {len(res) / fps:.0f}s (at least 2 s needed)."))
                continue
            played = [r for r in res if r[1]]
            tried = [r for r in played if r[2] is not None]
            worked = [r for r in tried if r[2]]
            share = len(played) / len(res)
            works = len(worked) / len(tried) if len(tried) >= fps else None
            detail = f"Played {share:.0%} of the {len(res) / fps:.0f}s it could be checked; played means {t['played']}."
            if works is not None and t["worked"]:
                detail += f" It worked {works:.0%} of the time it was played: {t['worked']}."
            bad = {r[0]: (1, "") for r in tried if r[2] is False}
            no = {r[0]: (1, "") for r in res if not r[1]}
            misses = [_m(frames, fps, e["frame"], [], f"{t['name']} played, didn't work") for e in analysis._events(bad, fps)] + \
                     [_m(frames, fps, e["frame"], [], f"{t['name']} not played") for e in analysis._events(no, fps)]
        kind = "good" if share >= target and (works is None or works >= 0.5) else "issue"
        if share < target:
            title = f"{t['name']}: only played {share:.0%} of the time"
        elif works is not None and works < 0.5:
            title = f"{t['name']}: played, but worked only {works:.0%} of the time"
        else:
            title = f"{t['name']}: stuck to it {share:.0%} of the time" + (f", worked {works:.0%}" if works is not None else "")
        out.append(_tp(t, kind, title, detail, misses[:4]))
    return out
