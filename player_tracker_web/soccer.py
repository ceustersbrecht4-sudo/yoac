"""
Football (soccer) layer of the coach report.

Tracking, teams and the pitch marking are shared with rugby; this module
turns them into football: the goalkeepers, which end each team defends,
who has the ball, the team's shape in metres (how high the back line
holds, length, width, the space between the lines, the shape it plays in)
and the tactics a coach can pick.

Numbers come from coaching and analysis material where it gives them:
  - line height: a low block holds the back line about 22-28 m from its
    own goal, a high line / press about 45-55 m (FIFA EFI definitions)
  - a compact block is 25-35 m from back to front and 30-35 m wide
  - 8-12 m between the lines is compact, over 15 m leaves space between them
  - "pressure": a defender within about 5 m of the ball (StatsBomb, 5 yd)
  - counter-press: win it back within about 6 s of losing it
Others are rules of thumb, named below so they can be tuned on real footage.
"""

import math
from statistics import median

import cv2
import numpy as np

import analysis
import pitch

TEAM_SKIP = ("other", "unsure", "unknown", "ball")

# ---- goalkeepers (found in the picture, before the pitch is marked)
KEEPER_BEYOND = 1        # team players allowed nearer the goal than the keeper (a striker at a corner)
MIN_KEEPER_PLAYERS = 4   # team players in view before someone can be the last one at an end
MIN_KEEPER_FRAMES = 10   # sightings before a track can be a keeper
KEEPER_END_SHARE = 0.6   # share of sightings the keeper must be the last one at an end

# ---- shape
MIN_OUTFIELD = 6         # outfield players in view before length / line height count
MIN_LINES = 8            # ... before the lines (and the space between them) count
FULL_TEAM = 10           # all outfield players in view: the shape (4-4-2...) can be read
BACK_LINE = 4            # the deepest players that make the back line
LINE_SPLIT = 5.0         # m along the pitch between two players = different lines
KEEPER_GAP = 10.0        # m behind every teammate, inside the box = the keeper (when not found by colour)
VIEW_MARGIN = 5.0        # m past the last player that must be on screen (else a teammate may be off it)
LOW_BLOCK = 28.0         # m from own goal: back line at or below this = low block
HIGH_LINE = 45.0         # m: at or above this = high line / high press
HIGH_LINE_TRAP = 35.0    # m: a line held this high is playing offside, not dropping off
STRETCHED = 40.0         # m back to front without the ball = stretched
COMPACT = 35.0           # m back to front = compact
WIDE_OUT = 45.0          # m across without the ball = too wide
NARROW_OUT = 38.0        # m across without the ball = compact across
LINE_GAP_BAD = 15.0      # m between the lines = space to play in
LINE_GAP_GOOD = 12.0
BACK_LEVEL = 5.0         # m between the deepest and highest of the back line = not level
PRESS_REACH = 5.0        # m from the ball = pressing it
LEFT_ALONE = 8.0         # m: nobody nearer than this = the ball carrier is left alone
NARROW_IN = 40.0         # m across with the ball = narrow
WIDE_IN = 50.0           # m across with the ball = stretching the pitch
WING = 8.0               # m from a touchline = holding the width
REST_BEHIND = 10.0       # m behind the ball = in the rest defence
REST_MIN = 4             # players behind the ball when attacking in the other half
POSSESS_REACH = 2.0      # m from the ball = on it (one camera isn't exact to 1 m)
POSSESS_FRAMES = 5       # frames on the ball in a row before possession changes
POSSESS_HOLD = 3.0       # s without anyone on the ball before possession is unknown
USE_POSSESSION = 0.3     # share of the measured footage with possession known to split by it
NEAR_GOAL = 30.0         # m from a goal line: the last player there tells which end a team defends

# tactics
LONG_BALL = 30.0         # m forward in LONG_BALL_TIME = a long ball
LONG_BALL_TIME = 2.0
SWITCH = 30.0            # m across in SWITCH_TIME = a switch of play
SWITCH_TIME = 3.0
KEEP_AFTER = 5.0         # s after a long ball the team must still have it
PRESS_AFTER = 2.0        # s after losing the ball that the counter-press must be on it
WIN_BACK = 6.0           # s to win it back
COUNTER_TIME = 10.0      # s for a counter-attack to go 30 m / reach the box
OVERLOAD_REACH = 15.0    # m around the ball where numbers are counted
OVERLOAD_GAIN = 15.0     # m forward within OVERLOAD_TIME = the overload worked
OVERLOAD_TIME = 5.0
MIN_SPELL = 10.0         # s of possession before a switch of play is expected
MIN_EVENTS = 3
PLAYED_ENOUGH = 0.6
PLAYED_ENOUGH_RARE = {"direct": 0.3, "switch": 0.3, "overload": 0.3, "counter_attack": 0.4, "build_up": 0.5,
                      "overlap": 0.3, "possession": 0.4, "short_corners": 0.3, "short_goal_kicks": 0.5,
                      "long_goal_kicks": 0.5}


# =============================================================== goalkeepers

def find_keepers(detections):
    """Goalkeepers wear their own colour, so teams.py puts them with the
    officials. A keeper is the one "other" person who is nearly always the
    last one at an end of the play, level with the middle of it (an
    assistant referee is at the edge, the referee in the middle). The offside
    law keeps the last outfield player in front of a keeper one of their own
    defenders, so that player's team is the keeper's team.
    Returns (detections, {track id: team})."""
    counts = {}
    for frame in detections:
        for d in frame:
            if d[4] not in TEAM_SKIP:
                counts[d[4]] = counts.get(d[4], 0) + 1
    teams = sorted(counts, key=counts.get, reverse=True)[:2]
    if len(teams) < 2:
        return detections, {}
    looks = {}
    for frame in detections:
        feet = [((d[0] + d[2]) / 2, d[3], d[4]) for d in frame if d[4] in teams]
        if len(feet) < MIN_KEEPER_PLAYERS:
            continue
        for d in frame:
            if d[4] != "other" or d[5] is None:
                continue
            x, y = (d[0] + d[2]) / 2, d[3]
            left = [p for p in feet if p[0] < x]
            right = [p for p in feet if p[0] > x]
            level = any(p[1] < y for p in feet) and any(p[1] > y for p in feet)
            side = None
            if level and len(left) <= KEEPER_BEYOND and len(right) >= MIN_KEEPER_PLAYERS - 1:
                side = right
            elif level and len(right) <= KEEPER_BEYOND and len(left) >= MIN_KEEPER_PLAYERS - 1:
                side = left
            vote = min(side, key=lambda p: abs(p[0] - x))[2] if side else None
            looks.setdefault(d[5], []).append(vote)
    keepers = {}
    for tid, seen in looks.items():
        votes = [v for v in seen if v]
        if len(seen) >= MIN_KEEPER_FRAMES and len(votes) >= KEEPER_END_SHARE * len(seen):
            keepers[tid] = max(teams, key=votes.count)
    new = [[list(d) for d in frame] for frame in detections]
    for frame in new:
        for d in frame:
            if d[5] in keepers and d[4] == "other":
                d[4] = keepers[d[5]]
    return new, keepers


# ================================================================ context

def _ball(frame, H, L, W):
    """Where the ball is on the pitch in this frame (the surest detection on
    the pitch), or None."""
    balls = sorted(frame[4], key=lambda b: -b[2])
    if not balls:
        return None
    for x, y in pitch.project(H, [(b[0], b[1]) for b in balls]):
        if -2 <= x <= L + 2 and -2 <= y <= W + 2:
            return float(x), float(y)
    return None


def _view(H, size):
    """The part of the pitch on screen, as a polygon in metres."""
    if size is None:
        return None
    w, h = size
    corners = pitch.project(H, [(0, 0), (w, 0), (w, h), (0, h)])
    if not np.all(np.isfinite(corners)):
        return None
    return corners.astype(np.float32).reshape(-1, 1, 2)


def _seen(poly, x, y):
    return poly is None or cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def _shape(players, xy, team, flip, keepers, L, W, poly):
    """One team's shape in one frame, in metres from its own goal line."""
    own = lambda x: L - x if flip else x
    pts = [(own(float(x)), float(y), i) for i, (p, (x, y)) in enumerate(zip(players, xy)) if p["bucket"] == team]
    out = [q for q in pts if players[q[2]]["id"] not in keepers]
    keeper_seen = len(out) < len(pts)
    out.sort()
    if not keeper_seen and len(out) >= 2 and out[0][0] < pitch.BOX_DEPTH and out[1][0] - out[0][0] >= KEEPER_GAP:
        out = out[1:]  # the keeper, by where they stand
    r = {"n": len(out), "idx": [q[2] for q in out], "pos": [(q[0], q[1]) for q in out]}
    if len(out) < MIN_OUTFIELD:
        return r
    xs = [q[0] for q in out]
    ys = [q[1] for q in out]
    pitch_x = lambda x: L - x if flip else x
    # Only trust a measurement when the space just past the last player is on
    # screen; otherwise a teammate could be standing out of the picture.
    cy = median(ys)
    back_ok = _seen(poly, pitch_x(max(0.0, xs[0] - VIEW_MARGIN)), cy)
    front_ok = _seen(poly, pitch_x(min(L, xs[-1] + VIEW_MARGIN)), cy)
    cx = pitch_x(median(xs))
    wide_ok = _seen(poly, cx, max(0.0, min(ys) - VIEW_MARGIN)) and _seen(poly, cx, min(W, max(ys) + VIEW_MARGIN))
    back = xs[:BACK_LINE]
    if back_ok:
        r["line_height"] = sum(back) / len(back)
        r["back_spread"] = back[-1] - back[0]
        r["back_idx"] = r["idx"][:BACK_LINE]
    if back_ok and front_ok:
        r["length"] = xs[-1] - xs[0]
    if wide_ok:
        r["width"] = max(ys) - min(ys)
        r["wings"] = (min(ys) <= WING, max(ys) >= W - WING)
    if back_ok and front_ok and len(out) >= MIN_LINES:
        lines, cur = [], [xs[0]]
        for a, b in zip(xs, xs[1:]):
            if b - a > LINE_SPLIT:
                lines.append(cur)
                cur = []
            cur.append(b)
        lines.append(cur)
        if len(lines) >= 2:
            mids = [sum(l) / len(l) for l in lines]
            r["line_gap"] = max(b - a for a, b in zip(mids, mids[1:]))
            r["lines"] = [len(l) for l in lines]
            if len(out) == FULL_TEAM and 3 <= len(lines) <= 5:
                r["formation"] = "-".join(str(len(l)) for l in lines)
    return r


def _directions(frames, maps, teams, L):
    """{team: True if it defends the right-hand goal (x = length)}. The last
    player near a goal is almost always one of that end's defenders (or its
    keeper): the offside law sees to that."""
    low = {t: 0 for t in teams}
    high = {t: 0 for t in teams}
    for f, H in enumerate(maps):
        if H is None:
            continue
        players = [p for p in frames[f][0] if p["bucket"] in teams]
        if len(players) < 4:
            continue
        xy = pitch.project(H, [(p["x"], p["y"]) for p in players])
        i, j = int(np.argmin(xy[:, 0])), int(np.argmax(xy[:, 0]))
        if xy[i, 0] < NEAR_GOAL:
            low[players[i]["bucket"]] += 1
        if xy[j, 0] > L - NEAR_GOAL:
            high[players[j]["bucket"]] += 1
    a, b = teams
    a_low = low[a] + high[b]
    b_low = low[b] + high[a]
    return {a: a_low < b_low, b: a_low >= b_low}


def _possession(ctx, frames):
    """Which team has the ball, frame by frame: the team of whoever stays
    within 2 m of it for a few frames keeps it until the other team does, or
    until nobody's seen on it for a few seconds."""
    n, fps = ctx["n"], ctx["fps"]
    owner = [None] * n
    for f in range(n):
        b, xy = ctx["ball"][f], ctx["xy"][f]
        if b is None or xy is None:
            continue
        best, who, tid = POSSESS_REACH, None, None
        for p, (x, y) in zip(frames[f][0], xy):
            if p["bucket"] in ctx["teams"]:
                d = math.hypot(x - b[0], y - b[1])
                if d < best:
                    best, who, tid = d, p["bucket"], p["id"]
        owner[f] = who
        if who is not None:
            ctx["owner_id"][f] = (tid, who)
    poss, turnovers = [None] * n, []
    cur, cand, streak, last = None, None, 0, -10 ** 9
    for f in range(n):
        o = owner[f]
        if o is not None:
            last = f
            if o == cur:
                cand, streak = None, 0
            else:
                streak = streak + 1 if o == cand else 1
                cand = o
                if streak >= POSSESS_FRAMES:
                    start = f - POSSESS_FRAMES + 1
                    if cur is not None:
                        turnovers.append({"f": start, "lost": cur, "won": o})
                    for g in range(start, f):
                        poss[g] = o
                    cur, cand, streak = o, None, 0
        elif f - last > POSSESS_HOLD * fps:
            cur = None
        poss[f] = cur
    return poss, turnovers


def context(frames, fps, maps, calib, teams, keepers=None, size=None):
    """Everything the football report and tactics need, worked out once."""
    dims = pitch.SIZES["soccer"]
    L = float((calib or {}).get("length") or dims["length"])
    W = float((calib or {}).get("width") or dims["width"])
    n = len(frames)
    ctx = {"fps": fps, "L": L, "W": W, "n": n, "teams": list(teams), "mapped": [],
           "xy": [None] * n, "ball": [None] * n, "poss": [None] * n, "turnovers": [],
           "shape": {t: [None] * n for t in teams}, "flip": {t: False for t in teams}, "use_poss": False,
           "keepers": {int(k) for k in (keepers or {})}, "view": [None] * n, "owner_id": [None] * n,
           "ball_share": 0}
    if not maps or len(teams) < 2:
        return ctx
    ctx["flip"] = _directions(frames, maps, teams, L)
    for f, H in enumerate(maps):
        if H is None or not frames[f][0]:
            continue
        players = frames[f][0]
        xy = pitch.project(H, [(p["x"], p["y"]) for p in players])
        ctx["xy"][f] = xy
        ctx["ball"][f] = _ball(frames[f], H, L, W)
        poly = _view(H, size)
        ctx["view"][f] = poly
        for t in teams:
            ctx["shape"][t][f] = _shape(players, xy, t, ctx["flip"][t], ctx["keepers"], L, W, poly)
        ctx["mapped"].append(f)
    ctx["poss"], ctx["turnovers"] = _possession(ctx, frames)
    known = sum(1 for f in ctx["mapped"] if ctx["poss"][f])
    ctx["use_poss"] = bool(ctx["mapped"]) and known >= USE_POSSESSION * len(ctx["mapped"])
    ctx["ball_share"] = known / len(ctx["mapped"]) if ctx["mapped"] else 0
    return ctx


def _own_x(ctx, team, x):
    """Metres from `team`'s own goal line."""
    return ctx["L"] - x if ctx["flip"][team] else x


def _frames(ctx, team, opp, side, phase):
    """Measured frames where `team` is attacking (side="attack") or defending."""
    if ctx["use_poss"]:
        want = team if side == "attack" else opp
        return [f for f in ctx["mapped"] if ctx["poss"][f] == want]
    return list(ctx["mapped"]) if phase == side else []


def _hl(frames, f, idx):
    players = [frames[f][0][i] for i in idx if i < len(frames[f][0])]
    return analysis._hl_bounds(players) if players else ""


def _ball_idx(ctx, frames, f, team):
    """Index of `team`'s player nearest the ball in frame f, and the distance."""
    b, xy = ctx["ball"][f], ctx["xy"][f]
    if b is None or xy is None:
        return None, None
    best, who = None, None
    for i, (p, (x, y)) in enumerate(zip(frames[f][0], xy)):
        if p["bucket"] == team:
            d = math.hypot(x - b[0], y - b[1])
            if best is None or d < best:
                best, who = d, i
    return who, best


# ================================================================= report

def _point(kind, title, detail, why, moments=()):
    return {"kind": kind, "title": title, "detail": detail, "why": why, "moments": list(moments)}


def _block_name(h):
    return "a low block" if h <= LOW_BLOCK else "a high line" if h >= HIGH_LINE else "a mid block"


def analyse_team(ctx, frames, team, opp, phase):
    """Coach points for one team (same shape as analysis.analyse_team)."""
    fps = ctx["fps"]
    T, O = team.capitalize(), (opp or "").capitalize()
    seconds = len(frames) / fps if fps else 0
    minutes = max(seconds / 60, 0.25)
    seen = [sum(1 for p in fr[0] if p["bucket"] == team) for fr in frames if fr[0]]
    seen = [n for n in seen if n]
    stats = {"players_in_view": round(sum(seen) / len(seen), 1) if seen else 0}
    points, notes = [], []
    base = {"team": team, "opponent": opp, "phase": phase, "stats": stats, "points": points, "notes": notes}
    if not ctx["mapped"]:
        notes.append("Mark the pitch in the Pitch tab to get the football report: it measures in metres how high "
                     f"{T}'s back line holds, how long and wide the team is, the space between the lines and the "
                     "shape it plays in. Without the pitch marked only the players in view can be counted.")
        return base
    wide = sum(1 for fr in frames if fr[0])
    if wide:
        notes.append(f"{len(ctx['mapped']) / wide:.0%} of the wide footage could be measured on the pitch.")
    if ctx["use_poss"]:
        notes.append(f"Which team had the ball comes from the video (the ball was seen with a player in "
                     f"{ctx['ball_share']:.0%} of the measured footage): with the ball = attacking, without = defending.")
    elif phase == "mixed":
        notes.append("The ball was seen too rarely to tell which team had it, so the report can't split attacking "
                     "from defending. Pick Attacking or Defending above to get the team-shape points.")
    else:
        notes.append(f"The ball was seen too rarely to tell which team had it, so all of the measured footage is "
                     f"treated as {T} {'attacking' if phase == 'attack' else 'defending'}, as you picked.")
    shape = ctx["shape"][team]
    out_f = _frames(ctx, team, opp, "defence", phase) if phase in ("defence", "mixed") else []
    in_f = _frames(ctx, team, opp, "attack", phase) if phase in ("attack", "mixed") else []
    allf = sorted(set(out_f) | set(in_f)) or list(ctx["mapped"])
    vals = lambda fs, k: [shape[f][k] for f in fs if shape[f] and k in shape[f]]

    for key, name in (("line_height", "line_height"), ("length", "team_length"), ("width", "team_width"),
                      ("line_gap", "line_gap")):
        v = vals(out_f or allf, key)
        stats[name] = round(median(v), 1) if len(v) >= fps else None
    forms = [shape[f]["formation"] for f in allf if shape[f] and "formation" in shape[f]]
    if len(forms) >= 2 * fps:
        best = max(set(forms), key=forms.count)
        if forms.count(best) >= 0.3 * len(forms):
            stats["formation"] = best
            points.append(_point("info", f"Played in a {best}",
                                 f"With all ten of {T}'s outfield players in view ({len(forms) / fps:.0f}s), they stood in "
                                 f"{best} lines from back to front {forms.count(best) / len(forms):.0%} of the time.",
                                 "Read from where the players stood, so a midfielder pushing up can make it look like "
                                 "a different shape for a moment. Check it matches your plan."))
    if len(vals(allf, "line_height")) < 2 * fps:
        notes.append(f"{T}'s back line and at least {MIN_OUTFIELD} outfield players were rarely all on screen, "
                     "so the team shape couldn't be measured.")

    # ---------------------------------------------------- without the ball
    if out_f:
        h = vals(out_f, "line_height")
        if len(h) >= 2 * fps:
            mh = median(h)
            points.append(_point("info", f"Defended in {_block_name(mh)}",
                                 f"Without the ball {T}'s back line held about {mh:.0f} m from their own goal "
                                 f"(middle half of the time between {np.percentile(h, 25):.0f} and {np.percentile(h, 75):.0f} m).",
                                 "About 22-28 m is a low block, 35-45 m a mid block and 45 m or more a high line. "
                                 "Check it's the height you asked for."))
        ln = [f for f in out_f if shape[f] and "length" in shape[f]]
        if len(ln) >= 2 * fps:
            stretched = {f: (shape[f]["length"], _hl(frames, f, shape[f]["idx"])) for f in ln
                         if shape[f]["length"] > STRETCHED}
            share = len(stretched) / len(ln)
            ml = median(shape[f]["length"] for f in ln)
            ev = analysis._events(stretched, fps)
            if share >= 0.25:
                points.append(_point("issue", "Stretched out of possession",
                                     f"{T} was more than {STRETCHED:.0f} m from back to front {share:.0%} of the time without "
                                     f"the ball (usually {ml:.0f} m).",
                                     "A compact block is 25-35 m long. When the front players press and the back line "
                                     "stays deep, the space in between is where the opposition plays through.",
                                     analysis._moments(ev, lambda e: f"{e['severity']:.0f} m back to front")))
            elif ml <= COMPACT:
                points.append(_point("good", "Compact from back to front",
                                     f"{T} was usually {ml:.0f} m from back line to front line without the ball.",
                                     "25-35 m is what compact blocks keep: little room between the lines."))
        wd = [f for f in out_f if shape[f] and "width" in shape[f]]
        if len(wd) >= 2 * fps:
            mw = median(shape[f]["width"] for f in wd)
            flags = {f: (shape[f]["width"], _hl(frames, f, shape[f]["idx"])) for f in wd if shape[f]["width"] > WIDE_OUT}
            if mw > WIDE_OUT:
                points.append(_point("issue", "Too wide without the ball",
                                     f"{T} usually covered {mw:.0f} m across when defending.",
                                     "Compact teams defend 30-35 m wide and slide across with the ball, giving up the far "
                                     "side. Spread wider and the gaps between players open up.",
                                     analysis._moments(analysis._events(flags, fps), lambda e: f"{e['severity']:.0f} m wide")))
            elif mw <= NARROW_OUT:
                points.append(_point("good", "Compact across the pitch",
                                     f"{T} usually covered {mw:.0f} m across when defending.",
                                     "30-35 m wide and shifting with the ball is how compact blocks defend."))
        gp = [f for f in out_f if shape[f] and "line_gap" in shape[f]]
        if len(gp) >= 2 * fps:
            flags = {f: (shape[f]["line_gap"], _hl(frames, f, shape[f]["idx"])) for f in gp
                     if shape[f]["line_gap"] > LINE_GAP_BAD}
            share = len(flags) / len(gp)
            mg = median(shape[f]["line_gap"] for f in gp)
            if share >= 0.25:
                points.append(_point("issue", "Space between the lines",
                                     f"{share:.0%} of the time the gap between two of {T}'s lines was over {LINE_GAP_BAD:.0f} m "
                                     f"(usually {mg:.0f} m).",
                                     "8-12 m between the lines leaves no room to receive. Over 15 m, a player can turn "
                                     "between them. Check whether the back line dropped or the midfield went to press alone.",
                                     analysis._moments(analysis._events(flags, fps), lambda e: f"{e['severity']:.0f} m between lines")))
            elif mg <= LINE_GAP_GOOD:
                points.append(_point("good", "Lines stayed close",
                                     f"Usually {mg:.0f} m between {T}'s lines without the ball.",
                                     "8-12 m between the lines is what compact teams keep."))
        bk = [f for f in out_f if shape[f] and "back_spread" in shape[f]]
        if len(bk) >= 2 * fps:
            flags = {f: (shape[f]["back_spread"], _hl(frames, f, shape[f]["back_idx"])) for f in bk
                     if shape[f]["back_spread"] > BACK_LEVEL}
            ev = analysis._events(flags, fps)
            if ev and len(ev) / minutes >= 0.5:
                points.append(_point("issue", "Back line not level",
                                     f"{len(ev)} moment(s) where one of {T}'s back four stood more than {BACK_LEVEL:.0f} m "
                                     "deeper or higher than the others.",
                                     "A defender behind the rest plays the forwards onside; one who steps up alone leaves "
                                     "space behind. The back line steps up and drops together.",
                                     analysis._moments(ev, lambda e: f"{e['severity']:.0f} m between the back four")))
            elif not ev:
                points.append(_point("good", "Back line stayed level",
                                     f"{T}'s back four held a level line without the ball.",
                                     "A level line is what makes an offside trap and stepping up work."))
        # pressure on the ball
        pr = []
        flags = {}
        for f in out_f:
            if ctx["poss"][f] != opp:
                continue
            i, d = _ball_idx(ctx, frames, f, team)
            if d is None:
                continue
            pr.append(d <= PRESS_REACH)
            if d > LEFT_ALONE:
                flags[f] = (d, _hl(frames, f, [i]))
        if len(pr) >= 2 * fps:
            share = sum(pr) / len(pr)
            if share >= 0.5:
                points.append(_point("good", "Pressure on the ball",
                                     f"{O}'s player on the ball had a {T} player within {PRESS_REACH:.0f} m {share:.0%} of the time.",
                                     "Pressure (a defender within about 5 m) stops the passer looking up and picking a pass."))
            elif share < 0.25:
                points.append(_point("issue", "Ball carrier left alone",
                                     f"{O}'s player on the ball had a {T} player within {PRESS_REACH:.0f} m only {share:.0%} of the time.",
                                     "Without pressure the passer has time to pick any pass. Check who should step out.",
                                     analysis._moments(analysis._events(flags, fps), lambda e: f"Nearest {T} player {e['severity']:.0f} m away")))

    # ------------------------------------------------------- with the ball
    if in_f:
        wd = [f for f in in_f if shape[f] and "width" in shape[f]]
        if len(wd) >= 2 * fps:
            mw = median(shape[f]["width"] for f in wd)
            flags = {f: (NARROW_IN - shape[f]["width"] + 1, _hl(frames, f, shape[f]["idx"])) for f in wd
                     if shape[f]["width"] < NARROW_IN}
            both = sum(1 for f in wd if all(shape[f]["wings"])) / len(wd)
            if mw < NARROW_IN:
                points.append(_point("issue", "Narrow in possession",
                                     f"With the ball {T} usually covered only {mw:.0f} m across; both wings were held "
                                     f"{both:.0%} of the time.",
                                     "Width makes the defence spread, and spread defences have gaps. Keep a player "
                                     f"within {WING:.0f} m of each touchline.",
                                     analysis._moments(analysis._events(flags, fps), lambda e: "Narrow shape")))
            elif mw >= WIDE_IN:
                points.append(_point("good", "Stretched the pitch",
                                     f"With the ball {T} usually covered {mw:.0f} m across; both wings were held {both:.0%} of the time.",
                                     "Width pulls the defence apart and opens the middle."))
        rest, flags = [], {}
        for f in in_f:
            b = ctx["ball"][f]
            if b is None or not shape[f] or ctx["poss"][f] != team:
                continue
            bx = _own_x(ctx, team, b[0])
            if bx < ctx["L"] / 2:
                continue
            behind = [i for i, (x, _) in zip(shape[f]["idx"], shape[f]["pos"]) if x < bx - REST_BEHIND]
            rest.append(len(behind))
            if len(behind) < REST_MIN - 1:
                flags[f] = (REST_MIN - len(behind), _hl(frames, f, behind or shape[f]["idx"]))
        if len(rest) >= 2 * fps:
            mr = median(rest)
            if mr < REST_MIN - 1:
                points.append(_point("issue", "Few players behind the ball",
                                     f"When {T} attacked in {O}'s half, usually only {mr:.0f} outfield player(s) stayed "
                                     f"{REST_BEHIND:.0f} m or more behind the ball.",
                                     "Teams keep about 4-5 behind the ball (the rest defence) so a lost ball doesn't "
                                     "become a counter-attack.",
                                     analysis._moments(analysis._events(flags, fps), lambda e: "Thin rest defence")))
            elif mr >= REST_MIN:
                points.append(_point("good", "Rest defence in place",
                                     f"When {T} attacked in {O}'s half, usually {mr:.0f} outfield players stayed behind the ball.",
                                     "Four or five behind the ball stops counter-attacks before they start."))
        gp = [f for f in in_f if shape[f] and "line_gap" in shape[f]]
        if len(gp) >= 2 * fps:
            split = {f: (shape[f]["line_gap"], _hl(frames, f, shape[f]["idx"])) for f in gp if shape[f]["line_gap"] > 20}
            if len(split) >= 0.3 * len(gp):
                points.append(_point("info", "Team split in two with the ball",
                                     f"{len(split) / len(gp):.0%} of the time there were more than 20 m between two of {T}'s lines.",
                                     "Long gaps with the ball mean passes have to go long. Check whether a midfielder "
                                     "should drop in to link the play.",
                                     analysis._moments(analysis._events(split, fps), lambda e: f"{e['severity']:.0f} m between lines")))

    order = {"issue": 0, "info": 1, "good": 2}
    points.sort(key=lambda p: order[p["kind"]])
    for p in points:
        for m in p["moments"]:
            m["time"] = analysis._fmt_t(m["t"])
    return base


def summary(ctx):
    """Both teams together: how much of the video could be measured and who
    had the ball."""
    if not ctx["mapped"]:
        return None
    fps = ctx["fps"]
    had = {t: sum(1 for f in ctx["mapped"] if ctx["poss"][f] == t) for t in ctx["teams"]}
    known = sum(had.values())
    return {
        "measured": round(len(ctx["mapped"]) / fps, 1),
        "ball_seen": round(100 * ctx["ball_share"]),
        "possession": {t: round(100 * n / known) for t, n in had.items()} if ctx["use_poss"] and known else None,
        "turnovers": len(ctx["turnovers"]) if ctx["use_poss"] else None,
        "keepers": len(ctx["keepers"]),
        "defends": {t: ("right" if ctx["flip"][t] else "left") for t in ctx["teams"]},
    }


# ================================================================ tactics

CATALOGUE = [
    # defence: where the block sits (pick one)
    {"id": "high_press", "side": "defence", "group": "block", "name": "High press", "needs_metres": True,
     "about": "Win the ball high: the back line up at halfway and players straight onto the ball.",
     "played": "the back line holds 45 m or more from its own goal",
     "worked": "the player on the ball has one of yours within 5 m (needs the ball in view)"},
    {"id": "mid_block", "side": "defence", "group": "block", "name": "Mid block", "needs_metres": True,
     "about": "Let them have the ball at the back, engage around halfway, stay compact.",
     "played": "the back line holds 28-45 m from its own goal and the team is 35 m or less from back to front",
     "worked": "12 m or less between the lines"},
    {"id": "low_block", "side": "defence", "group": "block", "name": "Low block", "needs_metres": True,
     "about": "Defend the box: sit deep and narrow, no space between or behind the lines.",
     "played": "the back line holds 28 m or less from its own goal",
     "worked": "28 m or less from back to front and 10 m or less between the lines"},
    # defence: how the back line and block behave (any)
    {"id": "high_line", "side": "defence", "group": "line", "name": "High line / offside trap", "needs_metres": True,
     "about": "A level back line held high, stepping up together to catch runners offside.",
     "played": "the back four hold 35 m or more from goal and within 3 m of each other",
     "worked": "no opposition player gets in behind the back line"},
    {"id": "two_banks", "side": "defence", "group": "line", "name": "Two banks of four", "needs_metres": True,
     "about": "A back four and a midfield four, close together, with two up front.",
     "played": "with 9+ outfield players in view, the two deepest lines each have four players",
     "worked": "12 m or less between the two banks"},
    {"id": "counter_press", "side": "defence", "group": "line", "name": "Counter-press", "needs_metres": True,
     "about": "Win it back straight after losing it, before the opposition can look up.",
     "played": "2 or more players within 5 m of the ball within 2 s of losing it (needs the ball in view)",
     "worked": "the ball is won back within 6 s"},
    # attack
    {"id": "build_up", "side": "attack", "group": "build", "name": "Build up from the back", "needs_metres": True,
     "about": "Play out on the ground: centre-backs split wide and the keeper joins in.",
     "played": "when the ball is won deep, two players 25 m or more apart across the pitch stay near the own box (needs the ball)",
     "worked": "the ball crosses halfway without a long ball and without losing it"},
    {"id": "direct", "side": "attack", "group": "build", "name": "Direct / long ball", "needs_metres": True,
     "about": "Skip the midfield and play early into the forwards.",
     "played": "a spell of possession from the own half has a pass of 30 m or more forward (needs the ball)",
     "worked": "the team still has the ball 5 s after it lands"},
    {"id": "switch", "side": "attack", "group": "build", "name": "Switch the play", "needs_metres": True,
     "about": "Move the ball quickly from one side to the other to find the free space.",
     "played": "a spell of 10 s or more has the ball move 30 m across within 3 s (needs the ball)",
     "worked": "nobody from the other team within 5 m when it arrives"},
    {"id": "overload", "side": "attack", "group": "space", "name": "Overload the ball side", "needs_metres": True,
     "about": "More players than them around the ball to pass through.",
     "played": "more of yours than theirs within 15 m of the ball (needs the ball)",
     "worked": "the ball moves 15 m or more forward within 5 s"},
    {"id": "wide", "side": "attack", "group": "space", "name": "Hold the width", "needs_metres": True,
     "about": "A player on each touchline to stretch the defence.",
     "played": "a player within 8 m of each touchline",
     "worked": None},
    {"id": "counter_attack", "side": "attack", "group": "space", "name": "Counter-attack", "needs_metres": True,
     "about": "Go forward fast after winning the ball, before the other team is set.",
     "played": "after winning the ball in the own half, it goes 30 m forward within 10 s (needs the ball)",
     "worked": "the ball reaches the penalty box within 10 s"},
    {"id": "rest_defence", "side": "attack", "group": "space", "name": "Rest defence", "needs_metres": True,
     "about": "Keep players behind the ball while attacking, ready for a lost ball.",
     "played": "4 or more outfield players 10 m behind the ball when attacking in their half (needs the ball)",
     "worked": "after losing the ball there, the other team doesn't go 30 m forward within 10 s"},
    {"id": "overlap", "side": "attack", "group": "space", "name": "Overlapping runs", "needs_metres": True,
     "about": "A full-back or midfielder runs round the outside of the player on the ball on the wing.",
     "played": "with the ball in a wide channel, a teammate from behind and outside gets ahead of the ball within 3 s (needs the ball)",
     "worked": "the runner gets the ball within 5 s"},
    {"id": "possession", "side": "attack", "group": "build", "name": "Possession play", "needs_metres": True,
     "about": "Keep the ball with short passes and move the other team around until a gap opens.",
     "played": "a spell of 5 s or more has 5 or more passes (needs the ball)",
     "worked": "the spell reaches the final third (70 m or more from your own goal)"},
    # defence: marking and pressing traps
    {"id": "man_marking", "side": "defence", "group": "marking", "name": "Man-marking", "needs_metres": True,
     "about": "Every defender picks up one opponent and follows them.",
     "played": "60% or more of the outfield players are within 3 m of the same opponent as a second earlier",
     "worked": "their player receiving a pass has one of yours within 3 m (needs the ball)"},
    {"id": "zonal", "side": "defence", "group": "marking", "name": "Zonal defending", "needs_metres": True,
     "about": "Players hold zones and the whole block slides with the ball.",
     "played": "fewer than 30% follow one opponent, and the block's middle stays within 12 m (across) of the ball",
     "worked": "38 m or less across and 12 m or less between the lines"},
    {"id": "press_wide", "side": "defence", "group": "marking", "name": "Show them wide (touchline trap)", "needs_metres": True,
     "about": "Let them go wide, then close the ball in against the touchline.",
     "played": "with their ball in a wide channel (15 m from touch), 3 or more of yours within 15 m of it (needs the ball)",
     "worked": "the player on the ball has one of yours within 5 m"},
    # set pieces
    {"id": "zonal_corners", "side": "defence", "group": "set", "name": "Zonal at corners", "needs_metres": True,
     "about": "Defenders cover zones of the box at corners instead of following runners.",
     "played": "at their corners, 30% or fewer of your box defenders are within 1.5 m of an attacker",
     "worked": "you get the first touch"},
    {"id": "man_corners", "side": "defence", "group": "set", "name": "Man-to-man at corners", "needs_metres": True,
     "about": "Each defender picks up one attacker in the box at corners.",
     "played": "at their corners, 60% or more of your box defenders are within 1.5 m of an attacker",
     "worked": "you get the first touch"},
    {"id": "short_corners", "side": "attack", "group": "set", "name": "Short corners", "needs_metres": True,
     "about": "Play corners short to change the angle of the cross.",
     "played": "the ball stays within 15 m of the corner flag and out of the box for 3 s",
     "worked": "you still have the ball 5 s later"},
    {"id": "short_goal_kicks", "side": "attack", "group": "set", "name": "Short goal kicks", "needs_metres": True,
     "about": "Play out from goal kicks to a centre-back or midfielder.",
     "played": "the first touch after the goal kick is within 30 m of it",
     "worked": "your team gets that first touch"},
    {"id": "long_goal_kicks", "side": "attack", "group": "set", "name": "Long goal kicks", "needs_metres": True,
     "about": "Kick long and win the first or second ball up the pitch.",
     "played": "the first touch after the goal kick is 30 m or more from it",
     "worked": "your team gets that first touch"},
]
BY_ID = {t["id"]: t for t in CATALOGUE}
EVENT_CHECKS = ("direct", "switch", "build_up", "counter_press", "counter_attack", "overlap", "possession",
                "zonal_corners", "man_corners", "short_corners", "short_goal_kicks", "long_goal_kicks")
NEEDS_BALL = {"high_press", "counter_press", "build_up", "direct", "switch", "overload", "counter_attack", "rest_defence",
              "overlap", "possession", "press_wide", "zonal_corners", "man_corners", "short_corners", "short_goal_kicks",
              "long_goal_kicks"}
UNITS = {"direct": "spells from your own half", "switch": "spells of 10 s or more",
         "build_up": "spells starting near your own goal", "counter_press": "lost balls",
         "counter_attack": "balls won in your own half", "overlap": "spells on the wing", "possession": "spells of 5 s or more",
         "zonal_corners": "corners against you", "man_corners": "corners against you", "short_corners": "corners",
         "short_goal_kicks": "goal kicks", "long_goal_kicks": "goal kicks"}
MARK_R = 3.0            # m from an opponent = marking them
MARK_SHARE, ZONE_SHARE = 0.6, 0.3
WIDE_CHANNEL = 15.0     # m from a touchline = the ball is wide
OVERLAP_TIME = 3.0      # s for the overlapping runner to get ahead of the ball
POSSESSION_PASSES = 5
FINAL_THIRD = 70.0


def _marks(ctx, frames, f, team, opp):
    """{team player id: nearest opponent id within MARK_R} for team's outfield in frame f."""
    xy = ctx["xy"][f]
    if xy is None:
        return None
    pl = list(zip(frames[f][0], xy))
    theirs = [(p["id"], x, y) for p, (x, y) in pl if p["bucket"] == opp and p["id"] is not None]
    out = {}
    for p, (x, y) in pl:
        if p["bucket"] != team or p["id"] is None or p["id"] in ctx["keepers"]:
            continue
        near = min(((math.hypot(x - a, y - b), i) for i, a, b in theirs), default=None)
        out[p["id"]] = near[1] if near and near[0] <= MARK_R else None
    return out


def _follow_share(ctx, frames, f, team, opp):
    """Share of team's outfield players marking the same opponent as a second earlier."""
    now = _marks(ctx, frames, f, team, opp)
    g = f - int(ctx["fps"])
    before = _marks(ctx, frames, g, team, opp) if g >= 0 else None
    if not now or before is None or len(now) < MIN_OUTFIELD:
        return None
    same = sum(1 for k, v in now.items() if v is not None and before.get(k) == v)
    return same / len(now)


def catalogue():
    return CATALOGUE


def _frame_check(tid, ctx, frames, f, team, opp):
    """(played, worked) for a frame-by-frame tactic; None = can't judge this frame."""
    s = ctx["shape"][team][f]
    if not s:
        return None, None
    if tid in ("high_press", "mid_block", "low_block"):
        if "line_height" not in s:
            return None, None
        h = s["line_height"]
        if tid == "high_press":
            if h < HIGH_LINE:
                return False, None
            if ctx["poss"][f] != opp:
                return True, None
            _, d = _ball_idx(ctx, frames, f, team)
            return True, (None if d is None else d <= PRESS_REACH)
        if tid == "mid_block":
            if "length" not in s:
                return None, None
            played = LOW_BLOCK < h < HIGH_LINE and s["length"] <= COMPACT
            return played, (s["line_gap"] <= LINE_GAP_GOOD if played and "line_gap" in s else None)
        played = h <= LOW_BLOCK
        if not played or "length" not in s:
            return played, None
        return True, s["length"] <= LOW_BLOCK and s.get("line_gap", 0) <= 10
    if tid == "high_line":
        if "back_spread" not in s:
            return None, None
        played = s["line_height"] >= HIGH_LINE_TRAP and s["back_spread"] <= 3.0
        if not played:
            return False, None
        o = ctx["shape"][opp][f]
        last = s["pos"][0][0]
        # the opposition in `team`'s own direction
        behind = o and any(ctx["L"] - x < last - 2.0 for x, _ in o["pos"])
        return True, not behind
    if tid == "two_banks":
        if s["n"] < 9 or "lines" not in s:
            return None, None
        played = len(s["lines"]) >= 2 and s["lines"][0] == 4 and s["lines"][1] == 4
        if not played:
            return False, None
        xs = [x for x, _ in s["pos"]]
        mids = [sum(xs[:4]) / 4, sum(xs[4:8]) / 4]
        return True, mids[1] - mids[0] <= LINE_GAP_GOOD
    if tid in ("man_marking", "zonal"):
        share = _follow_share(ctx, frames, f, team, opp)
        if share is None:
            return None, None
        if tid == "man_marking":
            return share >= MARK_SHARE, None
        b = ctx["ball"][f]
        cy = median(y for _, y in s["pos"])
        played = share < ZONE_SHARE and (b is None or abs(cy - b[1]) <= 12.0)
        if not played or "width" not in s:
            return played, None
        return True, s["width"] <= NARROW_OUT and s.get("line_gap", 0) <= LINE_GAP_GOOD
    if tid == "press_wide":
        b = ctx["ball"][f]
        if b is None or ctx["poss"][f] != opp or WIDE_CHANNEL < b[1] < ctx["W"] - WIDE_CHANNEL:
            return None, None
        near = sum(1 for p, (x, y) in zip(frames[f][0], ctx["xy"][f])
                   if p["bucket"] == team and math.hypot(x - b[0], y - b[1]) <= OVERLOAD_REACH)
        if near < 3:
            return False, None
        _, d = _ball_idx(ctx, frames, f, team)
        return True, None if d is None else d <= PRESS_REACH
    if tid == "wide":
        if "wings" not in s:
            return None, None
        return all(s["wings"]), None
    if tid == "overload":
        b = ctx["ball"][f]
        if b is None or ctx["poss"][f] != team:
            return None, None
        near = lambda t: sum(1 for p, (x, y) in zip(frames[f][0], ctx["xy"][f])
                             if p["bucket"] == t and math.hypot(x - b[0], y - b[1]) <= OVERLOAD_REACH)
        if not near(team) and not near(opp):
            return None, None
        if near(team) < near(opp) + 1:
            return False, None
        bx = _own_x(ctx, team, b[0])
        later = [ctx["ball"][g] for g in range(f + 1, min(ctx["n"], f + int(OVERLOAD_TIME * ctx["fps"]) + 1))
                 if ctx["ball"][g] is not None and ctx["poss"][g] == team]
        if not later:
            return True, None
        return True, max(_own_x(ctx, team, g[0]) for g in later) - bx >= OVERLOAD_GAIN
    if tid == "rest_defence":
        b = ctx["ball"][f]
        if b is None or ctx["poss"][f] != team:
            return None, None
        bx = _own_x(ctx, team, b[0])
        if bx < ctx["L"] / 2:
            return None, None
        return sum(1 for x, _ in s["pos"] if x < bx - REST_BEHIND) >= REST_MIN, None
    return None, None


def _spells(ctx, team):
    """Spells of possession for `team`: [(first frame, last frame)]."""
    out, start = [], None
    for f, p in enumerate(ctx["poss"] + [None]):
        if p == team and start is None:
            start = f
        elif p != team and start is not None:
            out.append((start, f - 1))
            start = None
    return out


def _ball_path(ctx, a, b):
    return [(g, ctx["ball"][g]) for g in range(a, min(b, ctx["n"] - 1) + 1) if ctx["ball"][g] is not None]


def _pos(ctx, frames, f, tid, team):
    """(own x, y) of track `tid` in frame f, or None."""
    if f >= ctx["n"] or ctx["xy"][f] is None:
        return None
    for p, (x, y) in zip(frames[f][0], ctx["xy"][f]):
        if p["id"] == tid:
            return _own_x(ctx, team, float(x)), float(y)
    return None


def _event_checks(tid, ctx, frames, team, opp, extra=None):
    """[(frame, played, worked)] for tactics judged per spell or per lost ball."""
    fps, L, W = ctx["fps"], ctx["L"], ctx["W"]
    extra = extra or {}
    out = []
    if tid in ("zonal_corners", "man_corners", "short_corners", "short_goal_kicks", "long_goal_kicks"):
        kind = "goal_kick" if "goal_kicks" in tid else "corner"
        for e in extra.get("set_pieces", []):
            if e["kind"] != kind or (e["against"] if tid in ("zonal_corners", "man_corners") else e["team"]) != team:
                continue
            if tid in ("zonal_corners", "man_corners"):
                if not e.get("marking"):
                    continue
                played = e["marking"] == ("zonal" if tid == "zonal_corners" else "man")
                out.append((e["f"], played, (e["first_ball"] == team) if played and e["first_ball"] else None))
            elif tid == "short_corners":
                if "short" not in e:
                    continue
                out.append((e["f"], e["short"], e["kept"] if e["short"] else None))
            else:
                if "short" not in e:
                    continue
                played = e["short"] if tid == "short_goal_kicks" else not e["short"]
                out.append((e["f"], played, (e["first_ball"] == team) if played and e["first_ball"] else None))
        return out
    if tid == "possession":
        ps = [p for p in extra.get("passes", []) if p["team"] == team]
        for a, b in _spells(ctx, team):
            if b - a < 5 * fps:
                continue
            n = sum(1 for p in ps if a <= p["release"] <= b)
            reached = any(_own_x(ctx, team, ctx["ball"][g][0]) >= FINAL_THIRD for g in range(a, b + 1) if ctx["ball"][g])
            played = n >= POSSESSION_PASSES
            out.append((a, played, reached if played else None))
        return out
    if tid == "overlap":
        for a, b in _spells(ctx, team):
            wide = [g for g in range(a, b + 1) if ctx["ball"][g] and ctx["owner_id"][g]
                    and (ctx["ball"][g][1] <= WIDE_CHANNEL or ctx["ball"][g][1] >= W - WIDE_CHANNEL)]
            if not wide:
                continue
            f = wide[0]
            bx, by = _own_x(ctx, team, ctx["ball"][f][0]), ctx["ball"][f][1]
            touch = 0.0 if by <= WIDE_CHANNEL else W
            carrier = ctx["owner_id"][f][0]
            runner = None
            if ctx["xy"][f] is not None:
                for p, (x, y) in zip(frames[f][0], ctx["xy"][f]):
                    if p["bucket"] != team or p["id"] in (carrier, None) or p["id"] in ctx["keepers"]:
                        continue
                    if _own_x(ctx, team, float(x)) >= bx - 2 or abs(y - touch) > abs(by - touch) + 3:
                        continue
                    for g in range(f + 1, min(b, f + int(OVERLAP_TIME * fps)) + 1):
                        q, ball = _pos(ctx, frames, g, p["id"], team), ctx["ball"][g]
                        if q and ball and q[0] > _own_x(ctx, team, ball[0]) + 2 and abs(q[1] - touch) <= abs(ball[1] - touch) + 3:
                            runner = p["id"]
                            break
                    if runner:
                        break
            got = None
            if runner:
                got = any(ctx["owner_id"][g] and ctx["owner_id"][g][0] == runner
                          for g in range(f, min(ctx["n"], f + int(5 * fps) + 1)))
            out.append((f, bool(runner), got))
        return out
    if tid in ("direct", "switch", "build_up"):
        for a, b in _spells(ctx, team):
            path = _ball_path(ctx, a, b)
            if len(path) < 3:
                continue
            first = _own_x(ctx, team, path[0][1][0])
            if tid == "switch" and (b - a) < MIN_SPELL * fps:
                continue
            if tid == "direct" and first >= L / 2:
                continue
            if tid == "build_up" and first >= 30:
                continue
            hit = None
            for i, (g, p) in enumerate(path):
                for h, q in path[i + 1:]:
                    if h - g > (LONG_BALL_TIME if tid != "switch" else SWITCH_TIME) * fps:
                        break
                    if tid == "switch" and abs(q[1] - p[1]) >= SWITCH:
                        hit = (g, h)
                    elif tid != "switch" and _own_x(ctx, team, q[0]) - _own_x(ctx, team, p[0]) >= LONG_BALL:
                        hit = (g, h)
                    if hit:
                        break
                if hit:
                    break
            if tid == "direct":
                worked = None
                if hit:
                    k = hit[1] + int(KEEP_AFTER * fps)
                    worked = ctx["poss"][k] == team if k < ctx["n"] else None
                out.append((a, bool(hit), worked))
            elif tid == "switch":
                worked = None
                if hit:
                    _, d = _ball_idx(ctx, frames, hit[1], opp)
                    worked = None if d is None else d > PRESS_REACH
                out.append((a, bool(hit), worked))
            else:
                s = ctx["shape"][team]
                split = False
                for g in range(a, min(b, a + int(3 * fps)) + 1):
                    if s[g]:
                        deep = [y for x, y in s[g]["pos"] if x < 25]
                        if len(deep) >= 2 and max(deep) - min(deep) >= 25:
                            split = True
                            break
                crossed = any(_own_x(ctx, team, p[0]) > L / 2 for _, p in path)
                out.append((a, split, (crossed and not hit) if split else None))
    elif tid == "counter_press":
        for t in ctx["turnovers"]:
            if t["lost"] != team:
                continue
            f = t["f"]
            played = False
            for g in range(f, min(ctx["n"], f + int(PRESS_AFTER * fps) + 1)):
                b, xy = ctx["ball"][g], ctx["xy"][g]
                if b is None or xy is None:
                    continue
                near = sum(1 for p, (x, y) in zip(frames[g][0], xy)
                           if p["bucket"] == team and math.hypot(x - b[0], y - b[1]) <= PRESS_REACH)
                if near >= 2:
                    played = True
                    break
            back = any(ctx["poss"][g] == team for g in range(f + 1, min(ctx["n"], f + int(WIN_BACK * fps) + 1)))
            out.append((f, played, back if played else None))
    elif tid == "counter_attack":
        for t in ctx["turnovers"]:
            if t["won"] != team or ctx["ball"][t["f"]] is None:
                continue
            f = t["f"]
            x0 = _own_x(ctx, team, ctx["ball"][f][0])
            if x0 >= L / 2:
                continue
            path = _ball_path(ctx, f, f + int(COUNTER_TIME * fps))
            gone = max((_own_x(ctx, team, p[0]) for _, p in path), default=x0) - x0
            box = any(_own_x(ctx, team, p[0]) >= L - pitch.BOX_DEPTH and abs(p[1] - W / 2) <= pitch.BOX_HALF
                      for _, p in path)
            out.append((f, gone >= LONG_BALL, box if gone >= LONG_BALL else None))
    return out


def _rest_worked(ctx, team):
    """Lost balls in the other half: did the other team go 30 m forward within 10 s?"""
    res = []
    for t in ctx["turnovers"]:
        if t["lost"] != team or ctx["ball"][t["f"]] is None:
            continue
        x0 = _own_x(ctx, team, ctx["ball"][t["f"]][0])
        if x0 < ctx["L"] / 2:
            continue
        path = _ball_path(ctx, t["f"], t["f"] + int(COUNTER_TIME * ctx["fps"]))
        lowest = min((_own_x(ctx, team, p[0]) for _, p in path), default=x0)
        res.append((t["f"], x0 - lowest < LONG_BALL))
    return res


def _tpoint(t, kind, title, detail, moments=()):
    return {"kind": kind, "title": title, "detail": detail, "moments": list(moments), "tactic": t["id"],
            "why": f"Your plan: {t['about']}" + (" Checked from the video; confirm each moment before acting on it."
                                                  if kind != "info" else "")}


def _moment(frames, fps, f, label, idx=None):
    players = frames[f][0] if f < len(frames) else []
    hl = analysis._hl_bounds([players[i] for i in idx]) if idx else (analysis._hl_bounds(players) if players else "")
    return {"frame": f, "t": round(f / fps, 2), "time": analysis._fmt_t(f / fps), "hl": hl, "label": label}


def evaluate(chosen, ctx, frames, team, opp, phase, extra=None):
    """Report points for the football tactics `team`'s coach picked."""
    fps = ctx["fps"]
    T = team.capitalize()
    points = []
    for tid in chosen:
        t = BY_ID.get(tid)
        if not t:
            continue
        if not ctx["mapped"]:
            points.append(_tpoint(t, "info", f"{t['name']}: mark the pitch to check it",
                                  "This tactic is measured in metres. Mark the pitch in the Pitch tab and make the report again."))
            continue
        if tid in NEEDS_BALL and not ctx["use_poss"] and tid not in ("high_press",):
            points.append(_tpoint(t, "info", f"{t['name']}: the ball wasn't seen enough",
                                  f"This check needs to know who had the ball, and the ball was seen with a player in only "
                                  f"{ctx['ball_share']:.0%} of the measured footage. A higher camera, or a clip where the ball "
                                  "stays in view, helps."))
            continue
        target = PLAYED_ENOUGH_RARE.get(tid, PLAYED_ENOUGH)
        if tid not in EVENT_CHECKS and not ctx["use_poss"] and phase == "mixed":
            points.append(_tpoint(t, "info", f"{t['name']}: pick Attacking or Defending",
                                  "The ball was seen too rarely to tell when your team had it. Pick Attacking or "
                                  "Defending above so the report knows which moments to check."))
            continue
        if tid in EVENT_CHECKS:
            checks = _event_checks(tid, ctx, frames, team, opp, extra)
            unit = UNITS[tid]
            if len(checks) < MIN_EVENTS:
                points.append(_tpoint(t, "info", f"{t['name']}: not enough to judge",
                                      f"Only {len(checks)} {unit} could be checked (at least {MIN_EVENTS} needed)."))
                continue
            played = [c for c in checks if c[1]]
            tried = [c for c in played if c[2] is not None]
            worked = [c for c in tried if c[2]]
            misses = [_moment(frames, fps, f, f"{t['name']} not played") for f, p, _ in checks if not p]
            misses = [_moment(frames, fps, f, f"{t['name']} played, didn't work") for f, p, w in checks if p and w is False] + misses
            share = len(played) / len(checks)
            works = len(worked) / len(tried) if tried else None
            detail = f"Played in {len(played)} of {len(checks)} {unit} ({share:.0%}); played means {t['played']}."
        else:
            side = t["side"]
            fs = _frames(ctx, team, opp, side, phase)
            res = [(f,) + _frame_check(tid, ctx, frames, f, team, opp) for f in fs]
            res = [r for r in res if r[1] is not None]
            if len(res) < 2 * fps:
                what = "defending" if side == "defence" else "attacking"
                points.append(_tpoint(t, "info", f"{t['name']}: not enough to judge",
                                      f"{T} was seen {what} with enough of the team in view for only {len(res) / fps:.0f}s "
                                      "(at least 2 s needed)."))
                continue
            played = [r for r in res if r[1]]
            tried = [r for r in played if r[2] is not None]
            worked = [r for r in tried if r[2]]
            share = len(played) / len(res)
            if tid == "rest_defence":
                rw = _rest_worked(ctx, team)
                tried, worked = rw, [r for r in rw if r[1]]
            if tid == "man_marking":
                # Their receivers: was one of yours within 3 m when the ball arrived?
                rw = []
                for ps in (extra or {}).get("passes", []):
                    if ps["team"] != opp:
                        continue
                    q = _pos(ctx, frames, ps["receive"], ps["receiver"], team)
                    if q is None:
                        continue
                    near = min((math.hypot(_own_x(ctx, team, float(x)) - q[0], float(y) - q[1])
                                for p, (x, y) in zip(frames[ps["receive"]][0], ctx["xy"][ps["receive"]])
                                if p["bucket"] == team), default=99)
                    rw.append((ps["receive"], near <= MARK_R))
                tried, worked = rw, [r for r in rw if r[1]]
            per_event = tid in ("rest_defence", "man_marking")
            works = len(worked) / len(tried) if len(tried) >= (MIN_EVENTS if per_event else fps) else None
            no = {r[0]: (1, "") for r in res if not r[1]}
            bad = {r[0]: (1, "") for r in tried if r[2] is False} if not per_event else {}
            if per_event:
                bad = {f: (1, "") for f, ok in rw if not ok}
            misses = [_moment(frames, fps, e["frame"], f"{t['name']} played, didn't work") for e in analysis._events(bad, fps)] + \
                     [_moment(frames, fps, e["frame"], f"{t['name']} not played") for e in analysis._events(no, fps)]
            detail = f"Played {share:.0%} of the {len(res) / fps:.0f}s it could be checked; played means {t['played']}."
        if works is not None and t["worked"]:
            if tid in EVENT_CHECKS or tid in ("rest_defence", "man_marking"):
                detail += f" It worked {len(worked)} of {len(tried)} times ({works:.0%}): {t['worked']}."
            else:
                detail += f" It worked {works:.0%} of the time it was played: {t['worked']}."
        kind = "good" if share >= target and (works is None or works >= 0.5) else "issue"
        if share < target:
            title = f"{t['name']}: only played {share:.0%} of the time"
        elif works is not None and works < 0.5:
            title = f"{t['name']}: played, but worked only {works:.0%} of the time"
        else:
            title = f"{t['name']}: stuck to it {share:.0%} of the time" + (f", worked {works:.0%}" if works is not None else "")
        points.append(_tpoint(t, kind, title, detail, misses[:4]))
    return points
