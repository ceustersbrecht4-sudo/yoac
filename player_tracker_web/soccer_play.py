"""
Soccer moments, both teams: set pieces, what happens after the ball is
lost (pressing and getting back), passes and possible offside.

Built on soccer.context(): the ball and every player in metres, who is on
the ball, and which end each team defends. Everything needs the pitch
marked; the set pieces and passes also need the ball in view.

Set pieces are found the way a coach would spot them on video: the ball
sits still for a second or more, and where it sits says what it is (a
corner arc, the touchline, the goal area, a spot, or anywhere else = a free
kick). The law distances come from the Laws of the Game (9.15 m at free
kicks and corners); the rest are rules of thumb named below.
"""

import math
from statistics import median

import analysis
import pitch
import soccer

# ---- set pieces
STILL = 0.8             # m: the ball stays within this of where it stopped
STILL_TIME = 1.0        # s it must stay still for a restart
STILL_GAP = 0.5         # s the ball may go unseen during the stop
STILL_SEEN = 0.5        # share of the stop it must be seen
KICK_MOVE = 1.5         # m from the spot = the ball is played
KICK_WAIT = 3.0         # s after the stop to look for the kick
CORNER_R = 2.5          # m from a corner flag = a corner
SPOT_R = 1.2            # m from a penalty spot = a penalty
CENTRE_R = 1.5          # m from the centre spot = a kick-off
TOUCH_R = 1.5           # m from a touchline = a throw-in
AREA_PAD = 1.0          # m around the goal area = a goal kick
CONTESTED = 3.0         # m: both teams this close to a still ball = a scramble, not a free kick
RUN_UP = 1.0            # m: at a free kick nobody stands on the ball for a while (the taker steps back),
RUN_UP_TIME = 0.5       # s  unlike a player holding it at their feet
TAKER_R = 3.0           # m: the nearest player within this at the kick took it
WALL = 9.15             # m: opponents must be this far away at free kicks and corners (Law 13)
WALL_MARGIN = 1.0       # m of measuring error before a wall counts as too close
SHOOTING_RANGE = 35.0   # m from goal: a free kick in shooting range
FIRST_BALL = 4.0        # s after the kick to see who gets the first touch
KEEP = 5.0              # s after the restart the taking team should still have it
SHORT_CORNER = 15.0     # m: a corner that stays within this and out of the box is short
SHORT_GOAL_KICK = 30.0  # m: a goal kick whose first touch is nearer than this is short
MARKED = 1.5            # m from an attacker = marking them (at corners)
MAN_SHARE, ZONAL_SHARE = 0.6, 0.3   # share of defenders marking = man-to-man / zonal

# ---- after the ball is lost
PRESS_WINDOW = 3.0      # s after losing the ball to look for pressure
PRESS_FAST = 2.0        # s: pressure within this = a counter-press
RECOVER = 5.0           # s after losing the ball to count who got back
BACK_GOOD, BACK_FEW = 8, 6   # outfield players behind the ball 5 s after losing it

# ---- passes and offside
PASS_GAP = 2.5          # s between the passer's last touch and the receiver's first
PASS_MIN = 5.0          # m the ball travels for a pass
OFFSIDE_MARGIN = 1.0    # m past the second-last defender before it counts (one camera isn't exact)


def _near(ctx, frames, f, x, y, r):
    """Players within r m of (x, y) in frame f: [(index, team, distance)]."""
    xy = ctx["xy"][f]
    if xy is None:
        return []
    out = []
    for i, (p, (px, py)) in enumerate(zip(frames[f][0], xy)):
        if p["bucket"] in ctx["teams"]:
            d = math.hypot(px - x, py - y)
            if d <= r:
                out.append((i, p["bucket"], d))
    return sorted(out, key=lambda t: t[2])


def _measured(ctx, f, reach=None):
    """The nearest frame to f (within `reach` s) with players in metres."""
    reach = int((reach or 0.5) * ctx["fps"])
    for d in range(reach + 1):
        for g in (f - d, f + d):
            if 0 <= g < ctx["n"] and ctx["xy"][g] is not None:
                return g
    return None


def _hl_near(ctx, frames, f, x, y, r=15.0):
    g = _measured(ctx, f)
    if g is None:
        return g, ""
    idx = [i for i, _, _ in _near(ctx, frames, g, x, y, r)]
    return g, analysis._hl_bounds([frames[g][0][i] for i in idx]) if idx else ""


def _other(ctx, team):
    return next(t for t in ctx["teams"] if t != team)


def _goal_x(ctx, team):
    """x of the goal `team` defends."""
    return ctx["L"] if ctx["flip"][team] else 0.0


# ================================================================ set pieces

def _stops(ctx):
    """Moments the ball sat still: [(first frame, last frame, x, y)]."""
    fps = ctx["fps"]
    out, run, anchor, last = [], [], None, None

    def close():
        if run and run[-1] - run[0] >= STILL_TIME * fps and len(run) >= STILL_SEEN * (run[-1] - run[0] + 1):
            xs, ys = [ctx["ball"][f][0] for f in run], [ctx["ball"][f][1] for f in run]
            out.append((run[0], run[-1], median(xs), median(ys)))

    for f in range(ctx["n"]):
        b = ctx["ball"][f]
        if b is None:
            if run and f - last > STILL_GAP * fps:
                close()
                run, anchor = [], None
            continue
        if anchor and math.hypot(b[0] - anchor[0], b[1] - anchor[1]) <= STILL:
            run.append(f)
        else:
            close()
            run, anchor = [f], b
        last = f
    close()
    return out


def _kind(ctx, x, y):
    L, W = ctx["L"], ctx["W"]
    if min(math.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W)) <= CORNER_R:
        return "corner"
    if min(math.hypot(x - sx, y - W / 2) for sx in (pitch.SPOT, L - pitch.SPOT)) <= SPOT_R:
        return "penalty"
    if math.hypot(x - L / 2, y - W / 2) <= CENTRE_R:
        return "kick_off"
    if (x <= pitch.AREA_DEPTH + AREA_PAD or x >= L - pitch.AREA_DEPTH - AREA_PAD) and abs(y - W / 2) <= pitch.AREA_HALF + AREA_PAD:
        return "goal_kick"
    if y <= TOUCH_R or y >= W - TOUCH_R:
        return "throw_in"
    return "free_kick"


KIND_NAME = {"corner": "corner", "penalty": "penalty", "kick_off": "kick-off", "goal_kick": "goal kick",
             "throw_in": "throw-in", "free_kick": "free kick"}


def set_pieces(ctx, frames):
    """Every restart the camera saw: kind, who took it, and what happened."""
    fps, L, W = ctx["fps"], ctx["L"], ctx["W"]
    out = []
    for f0, f1, x, y in _stops(ctx):
        kind = _kind(ctx, x, y)
        # the kick: the first time the ball is well away from the spot
        kick = None
        for g in range(f1 + 1, min(ctx["n"], f1 + int(KICK_WAIT * fps) + 1)):
            b = ctx["ball"][g]
            if b is not None and math.hypot(b[0] - x, b[1] - y) > KICK_MOVE:
                kick = g
                break
        k = kick if kick is not None else f1
        g = _measured(ctx, k - 1)
        near = _near(ctx, frames, g, x, y, TAKER_R) if g is not None else []
        if kind == "free_kick":
            # Both teams on a still ball is a scramble or a player shielding it, not a restart.
            late = range(max(f0, f1 - int(fps)), f1 + 1)
            if any(len({t for _, t, _ in _near(ctx, frames, h, x, y, CONTESTED)}) > 1 for h in late if ctx["xy"][h] is not None):
                continue
            clear = [h for h in range(f0, f1 + 1) if ctx["xy"][h] is not None and not _near(ctx, frames, h, x, y, RUN_UP)]
            if len(clear) < RUN_UP_TIME * fps:
                continue
        taker = near[0][1] if near else next((ctx["poss"][h] for h in range(k, min(ctx["n"], k + int(fps))) if ctx["poss"][h]), None)
        if taker is None:
            continue
        opp = _other(ctx, taker)
        _, hl = _hl_near(ctx, frames, k, x, y)
        ev = {"kind": kind, "f": k, "t": round(k / fps, 2), "stop": f0, "x": round(x, 1), "y": round(y, 1),
              "team": taker, "against": opp, "kicked": kick is not None, "hl": hl}
        after = range(k + 1, min(ctx["n"], k + int(FIRST_BALL * fps) + 1))
        first_f = next((h for h in after if ctx["owner_id"][h]), None)
        ev["first_ball"] = ctx["owner_id"][first_f][1] if first_f is not None else None
        later = k + int(KEEP * fps)
        ev["kept"] = (ctx["poss"][later] == taker) if later < ctx["n"] and ctx["poss"][later] else None
        if kind == "corner":
            gx = 0.0 if x < L / 2 else L
            in_box = lambda px, py: abs(px - gx) <= pitch.BOX_DEPTH and abs(py - W / 2) <= pitch.BOX_HALF
            g = _measured(ctx, k)
            if g is not None:
                pl = list(zip(frames[g][0], ctx["xy"][g]))
                att = [(px, py) for p, (px, py) in pl if p["bucket"] == taker and in_box(px, py)]
                dfn = [(px, py) for p, (px, py) in pl if p["bucket"] == opp and in_box(px, py)
                       and p["id"] not in ctx["keepers"] and abs(px - gx) > 1.0]
                ev["attackers_in_box"], ev["defenders_in_box"] = len(att), len(dfn)
                if dfn and att:
                    marking = sum(1 for d in dfn if min(math.hypot(d[0] - a[0], d[1] - a[1]) for a in att) <= MARKED) / len(dfn)
                    ev["marking"] = "man" if marking >= MAN_SHARE else "zonal" if marking <= ZONAL_SHARE else "mixed"
                ev["hl"] = analysis._hl_bounds([p for p, (px, py) in pl if in_box(px, py)]) or ev["hl"]
            path = [ctx["ball"][h] for h in range(k, min(ctx["n"], k + int(3 * fps))) if ctx["ball"][h] is not None]
            if len(path) >= 3:
                ev["short"] = all(math.hypot(b[0] - x, b[1] - y) <= SHORT_CORNER and not in_box(*b) for b in path)
        elif kind == "free_kick":
            goal = _goal_x(ctx, opp)
            ev["distance"] = round(math.hypot(x - goal, y - W / 2), 1)
            ev["shooting"] = ev["distance"] <= SHOOTING_RANGE
            closest, at = None, None
            for h in range(max(f0, f1 - int(fps)), f1 + 1):
                for i, t, d in _near(ctx, frames, h, x, y, WALL):
                    if t == opp and (closest is None or d < closest):
                        closest, at = d, (h, i)
            ev["wall"] = round(closest, 1) if closest is not None else None
            ev["too_close"] = closest is not None and closest < WALL - WALL_MARGIN
            if ev["too_close"]:
                h, i = at
                ev["hl"] = analysis._hl_box(frames[h][0][i]["box"])
                ev["f"], ev["t"] = h, round(h / fps, 2)
        elif kind == "goal_kick" and first_f is not None and ctx["ball"][first_f] is not None:
            b = ctx["ball"][first_f]
            ev["short"] = math.hypot(b[0] - x, b[1] - y) < SHORT_GOAL_KICK
        out.append(ev)
    return out


def set_piece_summary(events, teams):
    if not events:
        return {"count": 0}
    out = {"count": len(events), "by_kind": {}}
    for k in KIND_NAME:
        mine = [e for e in events if e["kind"] == k]
        if mine:
            out["by_kind"][k] = {t: sum(1 for e in mine if e["team"] == t) for t in teams}
    return out


def _m(ev, label):
    return {"frame": ev["f"], "t": ev["t"], "time": analysis._fmt_t(ev["t"]), "hl": ev["hl"], "label": label}


def set_piece_points(events, team, phase):
    """Coach points about `team`'s set pieces, taken and defended."""
    T = team.capitalize()
    pts = []
    ours = lambda k: [e for e in events if e["kind"] == k and e["team"] == team]
    theirs = lambda k: [e for e in events if e["kind"] == k and e["against"] == team]
    if phase in ("defence", "mixed"):
        cs = [e for e in theirs("corner") if "defenders_in_box" in e]
        if cs:
            short = [e for e in cs if e["attackers_in_box"] > e["defenders_in_box"]]
            won = [e for e in theirs("corner") if e["first_ball"]]
            first = sum(1 for e in won if e["first_ball"] == team)
            styles = [e.get("marking") for e in cs if e.get("marking")]
            style = max(set(styles), key=styles.count) if styles else None
            pts.append({
                "kind": "issue" if short else "good" if won and first / len(won) >= 0.5 else "info",
                "title": f"Defending corners: outnumbered in the box at {len(short)} of {len(cs)}" if short
                         else f"Defending corners: {len(cs)} seen",
                "detail": f"At the kick, {T} had {median(e['defenders_in_box'] for e in cs):.0f} outfield players in the box "
                          f"against {median(e['attackers_in_box'] for e in cs):.0f} attackers (typical)."
                          + (f" Marking looked mostly {'man-to-man' if style == 'man' else style} (a defender within "
                             f"{MARKED:.1f} m of an attacker or not)." if style else "")
                          + (f" {T} got the first touch at {first} of {len(won)}." if won else ""),
                "why": "More attackers than defenders in the box means someone is free at the back post. Count your "
                       "players onto posts and zones, and check who picks up the late runner.",
                "moments": [_m(e, f"{e['attackers_in_box']} v {e['defenders_in_box']} in the box") for e in short[:4]]})
        fks = theirs("free_kick")
        close = [e for e in fks if e["too_close"]]
        if fks and close:
            pts.append({
                "kind": "issue", "title": f"Too close at {len(close)} free kick(s)",
                "detail": f"At {len(close)} of {len(fks)} free kicks against {T}, a {T} player stood nearer than "
                          f"{WALL - WALL_MARGIN:.0f} m to the ball before it was taken (closest {min(e['wall'] for e in close):.1f} m).",
                "why": "Opponents must be 9.15 m away (Law 13). Standing closer can be a yellow card and the kick is retaken, "
                       "sometimes from a better spot.",
                "moments": [_m(e, f"{e['wall']:.1f} m from the ball") for e in close[:4]]})
    if phase in ("attack", "mixed"):
        th = [e for e in ours("throw_in") if e["kept"] is not None]
        if len(th) >= 3:
            kept = sum(1 for e in th if e["kept"])
            lost = [e for e in th if not e["kept"]]
            pts.append({
                "kind": "good" if kept / len(th) >= 0.7 else "issue" if kept / len(th) < 0.5 else "info",
                "title": f"Throw-ins kept: {kept} of {len(th)}",
                "detail": f"{T} still had the ball {KEEP:.0f} s after {kept} of their {len(th)} throw-ins.",
                "why": "Throw-ins are the most common restart, and teams often give them away. Movement to get free "
                       "(a run one way, then come short) keeps the ball.",
                "moments": [_m(e, "Ball lost after the throw") for e in lost[:4]]})
        gk = [e for e in ours("goal_kick") if "short" in e]
        if len(gk) >= 2:
            sh = [e for e in gk if e["short"]]
            kept = [e for e in gk if e["first_ball"] == team]
            pts.append({
                "kind": "info", "title": f"Goal kicks: {len(sh)} short, {len(gk) - len(sh)} long",
                "detail": f"{T}'s first touch after the goal kick came at {len(kept)} of {len(gk)}. Short means the first "
                          f"touch was within {SHORT_GOAL_KICK:.0f} m of the kick.",
                "why": "Long goal kicks are a contest for the first and second ball; short ones need the centre-backs split "
                       "and a free midfielder. Check it matches your plan.",
                "moments": [_m(e, "Lost the first ball") for e in gk if e["first_ball"] and e["first_ball"] != team][:4]})
        cs = ours("corner")
        if cs:
            won = [e for e in cs if e["first_ball"]]
            first = sum(1 for e in won if e["first_ball"] == team)
            box = [e["attackers_in_box"] for e in cs if "attackers_in_box" in e]
            sh = sum(1 for e in cs if e.get("short"))
            pts.append({
                "kind": "info", "title": f"Corners taken: {len(cs)}",
                "detail": (f"{T} had {median(box):.0f} players in the box at the kick (typical). " if box else "")
                          + (f"{sh} were played short. " if sh else "")
                          + (f"{T} got the first touch at {first} of {len(won)}." if won else ""),
                "why": "Winning the first contact is what turns a corner into a chance. Check the delivery and the timing of runs.",
                "moments": [_m(e, "Defence won the first ball") for e in won if e["first_ball"] != team][:4]})
        fks = [e for e in ours("free_kick") if e["shooting"]]
        if fks:
            pts.append({"kind": "info", "title": f"Free kicks in shooting range: {len(fks)}",
                        "detail": f"{T} had {len(fks)} free kick(s) within {SHOOTING_RANGE:.0f} m of goal "
                                  f"(closest {min(e['distance'] for e in fks):.0f} m).",
                        "why": "Worth reviewing: shot or cross, and where the wall stood.",
                        "moments": [_m(e, f"{e['distance']:.0f} m out") for e in fks[:4]]})
    return pts


# ====================================================== after losing the ball

def transitions(ctx, frames):
    """For every lost ball: how fast the team got pressure on it, and how
    many got back behind the ball."""
    fps = ctx["fps"]
    out = []
    for t in ctx["turnovers"]:
        f, team = t["f"], t["lost"]
        b0 = ctx["ball"][f]
        if b0 is None or ctx["xy"][f] is None:
            continue
        press = None
        for g in range(f, min(ctx["n"], f + int(PRESS_WINDOW * fps) + 1)):
            b = ctx["ball"][g]
            if b is not None and any(tm == team for _, tm, _ in _near(ctx, frames, g, b[0], b[1], soccer.PRESS_REACH)):
                press = (g - f) / fps
                break
        rec = {"f": f, "t": round(f / fps, 2), "team": team, "press": press, "x": soccer._own_x(ctx, team, b0[0])}
        for key, g in (("back_at", f), ("back_after", f + int(RECOVER * fps))):
            g = _measured(ctx, g, 0.3)
            s = ctx["shape"][team][g] if g is not None else None
            b = ctx["ball"][g] if g is not None else None
            if s and b is not None and s["n"] >= soccer.MIN_OUTFIELD:
                bx = soccer._own_x(ctx, team, b[0])
                rec[key] = sum(1 for x, _ in s["pos"] if x < bx)
                rec[key + "_seen"] = s["n"]
        rec["hl"] = _hl_near(ctx, frames, f, b0[0], b0[1], 12.0)[1]
        out.append(rec)
    return out


def transition_points(trans, team):
    T = team.capitalize()
    mine = [r for r in trans if r["team"] == team]
    pts = []
    if len(mine) >= 3:
        timed = [r["press"] for r in mine if r["press"] is not None]
        fast = [r for r in mine if r["press"] is not None and r["press"] <= PRESS_FAST]
        slow = [r for r in mine if r["press"] is None or r["press"] > PRESS_FAST]
        share = len(fast) / len(mine)
        pts.append({
            "kind": "good" if share >= 0.5 else "issue" if share < 0.25 else "info",
            "title": f"Pressure after losing the ball: within {PRESS_FAST:.0f} s at {share:.0%}",
            "detail": f"{T} lost the ball {len(mine)} times in the measured footage. A {T} player got within "
                      f"{soccer.PRESS_REACH:.0f} m of the ball within {PRESS_FAST:.0f} s {len(fast)} times"
                      + (f" (typically after {median(timed):.1f} s)." if timed else "."),
            "why": "The first seconds after losing the ball are when the other team is least organised. Pressure then "
                   "(the counter-press) wins it back or forces a hurried pass.",
            "moments": [_m(r, "No pressure on the ball" if r["press"] is None else f"Pressure after {r['press']:.1f} s")
                        for r in sorted(slow, key=lambda r: -(r["press"] or 99))[:4]]})
    back = [r for r in mine if "back_after" in r]
    if len(back) >= 3:
        mb = median(r["back_after"] for r in back)
        few = [r for r in back if r["back_after"] < BACK_FEW]
        pts.append({
            "kind": "good" if mb >= BACK_GOOD else "issue" if mb < BACK_FEW else "info",
            "title": f"Getting back: {mb:.0f} behind the ball {RECOVER:.0f} s after losing it",
            "detail": f"{RECOVER:.0f} s after losing the ball, {T} typically had {mb:.0f} outfield players between the ball "
                      f"and their own goal (of those in view).",
            "why": "If the first press doesn't win it back, everyone else has to sprint back behind the ball. "
                   f"Fewer than {BACK_FEW} back leaves gaps for a counter-attack.",
            "moments": [_m(r, f"{r['back_after']} back after {RECOVER:.0f} s") for r in few[:4]]})
    return pts


# ================================================================== passes

def passes(ctx):
    """Passes between teammates: [{release, receive, team, passer, receiver}]."""
    fps = ctx["fps"]
    owned = [(f, o[0], o[1]) for f, o in enumerate(ctx["owner_id"]) if o]
    out = []
    for (f, a, ta), (g, b, tb) in zip(owned, owned[1:]):
        if ta != tb or a == b or g - f > PASS_GAP * fps or a is None or b is None:
            continue
        p, q = ctx["ball"][f], ctx["ball"][g]
        if p is None or q is None or math.hypot(q[0] - p[0], q[1] - p[1]) < PASS_MIN:
            continue
        out.append({"release": f, "receive": g, "team": ta, "passer": a, "receiver": b,
                    "forward": soccer._own_x(ctx, ta, q[0]) - soccer._own_x(ctx, ta, p[0])})
    return out


def offsides(ctx, frames, pass_list):
    """Passes received by a player who was in an offside position when the
    ball was played: in the other half, nearer the goal line than the ball
    and than the second-last defender."""
    fps = ctx["fps"]
    out = []
    for ps in pass_list:
        r, team = ps["release"], ps["team"]
        d = _other(ctx, team)
        xy, b = ctx["xy"][r], ctx["ball"][r]
        if xy is None or b is None:
            continue
        players = frames[r][0]
        own = lambda tm, x: soccer._own_x(ctx, tm, x)
        recv = [(own(d, x), y, i) for i, (p, (x, y)) in enumerate(zip(players, xy)) if p["id"] == ps["receiver"]]
        if not recv:
            continue
        rx, ry, ri = recv[0]
        defenders = sorted((own(d, x), p["id"] in ctx["keepers"]) for p, (x, y) in zip(players, xy) if p["bucket"] == d)
        if not defenders:
            continue
        keeper_seen = any(k for _, k in defenders)
        line = defenders[1][0] if keeper_seen and len(defenders) >= 2 else defenders[0][0]
        # Only judge it when the space behind the line is on screen (no defender hidden deeper).
        gx = lambda x: ctx["L"] - x if ctx["flip"][d] else x
        if not soccer._seen(ctx["view"][r], gx(max(0.0, line - soccer.VIEW_MARGIN)), ry):
            continue
        bx = own(d, b[0])
        if rx < ctx["L"] / 2 and rx < bx and rx < line - OFFSIDE_MARGIN:
            out.append({"f": r, "t": round(r / fps, 2), "team": team, "against": d, "margin": round(float(line - rx), 1),
                        "hl": analysis._hl_box(players[ri]["box"])})
    return out


def offside_points(offs, team, phase):
    T = team.capitalize()
    pts = []
    att = [o for o in offs if o["team"] == team]
    dfn = [o for o in offs if o["against"] == team]
    if att and phase in ("attack", "mixed"):
        pts.append({"kind": "issue", "title": f"Possible offside: {len(att)} pass(es)",
                    "detail": f"{len(att)} time(s) a {T} player received a pass after standing more than "
                              f"{OFFSIDE_MARGIN:.0f} m past the second-last defender when it was played (up to "
                              f"{max(o['margin'] for o in att):.1f} m).",
                    "why": "Runs that start too early waste the attack. Check the timing: hold the run on the line, "
                           "go as the passer's head comes up. One camera can be off by a metre, so check each one.",
                    "moments": [_m(o, f"{o['margin']:.1f} m past the line") for o in att[:4]]})
    if dfn and phase in ("defence", "mixed"):
        pts.append({"kind": "info", "title": f"Caught {len(dfn)} attacker(s) offside",
                    "detail": f"{len(dfn)} time(s) a pass went to an attacker standing past {T}'s second-last defender.",
                    "why": "If your back line steps up on purpose, this is the trap working. If not, check who was "
                           "watching the runner: the flag isn't always raised.",
                    "moments": [_m(o, f"{o['margin']:.1f} m offside") for o in dfn[:4]]})
    return pts
