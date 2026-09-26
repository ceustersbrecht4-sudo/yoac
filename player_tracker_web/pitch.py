"""
From pixels to metres on the pitch.

1. The coach marks 4+ points on one frame where pitch lines cross (say "left
   22 x near touchline"). Those give a homography: a mapping from that
   frame's pixels to pitch metres.
2. The camera pans and zooms, so the mapping changes every frame.
   camera_motion() follows the background (pitch markings, grass, boards,
   not the players) from each frame to the next and stores how the picture
   moved. Chaining those moves from a marked frame gives a mapping for every
   frame around it, until a cut (replay, close-up, another camera).
3. Every mapped frame is checked against the players: if most of them would
   land off the pitch, that frame's mapping is not trusted.

Pitch coordinates: x = metres along the pitch from the LEFT try line (as seen
from the camera), y = metres across from the NEAR touchline.
"""

import math

import cv2
import numpy as np

DEFAULT_LENGTH = 100.0  # try line to try line (World Rugby: at most 100 m)
DEFAULT_WIDTH = 70.0    # touchline to touchline (at most 70 m)
MOTION_WIDTH = 640      # frames are shrunk to this width to follow the camera
MIN_INLIERS = 25        # background points that must agree on a move
ON_PITCH_MARGIN = 6.0   # metres outside the lines still counted as "on the pitch"
ON_PITCH_SHARE = 0.6    # share of players that must land on the pitch


def lines_across(length):
    """Lines that cross the pitch (constant x), left to right."""
    half = length / 2
    return {
        "left_try": ("Left try line", 0.0),
        "left_5": ("Left 5 m line", 5.0),
        "left_22": ("Left 22", 22.0),
        "left_10": ("Left 10 m line", half - 10),
        "halfway": ("Halfway line", half),
        "right_10": ("Right 10 m line", half + 10),
        "right_22": ("Right 22", length - 22),
        "right_5": ("Right 5 m line", length - 5),
        "right_try": ("Right try line", length),
    }


def lines_along(width):
    """Lines that run the length of the pitch (constant y), near to far."""
    return {
        "near_touch": ("Near touchline", 0.0),
        "near_5": ("Near 5 m line", 5.0),
        "near_15": ("Near 15 m line", 15.0),
        "far_15": ("Far 15 m line", width - 15),
        "far_5": ("Far 5 m line", width - 5),
        "far_touch": ("Far touchline", width),
    }


class CalibrationError(ValueError):
    """Message is safe to show the coach."""


def solve(points):
    """points: [(img_x, img_y, pitch_x, pitch_y), ...] on one frame.
    Returns the 3x3 pixel -> metres homography."""
    if len(points) < 4:
        raise CalibrationError("Each view needs at least 4 marked points.")
    img = np.array([(p[0], p[1]) for p in points], dtype=np.float64)
    world = np.array([(p[2], p[3]) for p in points], dtype=np.float64)
    if len({(round(x, 1), round(y, 1)) for x, y in world}) < len(world):
        raise CalibrationError("Two points are the same pitch spot. Give each point a different pair of lines.")
    if min(np.linalg.norm(img[i] - img[j]) for i in range(len(img)) for j in range(i + 1, len(img))) < 8:
        raise CalibrationError("Two marked points are on top of each other. Spread them out.")
    hull = cv2.convexHull(world.astype(np.float32))
    if cv2.contourArea(hull) < 40:
        raise CalibrationError("The points are almost in a straight line on the pitch. Use points on at least "
                               "two different lines across and two different lines along the pitch.")
    H, _ = cv2.findHomography(img, world, 0)
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        raise CalibrationError("These points don't fit together. Check each point is on the lines you picked.")
    back = project(H, img)
    err = float(np.max(np.linalg.norm(back - world, axis=1)))
    if err > 4.0:  # only possible with 5+ points
        raise CalibrationError(f"The points don't agree with each other (one is about {err:.0f} m off). "
                               "Check the lines picked for each point.")
    return H / H[2, 2]


def project(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    if not len(pts):
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def camera_motion(video_path, all_detections, progress=None):
    """For each frame i, the 3x3 move that takes frame i's pixels to frame
    i-1's pixels (row of NaNs where it can't be followed: first frame, cuts).
    Returns an (n, 9) float32 array."""
    cap = cv2.VideoCapture(video_path)
    n = len(all_detections)
    moves = np.full((n, 9), np.nan, dtype=np.float32)
    prev = None
    scale = None
    try:
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            if scale is None:
                scale = min(1.0, MOTION_WIDTH / frame.shape[1])
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                M = _follow(prev[0], gray, prev[1])
                if M is not None:
                    S = np.diag([scale, scale, 1.0])
                    moves[i] = (np.linalg.inv(S) @ M @ S).reshape(9)
            prev = (gray, _background_mask(gray.shape, all_detections[i - 1] if i else [], scale))
            if progress and i % 25 == 0:
                progress(i, n)
    finally:
        cap.release()
    if progress:
        progress(n, n)
    return moves


def _background_mask(shape, detections, scale):
    """Everywhere except the players (they move on their own)."""
    mask = np.full(shape, 255, dtype=np.uint8)
    for x1, y1, x2, y2, *_ in detections:
        pad = 4
        cv2.rectangle(mask, (int(x1 * scale) - pad, int(y1 * scale) - pad),
                      (int(x2 * scale) + pad, int(y2 * scale) + pad), 0, -1)
    return mask


def _follow(prev_gray, gray, mask):
    """Move from `gray` pixels to `prev_gray` pixels, or None (a cut, or not
    enough background to follow)."""
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.005, minDistance=7, mask=mask)
    if pts is None or len(pts) < MIN_INLIERS:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    good = status.reshape(-1) == 1
    if good.sum() < MIN_INLIERS:
        return None
    a, b = pts[good].reshape(-1, 2), nxt[good].reshape(-1, 2)
    M, inliers = cv2.findHomography(b, a, cv2.RANSAC, 2.0)
    if M is None or inliers is None or inliers.sum() < max(MIN_INLIERS, 0.4 * len(a)):
        return None
    M = M / M[2, 2]
    # A camera can't jump far or zoom a lot in one frame: that's a cut.
    h, w = gray.shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)
    if np.max(np.linalg.norm(moved - corners, axis=1)) > 0.25 * w:
        return None
    return M


def frame_mappings(n, moves, keyframes):
    """Pixel -> metres mapping for every frame (None where unknown).
    keyframes: {frame: H}. Each frame uses the nearest marked frame it can
    be followed from."""
    maps = [None] * n
    dist = [math.inf] * n
    for k, H in keyframes.items():
        if not 0 <= k < n:
            continue
        maps[k], dist[k] = H, 0
        C = np.eye(3)
        for i in range(k + 1, n):  # forward: p_(i-1) = M_i p_i
            if moves is None or i >= len(moves) or np.isnan(moves[i][0]):
                break
            C = C @ moves[i].reshape(3, 3).astype(np.float64)
            C /= C[2, 2]
            if i - k >= dist[i]:
                break
            maps[i], dist[i] = H @ C, i - k
        C = np.eye(3)
        for i in range(k - 1, -1, -1):  # backward: p_i = inv(M_(i+1)) p_(i+1)
            m = moves[i + 1] if moves is not None and i + 1 < len(moves) else None
            if m is None or np.isnan(m[0]):
                break
            C = C @ np.linalg.inv(m.reshape(3, 3).astype(np.float64))
            C /= C[2, 2]
            if k - i >= dist[i]:
                break
            maps[i], dist[i] = H @ C, k - i
    return maps


def check_with_players(maps, frames, length, width):
    """Drop mappings that put most players off the pitch (the following
    drifted, or the view changed). frames: analysis.prepare() output."""
    lo_x, hi_x = -ON_PITCH_MARGIN, length + ON_PITCH_MARGIN
    lo_y, hi_y = -ON_PITCH_MARGIN, width + ON_PITCH_MARGIN
    for i, H in enumerate(maps):
        if H is None:
            continue
        players = frames[i][0] if i < len(frames) else []
        if not players:
            maps[i] = None  # close-up / replay: nothing to measure
            continue
        xy = project(H, [(p["x"], p["y"]) for p in players])
        inside = np.sum((xy[:, 0] >= lo_x) & (xy[:, 0] <= hi_x) & (xy[:, 1] >= lo_y) & (xy[:, 1] <= hi_y))
        if inside < ON_PITCH_SHARE * len(players):
            maps[i] = None
    return maps
