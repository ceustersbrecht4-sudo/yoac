"""
Checks that an upload really is match footage before any analysis runs, so
the app can't be used to store or pass around other videos (gore, violence,
explicit or otherwise unsuitable material).

Two gates:
1. check_upload(): straight after the upload, before it is queued. The file
   must be a readable video of sensible size and length, and most sampled
   frames must show a sports pitch (a large area of grass-green). Runs in
   about a second and never sends anything off the laptop.
2. check_players(): after detection. The video must actually contain several
   people in a good share of the frames, like a match does.

Rejected uploads are deleted straight away. This is an automatic filter, not
a guarantee: the terms of use forbid unsuitable content, every upload is
linked to an account, and the owner can review and delete any upload.
"""

import cv2
import numpy as np

MAX_SECONDS = 2 * 60 * 60        # 2 hours
MAX_SIDE = 4096                  # pixels; bigger frames are refused (decoder abuse / memory)
MIN_SECONDS = 1
SAMPLES = 16                     # frames sampled across the video
PITCH_GREEN = 0.18               # share of grass-green pixels for a frame to count as "pitch"
PITCH_FRAMES = 0.4               # share of sampled frames that must show a pitch
PEOPLE_MIN = 3                   # people needed in a frame to look like play...
PEOPLE_FRAMES = 0.3              # ...in at least this share of frames


class ContentRejected(Exception):
    """The upload isn't acceptable; the message is safe to show the user."""


def _green_share(frame):
    small = cv2.resize(frame, (160, max(1, int(160 * frame.shape[0] / frame.shape[1]))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Grass, natural or artificial, in sun, shade or floodlight.
    green = (h >= 30) & (h <= 90) & (s >= 40) & (v >= 35)
    return float(np.count_nonzero(green)) / green.size


def check_upload(path):
    """Raise ContentRejected unless this looks like usable match footage.
    Returns basic facts about the video."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ContentRejected("This file couldn't be opened as a video. Try an MP4 from your phone or camera.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
            raise ContentRejected("This file couldn't be read as a video. Try an MP4 from your phone or camera.")
        if max(width, height) > MAX_SIDE:
            raise ContentRejected(f"This video is larger than {MAX_SIDE} pixels. Export it at 4K or lower.")
        seconds = count / fps
        if seconds < MIN_SECONDS:
            raise ContentRejected("This video is too short to analyse.")
        if seconds > MAX_SECONDS:
            raise ContentRejected("This video is longer than 2 hours. Trim it to the part you want to analyse.")

        pitch_frames = read = 0
        for i in range(SAMPLES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * count / SAMPLES))
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            read += 1
            if _green_share(frame) >= PITCH_GREEN:
                pitch_frames += 1
        if read < max(3, SAMPLES // 4):
            raise ContentRejected("This video couldn't be read properly. Try exporting it again as MP4.")
        if pitch_frames < PITCH_FRAMES * read:
            raise ContentRejected("This doesn't look like match footage: no pitch was found in most of the video. "
                                  "Only upload videos of rugby matches or training.")
        return {"fps": fps, "frames": count, "seconds": seconds, "width": width, "height": height}
    finally:
        cap.release()


def check_players(all_detections):
    """Raise ContentRejected unless several people are visible in a good share
    of the frames, as in a match."""
    if not all_detections:
        raise ContentRejected("No frames could be read from this video.")
    busy = sum(1 for dets in all_detections if sum(1 for d in dets if d[4] != "ball") >= PEOPLE_MIN)
    if busy < PEOPLE_FRAMES * len(all_detections):
        raise ContentRejected("No match was found in this video: there were hardly any players on the pitch. "
                              "Only upload videos of rugby matches or training.")
