"""
Shrinks an upload before it is analysed, so each account's videos take far
less disk space (and less traffic when they're watched or downloaded).

Phone footage is often 1080p or 4K at 60 frames a second, 50-400 MB per
minute. The detector only looks at 1280 pixels across anyway, so the upload
is re-encoded to at most 1280 pixels on its longest side, at most 30 frames a
second, H.264, without sound. That is usually 5 to 15 times smaller and
makes the rest of the analysis faster too. The original file is deleted
afterwards. If FFmpeg isn't available or fails, the original is kept as is.
"""

import os
import subprocess

MAX_SIDE = 1280
MAX_FPS = 30
CRF = 26  # H.264 quality: lower = better and bigger; 23 is FFmpeg's default


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def shrink(src, dst, seconds=0, progress=None):
    """Write a smaller copy of `src` to `dst`. Returns True on success (and
    only then is `dst` a complete file). progress(done, total) gets
    milliseconds of video processed."""
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        return False
    total_ms = int(seconds * 1000)
    tmp = dst + ".part.mp4"
    cmd = [
        ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-i", src,
        "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
        # Fit inside MAX_SIDE x MAX_SIDE (upright phone videos too), never
        # enlarge, even sizes for H.264; cap the frame rate.
        "-vf", (f"scale='if(gte(iw,ih),min({MAX_SIDE},iw),-2)':'if(gte(iw,ih),-2,min({MAX_SIDE},ih))',"
                f"fps='min({MAX_FPS},source_fps)'"),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", tmp,
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError:
        return False
    try:
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and progress and total_ms and value.isdigit():
                progress(min(total_ms, int(value) // 1000), total_ms)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False
    os.replace(tmp, dst)
    if progress and total_ms:
        progress(total_ms, total_ms)
    return True
