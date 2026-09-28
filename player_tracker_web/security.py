"""
Security building blocks for the web app.

- Response headers, including a Content Security Policy: scripts only run if
  they carry this request's random nonce, so injected <script> tags and inline
  event handlers are blocked (defence against cross-site scripting).
- A small in-memory rate limiter (per network address and per account).
- Password rules, including a leak check against Have I Been Pwned. Only the
  first 5 characters of the password's SHA-1 hash leave the laptop
  (k-anonymity); the password itself never does.
- TOTP two-factor codes (RFC 6238, works with any authenticator app) and
  one-time recovery codes.
"""

import base64
import hashlib
import hmac
import ipaddress
import os
import secrets
import socket
import struct
import threading
import time
import urllib.request
from urllib.parse import quote

from flask import abort, g, request

# ------------------------------------------------------------------ headers

def csp_nonce():
    if not hasattr(g, "csp_nonce"):
        g.csp_nonce = secrets.token_urlsafe(16)
    return g.csp_nonce


def _host_allowed(host):
    """Only answer requests addressed to this laptop: localhost, a private
    network address, or the laptop's own name. This stops DNS rebinding, where
    a web page on another site tricks a browser into talking to the app."""
    if host.startswith("["):                      # [IPv6]:port
        name = host[1:host.find("]")] if "]" in host else ""
    else:
        name = host.rsplit(":", 1)[0]              # name:port or ipv4:port
    name = name.lower().rstrip(".")
    if name in ("localhost",) or name in _own_names():
        return True
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


_OWN = None


def _own_names():
    global _OWN
    if _OWN is None:
        me = socket.gethostname().lower()
        extra = {h.strip().lower() for h in os.environ.get("YOAC_ALLOWED_HOSTS", "").split(",") if h.strip()}
        _OWN = {me, me + ".local", me + ".lan", me + ".home"} | extra
    return _OWN


def init_security_headers(app):
    app.jinja_env.globals["csp_nonce"] = csp_nonce

    def _check_host():
        if not _host_allowed(request.host or ""):
            abort(400)
    # First of all checks, before the login redirect (which would otherwise
    # build a link with the foreign host name in it).
    app.before_request_funcs.setdefault(None, []).insert(0, _check_host)

    @app.after_request
    def _headers(resp):
        nonce = csp_nonce()
        resp.headers["Content-Security-Policy"] = "; ".join([
            "default-src 'self'",
            f"script-src 'self' 'nonce-{nonce}'",
            "style-src 'self' 'unsafe-inline'",   # inline styles only; they can't run code
            "img-src 'self' data: blob:",
            "media-src 'self' blob:",
            "font-src 'self'",
            "connect-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ])
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "same-origin"
        resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=()"
        resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        if resp.mimetype == "text/html":
            resp.headers["Cache-Control"] = "no-store"  # pages show account data
        return resp


# ------------------------------------------------------------------ online

def _env_on(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def init_online(app):
    """Settings for when the app is put online. Off by default (Wi-Fi use).

    YOAC_BEHIND_PROXY=1  the app runs behind a web server (Caddy, nginx, a
                         hosting platform) that handles HTTPS. Then the
                         visitor's real address is taken from that server's
                         X-Forwarded-For header, so rate limits work per
                         visitor. Never set this without such a server in
                         front: anyone could fake the header.
    YOAC_HTTPS=1         the site is only reached over HTTPS: the login
                         cookie is only ever sent encrypted, and browsers are
                         told to always use HTTPS (HSTS).
    YOAC_ALLOWED_HOSTS   your domain name(s), e.g. tracker.yoac.be (see
                         _host_allowed).
    """
    if _env_on("YOAC_BEHIND_PROXY"):
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    https = _env_on("YOAC_HTTPS")
    if https:
        app.config.update(SESSION_COOKIE_SECURE=True, PREFERRED_URL_SCHEME="https")

    @app.after_request
    def _hsts(resp):
        if https:
            resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return resp


# ------------------------------------------------------------ rate limiting

class Throttle:
    """At most `limit` hits per `window` seconds for each key; after that the
    key is blocked until the oldest hit falls out of the window."""

    def __init__(self, limit, window):
        self.limit, self.window = limit, window
        self._hits = {}
        self._lock = threading.Lock()

    def _recent(self, key, now):
        hits = [t for t in self._hits.get(key, []) if now - t < self.window]
        if hits:
            self._hits[key] = hits
        else:
            self._hits.pop(key, None)
        return hits

    def wait(self, key):
        """Seconds until `key` may try again (0 = allowed now)."""
        now = time.time()
        with self._lock:
            hits = self._recent(key, now)
            if len(hits) >= self.limit:
                return int(self.window - (now - hits[0])) + 1
        return 0

    def hit(self, key):
        now = time.time()
        with self._lock:
            self._hits.setdefault(key, []).append(now)
            # Online, many different addresses pass by: forget the old ones
            # so the list can't grow without end.
            if len(self._hits) > 5000:
                for k in list(self._hits):
                    self._recent(k, now)

    def clear(self, key):
        with self._lock:
            self._hits.pop(key, None)


# ----------------------------------------------------------- password rules

MIN_PASSWORD = 10
# A short list of the most common choices; the leak check catches the rest.
COMMON = {
    "password", "password1", "password123", "passw0rd", "wachtwoord", "motdepasse", "qwerty", "qwertyuiop",
    "azerty", "azertyuiop", "123456789", "1234567890", "12345678", "11111111", "iloveyou", "welcome1",
    "letmein", "football", "rugbyrugby", "admin123", "changeme", "abc12345", "trustno1", "sunshine1",
}


def pwned_count(password, timeout=3.0):
    """How often this password appears in known breaches, or None if the check
    couldn't run (offline). Uses the k-anonymity range API: only the first 5
    hex characters of the SHA-1 hash are sent."""
    digest = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix, suffix = digest[:5], digest[5:]
    req = urllib.request.Request(
        "https://api.pwnedpasswords.com/range/" + prefix,
        headers={"User-Agent": "YOAC-Player-Tracker", "Add-Padding": "true"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    for line in body.splitlines():
        tail, _, count = line.partition(":")
        if tail.strip() == suffix:
            return int(count.strip() or 0)
    return 0


def password_problem(password, username):
    """A message saying what's wrong with this password, or None if it's fine."""
    if len(password) < MIN_PASSWORD:
        return f"Use a password of at least {MIN_PASSWORD} characters."
    if len(password) > 200:
        return "That password is too long (200 characters at most)."
    lowered = password.lower()
    if username and username.lower() in lowered:
        return "Your password can't contain your username."
    if lowered in COMMON or len(set(password)) < 4:
        return "That password is too easy to guess. Try a few unrelated words together."
    count = pwned_count(password)
    if count:
        return (f"That password has appeared in {count:,} data breaches, so attackers try it first. "
                "Please choose a different one.")
    return None


# ------------------------------------------------------------- TOTP / 2FA

TOTP_PERIOD = 30
TOTP_DIGITS = 6


def new_totp_secret():
    return base64.b32encode(os.urandom(20)).decode("ascii").rstrip("=")


def _hotp(secret, counter):
    key = base64.b32decode(secret + "=" * (-len(secret) % 8), casefold=True)
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** TOTP_DIGITS)
    return str(code).zfill(TOTP_DIGITS)


def totp_now(secret, at=None):
    return _hotp(secret, int((at or time.time()) // TOTP_PERIOD))


def verify_totp(secret, code, last_step=None):
    """Checks a 6-digit code, allowing one step of clock drift either way.
    Returns the time step it matched (store it to stop the same code being
    used twice), or None."""
    code = "".join(ch for ch in str(code) if ch.isdigit())
    if len(code) != TOTP_DIGITS or not secret:
        return None
    now = int(time.time() // TOTP_PERIOD)
    for step in (now - 1, now, now + 1):
        if last_step is not None and step <= last_step:
            continue
        if hmac.compare_digest(_hotp(secret, step), code):
            return step
    return None


def totp_uri(secret, username, issuer="YOAC Player Tracker"):
    label = quote(f"{issuer}:{username}")
    return f"otpauth://totp/{label}?secret={secret}&issuer={quote(issuer)}&digits={TOTP_DIGITS}&period={TOTP_PERIOD}"


def totp_qr_svg(uri):
    """QR code as inline SVG if the optional `qrcode` package is installed."""
    try:
        import io
        import qrcode
        import qrcode.image.svg
    except ImportError:
        return None
    img = qrcode.make(uri, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode("utf-8")


def new_recovery_codes(n=8):
    """Human-friendly one-time codes like 7KQ4-M2XD."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return ["".join(secrets.choice(alphabet) for _ in range(4)) + "-" + "".join(secrets.choice(alphabet) for _ in range(4))
            for _ in range(n)]


def normalise_recovery(code):
    return "".join(ch for ch in code.upper() if ch.isalnum())
