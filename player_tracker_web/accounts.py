"""
Accounts for the local web app: log in, invite-only sign-up, two-factor codes,
password changes and an owner admin area.

Everything lives in files next to this one - nothing leaves the laptop:
  users.json    accounts: salted password hashes, role, 2FA secret, recovery-code hashes
  invites.json  single-use invite codes (hashed) made by the owner
  secret.key    signs the session cookie

Rules, all enforced on the server:
- The first account created becomes the owner. After that, signing up needs
  an invite code from the owner, so nobody on the Wi-Fi can just make an
  account and see the videos.
- Only the owner can make invites or manage other accounts (checked here,
  never in the browser).
- Logins are rate-limited per network address and per account; so are
  sign-ups, 2FA codes and password changes.
- The session is an HttpOnly, SameSite cookie - page scripts can't read it.
  Changing or resetting a password (or turning 2FA off) logs that account out
  everywhere.
"""

import functools
import hashlib
import json
import os
import re
import secrets
import threading
import time
from datetime import timedelta
from urllib.parse import urlparse

from flask import abort, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from security import (Throttle, new_recovery_codes, new_totp_secret, normalise_recovery, password_problem,
                      totp_qr_svg, totp_uri, verify_totp, MIN_PASSWORD)

HERE = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(HERE, "users.json")
INVITES_FILE = os.path.join(HERE, "invites.json")
SECRET_FILE = os.path.join(HERE, "secret.key")

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
REMEMBER_DAYS = 30
TERMS_VERSION = "2026-09-25"  # bump when the terms or privacy policy change
INVITE_DAYS = 7
PENDING_2FA_SECONDS = 300

# Pages and files anyone can open without logging in.
PUBLIC_ENDPOINTS = {"login", "static"}
# Pages a user who must change their password may still open.
FORCED_CHANGE_OK = {"account", "change_password", "logout", "legal_page", "static"}
# Paths the upload page calls from JavaScript: answer with JSON, not a redirect.
API_PREFIXES = ("/analyze", "/status/", "/hide/", "/delete/", "/report/", "/snapshot/", "/frame/", "/video/")

# Rate limits (in memory; reset when the app restarts).
login_ip = Throttle(10, 10 * 60)       # wrong passwords from one address
login_user = Throttle(5, 15 * 60)      # wrong passwords for one account, from anywhere
signup_ip = Throttle(5, 60 * 60)       # sign-up attempts from one address
code_user = Throttle(5, 15 * 60)       # wrong 2FA / recovery codes for one account
sensitive_user = Throttle(5, 15 * 60)  # wrong current password on account changes
admin_actions = Throttle(30, 60 * 60)  # owner actions (invites, resets)

_users_lock = threading.Lock()

# Set by app.py: the owner's list of all uploads, and deleting videos.
uploads_overview = lambda: []
delete_user_videos = lambda user_key: None
delete_video = lambda job_id: False
usage_for = lambda user_key: None


# ------------------------------------------------------------------ storage

def _secret_key():
    if os.path.exists(SECRET_FILE):
        with open(SECRET_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    key = secrets.token_hex(32)
    with open(SECRET_FILE, "w") as f:
        f.write(key)
    return key


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _load_users():
    users = _read(USERS_FILE)
    # Accounts made before roles existed: the oldest one becomes the owner.
    if users and not any(u.get("role") == "owner" for u in users.values()):
        oldest = min(users, key=lambda k: users[k].get("created", 0))
        users[oldest]["role"] = "owner"
        _write(USERS_FILE, users)
    return users


def _save_users(users):
    _write(USERS_FILE, users)


def _hash_code(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ helpers

def _safe_next(target):
    """Only follow redirects back into this app (no //evil.com or https://...)."""
    if not target:
        return None
    parts = urlparse(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return None
    return target


def _csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(24)
    return session["csrf"]


def _csrf_ok():
    return secrets.compare_digest(request.form.get("csrf", ""), session.get("csrf", "") or "x")


def _ip():
    return request.remote_addr or "?"


def current_user():
    return session.get("user")


def user_is_owner():
    me = _me()
    return bool(me and me.get("role") == "owner")


def _me():
    """The logged-in user's record, or None (also None if the session is stale)."""
    key = current_user()
    if not key:
        return None
    user = _load_users().get(key)
    if not user or session.get("epoch") != user.get("epoch", 0):
        return None
    return user


def _start_session(key, user, remember):
    session.clear()  # new session contents on login (no session fixation)
    session["user"] = key
    session["epoch"] = user.get("epoch", 0)
    session.permanent = bool(remember)


def _bump_epoch(user):
    """Log this account out everywhere (other browsers/phones)."""
    user["epoch"] = user.get("epoch", 0) + 1


def require_owner(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        me = _me()
        if not me or me.get("role") != "owner":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def _check_invite(code, use=True):
    """Is this a valid, unused, unexpired invite code? Uses it up if `use`."""
    invites = _read(INVITES_FILE)
    h = _hash_code(code.strip().upper())
    inv = invites.get(h)
    if not inv or inv.get("used") or inv.get("expires", 0) < time.time():
        return False
    if use:
        inv["used"] = int(time.time())
        _write(INVITES_FILE, invites)
    return True


def _account_page(status=200, **extra):
    me = _me()
    ctx = {"me": me, "me_key": current_user(), "is_owner": bool(me and me.get("role") == "owner"), "min_password": MIN_PASSWORD,
           "forced": bool(me and me.get("must_change_password")), "notice": None, "error": None,
           "setup": None, "recovery_codes": None, "new_invite": None, "temp_password": None,
           "users": [], "invites": [], "uploads": []}
    ctx.update(extra)
    ctx["usage"] = usage_for(current_user()) if me else None
    if ctx["is_owner"]:
        users = _load_users()
        ctx["users"] = sorted(({"key": k, **v, "usage": usage_for(k)} for k, v in users.items()),
                              key=lambda u: u.get("created", 0))
        import quota
        ctx["plans"] = quota.plans()
        now = time.time()
        names = {k: v["name"] for k, v in users.items()}
        ctx["uploads"] = [dict(u, owner_name=names.get(u["owner"], "(removed or before accounts)"))
                          for u in uploads_overview()]
        ctx["invites"] = sorted(
            ({"id": h, **v} for h, v in _read(INVITES_FILE).items() if not v.get("used") and v.get("expires", 0) > now),
            key=lambda i: i.get("created", 0), reverse=True)
    return render_template("account.html", **ctx), status


# ------------------------------------------------------------------- routes

def init_accounts(app):
    app.secret_key = _secret_key()
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,      # page scripts can't read the login cookie
        SESSION_COOKIE_SAMESITE="Lax",     # not sent with cross-site form posts
        PERMANENT_SESSION_LIFETIME=timedelta(days=REMEMBER_DAYS),
    )

    @app.context_processor
    def _inject_user():
        me = _me()
        return {"current_user": current_user() if me else None, "csrf_token": _csrf_token,
                "is_owner": bool(me and me.get("role") == "owner")}

    @app.template_global()
    def display_name():
        me = _me()
        return me["name"] if me else ""

    @app.before_request
    def _require_login():
        if request.endpoint in PUBLIC_ENDPOINTS:
            return None
        me = _me()
        if me:
            # Actions the page's JavaScript sends need the CSRF token in a header
            # (forms send it as a field). Stops other pages triggering them.
            if request.method == "POST" and request.path.startswith(API_PREFIXES):
                if not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), session.get("csrf", "") or "x"):
                    return jsonify(error="Your session expired. Reload the page and try again."), 403
            if me.get("must_change_password") and request.endpoint not in FORCED_CHANGE_OK:
                if request.path.startswith(API_PREFIXES):
                    return jsonify(error="Please change your password first.", login=url_for("account")), 401
                return redirect(url_for("account"))
            return None
        for k in ("user", "epoch"):
            session.pop(k, None)
        if request.path.startswith(API_PREFIXES):
            return jsonify(error="Please log in again.", login=url_for("login")), 401
        return redirect(url_for("login", next=request.full_path.rstrip("?")))

    # ------------------------------------------------------------- login
    @app.route("/login", methods=["GET", "POST"])
    def login():
        nxt = _safe_next(request.values.get("next")) or url_for("landing")
        has_users = bool(_load_users())

        def page(mode, error=None, username="", status=200):
            return render_template("login.html", mode=mode, next=nxt, error=error, username=username,
                                   need_invite=has_users, min_password=MIN_PASSWORD), status

        if request.method == "GET":
            if _me():
                return redirect(nxt)
            mode = "signup" if request.args.get("mode") == "signup" or not has_users else "login"
            return page(mode)

        mode = request.form.get("mode")
        if mode not in ("login", "signup", "2fa"):
            mode = "login"
        username = (request.form.get("username") or "").strip()
        key = username.lower()
        password = request.form.get("password") or ""
        if not _csrf_ok():
            return page("login" if mode == "2fa" else mode, "Your session expired. Please try again.", username, 400)

        # -- second step: 2FA code
        if mode == "2fa":
            pending = session.get("pending_2fa") or {}
            if not pending or time.time() - pending.get("at", 0) > PENDING_2FA_SECONDS:
                session.pop("pending_2fa", None)
                return page("login", "That took too long. Please log in again.")
            key = pending["user"]
            wait = code_user.wait(key)
            if wait:
                return page("2fa", f"Too many wrong codes. Try again in {wait} seconds.", status=429)
            code = request.form.get("code", "")
            with _users_lock:
                users = _load_users()
                user = users.get(key)
                if not user or "totp" not in user:
                    session.pop("pending_2fa", None)
                    return page("login", "Please log in again.")
                step = verify_totp(user["totp"]["secret"], code, user["totp"].get("last_step"))
                used_recovery = False
                if step is None:
                    h = _hash_code(normalise_recovery(code))
                    if h in user.get("recovery", []):
                        user["recovery"].remove(h)
                        used_recovery = True
                if step is None and not used_recovery:
                    code_user.hit(key)
                    return page("2fa", "That code isn't right. Check the time on your phone and try again.", status=401)
                if step is not None:
                    user["totp"]["last_step"] = step  # the same code can't be used twice
                _save_users(users)
            code_user.clear(key)
            login_user.clear(key)
            _start_session(key, user, pending.get("remember"))
            if used_recovery:
                session["notice"] = f"You used a recovery code. {len(user.get('recovery', []))} left."
            return redirect(pending.get("next") or nxt)

        # -- sign up
        if mode == "signup":
            wait = signup_ip.wait(_ip())
            if wait:
                return page(mode, f"Too many sign-up attempts. Try again in {wait // 60 + 1} minutes.", username, 429)
            signup_ip.hit(_ip())
            if not USERNAME_RE.match(username):
                return page(mode, "Pick a username of 3-32 letters, numbers, dots, dashes or underscores.", username, 400)
            if password != request.form.get("confirm", ""):
                return page(mode, "The two passwords don't match.", username, 400)
            if not request.form.get("terms"):
                return page(mode, "Please confirm you're 16 or older and accept the terms of use and privacy policy.", username, 400)
            problem = password_problem(password, username)
            if problem:
                return page(mode, problem, username, 400)
            with _users_lock:
                users = _load_users()
                first = not users
                # Invite first, so people without one can't find out which usernames exist.
                if not first and not _check_invite(request.form.get("invite", ""), use=False):
                    return page(mode, "That invite code isn't valid. Ask the owner of this app for a new one.", username, 400)
                if key in users:
                    return page(mode, "That username is taken. Log in instead, or pick another one.", username, 400)
                if not first:
                    _check_invite(request.form.get("invite", ""))  # use it up
                now = int(time.time())
                users[key] = {
                    "name": username,
                    "password": generate_password_hash(password),
                    "role": "owner" if first else "member",
                    "created": now,
                    "epoch": 0,
                    # Proof of what was accepted and when (GDPR accountability).
                    "terms_accepted": {"version": TERMS_VERSION, "at": now},
                }
                _save_users(users)
            _start_session(key, users[key], request.form.get("remember"))
            if first:
                session["notice"] = "You're the owner of this app. Turn on two-step login below, then invite your staff."
                return redirect(url_for("account"))
            return redirect(nxt)

        # -- log in (password step)
        wait = max(login_ip.wait(_ip()), login_user.wait(key))
        if wait:
            return page("login", f"Too many attempts. Try again in {wait} seconds.", username, 429)
        user = _load_users().get(key)
        # Always run a hash check, so a wrong username takes as long as a wrong password.
        ok = check_password_hash(user["password"] if user else _DUMMY_HASH, password) and user is not None
        if not ok:
            login_ip.hit(_ip())
            login_user.hit(key)
            return page("login", "Wrong username or password.", username, 401)
        if "totp" in user:
            session.clear()
            session["csrf"] = secrets.token_urlsafe(24)
            session["pending_2fa"] = {"user": key, "at": time.time(), "remember": bool(request.form.get("remember")),
                                      "next": nxt}
            return page("2fa")
        login_user.clear(key)
        _start_session(key, user, request.form.get("remember"))
        return redirect(nxt)

    @app.route("/logout", methods=["POST"])
    def logout():
        if _csrf_ok():
            session.clear()
        return redirect(url_for("login"))

    # ----------------------------------------------------------- account
    @app.route("/account")
    def account():
        return _account_page(notice=session.pop("notice", None))

    @app.route("/account/password", methods=["POST"])
    def change_password():
        me_key = current_user()
        if not _csrf_ok():
            return _account_page(400, error="Your session expired. Please try again.")
        wait = sensitive_user.wait(me_key)
        if wait:
            return _account_page(429, error=f"Too many attempts. Try again in {wait} seconds.")
        new = request.form.get("new", "")
        with _users_lock:
            users = _load_users()
            user = users[me_key]
            if not check_password_hash(user["password"], request.form.get("current", "")):
                sensitive_user.hit(me_key)
                return _account_page(400, error="Your current password isn't right.")
            if new != request.form.get("confirm", ""):
                return _account_page(400, error="The two new passwords don't match.")
            if check_password_hash(user["password"], new):
                return _account_page(400, error="Choose a password that's different from the current one.")
            problem = password_problem(new, user["name"])
            if problem:
                return _account_page(400, error=problem)
            user["password"] = generate_password_hash(new)
            user.pop("must_change_password", None)
            _bump_epoch(user)
            _save_users(users)
        _start_session(me_key, user, session.permanent)
        return _account_page(notice="Password changed. Other devices were logged out.")

    @app.route("/account/2fa/start", methods=["POST"])
    def twofa_start():
        if not _csrf_ok():
            return _account_page(400, error="Your session expired. Please try again.")
        secret = new_totp_secret()
        session["totp_setup"] = secret
        uri = totp_uri(secret, _me()["name"])
        return _account_page(setup={"secret": secret, "uri": uri, "qr": totp_qr_svg(uri)})

    @app.route("/account/2fa/confirm", methods=["POST"])
    def twofa_confirm():
        me_key = current_user()
        secret = session.get("totp_setup")
        if not _csrf_ok() or not secret:
            return _account_page(400, error="Please start the set-up again.")
        wait = code_user.wait(me_key)
        if wait:
            return _account_page(429, error=f"Too many wrong codes. Try again in {wait} seconds.")
        step = verify_totp(secret, request.form.get("code", ""))
        if step is None:
            code_user.hit(me_key)
            uri = totp_uri(secret, _me()["name"])
            return _account_page(400, error="That code isn't right. Make sure your phone's time is set automatically.",
                                 setup={"secret": secret, "uri": uri, "qr": totp_qr_svg(uri)})
        codes = new_recovery_codes()
        with _users_lock:
            users = _load_users()
            users[me_key]["totp"] = {"secret": secret, "last_step": step, "since": int(time.time())}
            users[me_key]["recovery"] = [_hash_code(normalise_recovery(c)) for c in codes]
            _save_users(users)
        session.pop("totp_setup", None)
        return _account_page(notice="Two-step login is on.", recovery_codes=codes)

    @app.route("/account/2fa/disable", methods=["POST"])
    def twofa_disable():
        me_key = current_user()
        if not _csrf_ok():
            return _account_page(400, error="Your session expired. Please try again.")
        wait = sensitive_user.wait(me_key)
        if wait:
            return _account_page(429, error=f"Too many attempts. Try again in {wait} seconds.")
        with _users_lock:
            users = _load_users()
            user = users[me_key]
            ok_pw = check_password_hash(user["password"], request.form.get("password", ""))
            ok_code = "totp" in user and verify_totp(user["totp"]["secret"], request.form.get("code", "")) is not None
            if not (ok_pw and ok_code):
                sensitive_user.hit(me_key)
                return _account_page(400, error="Your password or code isn't right, so two-step login stays on.")
            user.pop("totp", None)
            user.pop("recovery", None)
            _bump_epoch(user)
            _save_users(users)
        _start_session(me_key, user, session.permanent)
        return _account_page(notice="Two-step login is off. Other devices were logged out.")

    @app.route("/account/delete", methods=["POST"])
    def delete_account():
        """Right to erasure: remove the logged-in user's account after a password check."""
        back = url_for("legal_page", doc="privacy")
        if not _csrf_ok():
            return redirect(back + "?error=expired#delete")
        key = current_user()
        if sensitive_user.wait(key):
            return redirect(back + "?error=wait#delete")
        with _users_lock:
            users = _load_users()
            user = users.get(key)
            if not user or not check_password_hash(user["password"], request.form.get("password", "")):
                sensitive_user.hit(key)
                return redirect(back + "?error=password#delete")
            if user.get("role") == "owner" and len(users) > 1:
                return redirect(back + "?error=owner#delete")
            del users[key]
            _save_users(users)
        delete_user_videos(key)
        session.clear()
        return redirect(back + "?deleted=1#delete")

    # ------------------------------------------------------------- owner
    @app.route("/admin/invite", methods=["POST"])
    @require_owner
    def admin_invite():
        if not _csrf_ok():
            return _account_page(400, error="Your session expired. Please try again.")
        if admin_actions.wait(current_user()):
            return _account_page(429, error="Too many changes in a short time. Try again later.")
        admin_actions.hit(current_user())
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        code = "-".join("".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(3))
        invites = _read(INVITES_FILE)
        now = int(time.time())
        invites[_hash_code(code)] = {"created": now, "expires": now + INVITE_DAYS * 86400, "by": current_user(),
                                     "note": (request.form.get("note") or "").strip()[:40]}
        _write(INVITES_FILE, invites)
        return _account_page(new_invite=code)

    @app.route("/admin/invite/revoke", methods=["POST"])
    @require_owner
    def admin_invite_revoke():
        if not _csrf_ok():
            return _account_page(400, error="Your session expired. Please try again.")
        invites = _read(INVITES_FILE)
        invites.pop(request.form.get("id", ""), None)
        _write(INVITES_FILE, invites)
        return _account_page(notice="Invite cancelled.")

    @app.route("/admin/user/reset", methods=["POST"])
    @require_owner
    def admin_reset():
        """For someone locked out: give them a temporary password (they must
        change it at their next login) and turn off their two-step login."""
        if not _csrf_ok():
            return _account_page(400, error="Your session expired. Please try again.")
        if admin_actions.wait(current_user()):
            return _account_page(429, error="Too many changes in a short time. Try again later.")
        admin_actions.hit(current_user())
        target = request.form.get("user", "")
        if target == current_user():
            return _account_page(400, error="Use Change password for your own account.")
        temp = secrets.token_urlsafe(9)
        with _users_lock:
            users = _load_users()
            if target not in users:
                return _account_page(400, error="That account doesn't exist.")
            users[target]["password"] = generate_password_hash(temp)
            users[target]["must_change_password"] = True
            users[target].pop("totp", None)
            users[target].pop("recovery", None)
            _bump_epoch(users[target])
            _save_users(users)
        login_user.clear(target)
        return _account_page(temp_password={"name": users[target]["name"], "password": temp})

    @app.route("/admin/user/remove", methods=["POST"])
    @require_owner
    def admin_remove():
        if not _csrf_ok():
            return _account_page(400, error="Your session expired. Please try again.")
        target = request.form.get("user", "")
        if target == current_user():
            return _account_page(400, error="You can't remove your own owner account here.")
        with _users_lock:
            users = _load_users()
            removed = users.pop(target, None)
            _save_users(users)
        if not removed:
            return _account_page(400, error="That account doesn't exist.")
        delete_user_videos(target)
        return _account_page(notice=f"Removed {removed['name']} and their videos.")

    @app.route("/admin/user/plan", methods=["POST"])
    @require_owner
    def admin_set_plan():
        """The owner puts an account on a plan (Free, Plus, Pro...)."""
        import quota
        if not _csrf_ok():
            return _account_page(400, error="Your session expired. Please try again.")
        target, plan = request.form.get("user", ""), request.form.get("plan", "")
        if plan not in quota.plans():
            return _account_page(400, error="Unknown plan.")
        with _users_lock:
            users = _load_users()
            if target not in users:
                return _account_page(400, error="That account doesn't exist.")
            users[target]["plan"] = plan
            _save_users(users)
        return _account_page(notice=f"{users[target]['name']} is now on the {quota.plans()[plan].get('name', plan)} plan.")

    @app.route("/admin/video/delete", methods=["POST"])
    @require_owner
    def admin_delete_video():
        """Moderation: the owner removes any upload."""
        if not _csrf_ok():
            return _account_page(400, error="Your session expired. Please try again.")
        if not delete_video(request.form.get("id", "")):
            return _account_page(400, error="That video doesn't exist any more.")
        return _account_page(notice="Video deleted.")


# A real hash of a random password, used when the username doesn't exist.
_DUMMY_HASH = generate_password_hash(secrets.token_hex(16))
