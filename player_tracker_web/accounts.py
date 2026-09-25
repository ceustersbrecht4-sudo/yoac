"""
Log in / create account for the local web app.

Accounts live in users.json next to this file - nothing leaves the laptop.
Passwords are stored as salted hashes (werkzeug), never as plain text. The
session is signed with a random key kept in secret.key, so logins survive a
restart of the app.

Every page and API route needs a logged-in user except the login page itself
and static files. Pages redirect to the login page; API calls get a 401 so
the upload page can send the user there.
"""

import json
import os
import re
import secrets
import threading
import time
from datetime import timedelta
from urllib.parse import urlparse

from flask import jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

HERE = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(HERE, "users.json")
SECRET_FILE = os.path.join(HERE, "secret.key")

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 8
REMEMBER_DAYS = 30
TERMS_VERSION = "2026-09-25"  # bump when the terms or privacy policy change
MAX_FAILS = 5            # failed logins from one address...
FAIL_WINDOW = 5 * 60     # ...within this many seconds...
LOCKOUT = 60             # ...lock that address out for this long

# Pages and files anyone can open without logging in.
PUBLIC_ENDPOINTS = {"login", "static"}
# Paths the upload page calls from JavaScript: answer with JSON, not a redirect.
API_PREFIXES = ("/analyze", "/status/", "/hide/", "/delete/", "/report/", "/snapshot/", "/frame/", "/video/")

_users_lock = threading.Lock()  # one sign-up/login at a time touches users.json
_fails_lock = threading.Lock()
_fails = {}  # ip -> [timestamps of recent failed logins]


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


def _load_users():
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_users(users):
    tmp = USERS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(users, f, indent=2)
    os.replace(tmp, USERS_FILE)


def _safe_next(target):
    """Only follow redirects back into this app (no //evil.com or https://...)."""
    if not target:
        return None
    parts = urlparse(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return None
    return target


def _locked_out(ip):
    now = time.time()
    with _fails_lock:
        recent = [t for t in _fails.get(ip, []) if now - t < FAIL_WINDOW]
        _fails[ip] = recent
        if len(recent) >= MAX_FAILS and now - recent[-1] < LOCKOUT:
            return int(LOCKOUT - (now - recent[-1])) + 1
    return 0


def _record_fail(ip):
    with _fails_lock:
        _fails.setdefault(ip, []).append(time.time())


def _csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(24)
    return session["csrf"]


def current_user():
    return session.get("user")


def init_accounts(app):
    app.secret_key = _secret_key()
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=REMEMBER_DAYS),
    )

    @app.context_processor
    def _inject_user():
        return {"current_user": current_user(), "csrf_token": _csrf_token}

    @app.before_request
    def _require_login():
        if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None and request.path.startswith("/static/"):
            return None
        if current_user() and current_user() in _load_users():
            return None
        session.pop("user", None)
        if request.path.startswith(API_PREFIXES):
            return jsonify(error="Please log in again.", login=url_for("login")), 401
        return redirect(url_for("login", next=request.full_path.rstrip("?")))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        nxt = _safe_next(request.values.get("next")) or url_for("landing")
        if request.method == "GET":
            if current_user() and current_user() in _load_users():
                return redirect(nxt)
            mode = "signup" if request.args.get("mode") == "signup" or not _load_users() else "login"
            return render_template("login.html", mode=mode, next=nxt, error=None, username="")

        mode = "signup" if request.form.get("mode") == "signup" else "login"
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        def fail(msg, status=400):
            return render_template("login.html", mode=mode, next=nxt, error=msg, username=username), status

        if not secrets.compare_digest(request.form.get("csrf", ""), session.get("csrf", "")):
            return fail("Your session expired. Please try again.")
        ip = request.remote_addr or "?"
        wait = _locked_out(ip)
        if wait:
            return fail(f"Too many attempts. Try again in {wait} seconds.", 429)

        with _users_lock:
            users = _load_users()
            key = username.lower()
            if mode == "signup":
                if not USERNAME_RE.match(username):
                    return fail("Pick a username of 3-32 letters, numbers, dots, dashes or underscores.")
                if key in users:
                    return fail("That username is taken. Log in instead, or pick another one.")
                if len(password) < MIN_PASSWORD:
                    return fail(f"Use a password of at least {MIN_PASSWORD} characters.")
                if password != request.form.get("confirm", ""):
                    return fail("The two passwords don't match.")
                if not request.form.get("terms"):
                    return fail("Please confirm you're 16 or older and accept the terms of use and privacy policy.")
                now = int(time.time())
                users[key] = {
                    "name": username,
                    "password": generate_password_hash(password),
                    "created": now,
                    # Proof of what was accepted and when (GDPR accountability).
                    "terms_accepted": {"version": TERMS_VERSION, "at": now},
                }
                _save_users(users)
            else:
                user = users.get(key)
                if not user or not check_password_hash(user["password"], password):
                    _record_fail(ip)
                    return fail("Wrong username or password.", 401)

        session.clear()
        session["user"] = key
        session.permanent = bool(request.form.get("remember"))
        return redirect(nxt)

    @app.route("/account/delete", methods=["POST"])
    def delete_account():
        """Right to erasure: remove the logged-in user's account after a password check."""
        back = url_for("legal_page", doc="privacy")
        if not secrets.compare_digest(request.form.get("csrf", ""), session.get("csrf", "")):
            return redirect(back + "?error=expired#delete")
        key = current_user()
        with _users_lock:
            users = _load_users()
            user = users.get(key)
            if not user or not check_password_hash(user["password"], request.form.get("password", "")):
                return redirect(back + "?error=password#delete")
            del users[key]
            _save_users(users)
        session.clear()
        return redirect(back + "?deleted=1#delete")

    @app.route("/logout", methods=["POST"])
    def logout():
        if secrets.compare_digest(request.form.get("csrf", ""), session.get("csrf", "")):
            session.clear()
        return redirect(url_for("login"))

    @app.template_global()
    def display_name():
        user = _load_users().get(current_user() or "")
        return user["name"] if user else ""
