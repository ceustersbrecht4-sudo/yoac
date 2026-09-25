"""
Legal pages (privacy, cookies, terms, legal notice, licences) and the source
code download.

The owner's details come from legal_info.json, which the owner fills in with
Notepad. Anything still in [square brackets] is shown in orange on the pages,
with a notice that the documents aren't finished.

/source serves the app's own code as a zip. The tracking engine (Ultralytics
YOLO) is AGPL-3.0, which requires offering the complete source to everyone
who uses the app over a network. Accounts, secrets and videos are never
included.
"""

import io
import json
import os
import zipfile

from flask import abort, render_template, request, send_file

import accounts

HERE = os.path.dirname(os.path.abspath(__file__))
INFO_FILE = os.path.join(HERE, "legal_info.json")

TITLES = {
    "privacy": "Privacy policy",
    "cookies": "Cookie policy",
    "terms": "Terms of use",
    "notice": "Legal notice",
    "licences": "Licences",
}
DEFAULTS = {"country": "Belgium", "last_updated": "", "video_retention_days": 90}

# What goes into the source zip: code, pages, logos, fonts, docs.
SOURCE_FILES = ("*.py", "*.md", "*.txt", "*.bat", "legal_info.json")
SOURCE_DIRS = ("templates", "static")
NEVER_SHARE = {"users.json", "users.json.tmp", "secret.key"}


def load_info():
    info = dict(DEFAULTS)
    try:
        with open(INFO_FILE, encoding="utf-8") as f:
            info.update(json.load(f))
    except (OSError, ValueError):
        pass
    return info


def retention_days():
    try:
        return max(0, int(load_info().get("video_retention_days") or 0))
    except (TypeError, ValueError):
        return 0


def _source_zip():
    import glob
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        paths = [p for pattern in SOURCE_FILES for p in glob.glob(os.path.join(HERE, pattern))]
        for d in SOURCE_DIRS:
            for root, _, files in os.walk(os.path.join(HERE, d)):
                paths += [os.path.join(root, f) for f in files]
        for path in sorted(set(paths)):
            name = os.path.relpath(path, HERE)
            if os.path.basename(name) in NEVER_SHARE or "__pycache__" in name or name.startswith("static" + os.sep + "demo"):
                continue
            z.write(path, os.path.join("yoac-player-tracker", name))
    buf.seek(0)
    return buf


def init_legal(app):
    accounts.PUBLIC_ENDPOINTS.update({"legal_page", "source_code"})

    @app.route("/legal/<doc>")
    def legal_page(doc):
        if doc not in TITLES:
            abort(404)
        info = load_info()
        missing = any(isinstance(v, str) and v.startswith("[") for k, v in info.items() if not k.startswith("_"))
        errors = {"password": "That password isn't right, so the account was not deleted.",
                  "expired": "Your session expired. Please try again.",
                  "wait": "Too many attempts. Please wait a few minutes and try again.",
                  "owner": "You're the owner. Remove the other accounts on your account page first, "
                           "so nobody is left without an owner."}
        return render_template("legal.html", doc=doc, titles=TITLES, info=info, missing=missing,
                               deleted=request.args.get("deleted") == "1",
                               delete_error=errors.get(request.args.get("error")))

    @app.route("/source")
    def source_code():
        return send_file(_source_zip(), mimetype="application/zip", as_attachment=True,
                         download_name="yoac-player-tracker-source.zip")

    @app.context_processor
    def _inject_legal():
        return {"legal_titles": TITLES}
