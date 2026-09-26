"""
Usage limits per account: storage (GB of videos kept), video minutes analysed
per calendar month, and GB sent back (streaming + downloads) per calendar
month. Once the app is online, those three are what cost money: disk space,
processing time and outgoing traffic.

Plans come from plans.json (edit it with Notepad); these defaults are used
for anything it doesn't set. The owner picks each account's plan on the
Account page. The owner's own account has no limits. Paid plans can be
connected to online payments later; for now the owner assigns them.
"""

import json
import os
import threading
import time

import accounts

HERE = os.path.dirname(os.path.abspath(__file__))
PLANS_FILE = os.path.join(HERE, "plans.json")
GB = 1024 ** 3

DEFAULT_PLANS = {
    "free": {"name": "Free", "storage_gb": 2, "minutes_per_month": 120, "transfer_gb_per_month": 10,
             "keep_days": 30, "price": "Free"},
    "plus": {"name": "Plus", "storage_gb": 25, "minutes_per_month": 600, "transfer_gb_per_month": 100,
             "keep_days": 180, "price": "Coming soon"},
    "pro": {"name": "Pro", "storage_gb": 100, "minutes_per_month": 3000, "transfer_gb_per_month": 400,
            "keep_days": 365, "price": "Coming soon"},
}
# Limits for the whole server, whatever the plans add up to (plans.json "_server").
DEFAULT_SERVER = {"max_total_storage_gb": 0, "min_free_disk_gb": 5}


def plans():
    merged = {k: dict(v) for k, v in DEFAULT_PLANS.items()}
    try:
        with open(PLANS_FILE, encoding="utf-8") as f:
            for key, plan in json.load(f).items():
                if isinstance(plan, dict) and not key.startswith("_"):
                    merged.setdefault(key, {}).update(plan)
    except (OSError, ValueError):
        pass
    return merged


def server_limits():
    limits = dict(DEFAULT_SERVER)
    try:
        with open(PLANS_FILE, encoding="utf-8") as f:
            extra = json.load(f).get("_server")
        if isinstance(extra, dict):
            limits.update({k: v for k, v in extra.items() if k in DEFAULT_SERVER})
    except (OSError, ValueError, AttributeError):
        pass
    for k in limits:
        try:
            limits[k] = max(0.0, float(limits[k]))
        except (TypeError, ValueError):
            limits[k] = float(DEFAULT_SERVER[k])
    return limits


def _num(plan, key, default):
    try:
        return max(0.0, float(plan.get(key, default)))
    except (TypeError, ValueError):
        return float(default)


def _month():
    return time.strftime("%Y-%m")


def minutes_used(user):
    usage = (user or {}).get("usage") or {}
    return usage.get("seconds", 0) / 60 if usage.get("month") == _month() else 0.0


def record_seconds(user_key, seconds):
    """Add (or, with a negative number, give back) analysed video time this month."""
    with accounts._users_lock:
        users = accounts._load_users()
        user = users.get(user_key)
        if not user:
            return
        usage = user.get("usage") or {}
        if usage.get("month") != _month():
            usage = {"month": _month(), "seconds": 0}
        usage["seconds"] = max(0, usage.get("seconds", 0) + seconds)
        user["usage"] = usage
        accounts._save_users(users)


# Bytes sent to each account this month, counted in memory and written to
# users.json once a minute (a video stream is many small requests).
_sent = {}
_sent_lock = threading.Lock()


def record_sent(user_key, nbytes):
    if user_key and nbytes > 0:
        with _sent_lock:
            _sent[user_key] = _sent.get(user_key, 0) + nbytes


def _pending_sent(user_key):
    with _sent_lock:
        return _sent.get(user_key, 0)


def flush_sent():
    with _sent_lock:
        pending = dict(_sent)
        _sent.clear()
    if not pending:
        return
    with accounts._users_lock:
        users = accounts._load_users()
        for key, nbytes in pending.items():
            user = users.get(key)
            if not user:
                continue
            usage = user.get("usage") or {}
            if usage.get("month") != _month():
                usage = {"month": _month(), "seconds": 0}
            usage["sent"] = usage.get("sent", 0) + nbytes
            user["usage"] = usage
        accounts._save_users(users)


def bytes_sent(user_key, user=None):
    if user is None:
        user = accounts._load_users().get(user_key) or {}
    usage = user.get("usage") or {}
    saved = usage.get("sent", 0) if usage.get("month") == _month() else 0
    return saved + _pending_sent(user_key)


def plan_for(user):
    all_plans = plans()
    key = (user or {}).get("plan")
    key = key if key in all_plans else "free"
    return key, all_plans[key]


def summary(user_key, storage_bytes):
    """Everything the pages show about one account's usage."""
    user = accounts._load_users().get(user_key) or {}
    plan_key, plan = plan_for(user)
    unlimited = user.get("role") == "owner"
    storage_limit = _num(plan, "storage_gb", 2) * GB
    minutes_limit = _num(plan, "minutes_per_month", 120)
    transfer_limit = _num(plan, "transfer_gb_per_month", 10) * GB
    used_min = minutes_used(user)
    sent = bytes_sent(user_key, user)
    return {
        "plan_key": plan_key,
        "plan": "Owner (no limits)" if unlimited else plan.get("name", plan_key.title()),
        "unlimited": unlimited,
        "storage_used": storage_bytes,
        "storage_limit": storage_limit,
        "storage_gb_used": round(storage_bytes / GB, 2),
        "storage_gb_limit": round(storage_limit / GB, 1),
        "storage_pct": 0 if unlimited else min(100, round(100 * storage_bytes / storage_limit)) if storage_limit else 100,
        "minutes_used": round(used_min),
        "minutes_limit": int(minutes_limit),
        "minutes_pct": 0 if unlimited else min(100, round(100 * used_min / minutes_limit)) if minutes_limit else 100,
        "transfer_used": sent,
        "transfer_limit": transfer_limit,
        "transfer_gb_used": round(sent / GB, 2),
        "transfer_gb_limit": round(transfer_limit / GB, 1),
        "transfer_pct": 0 if unlimited else min(100, round(100 * sent / transfer_limit)) if transfer_limit else 100,
        "keep_days": 0 if unlimited else int(_num(plan, "keep_days", 0)),
    }


def storage_problem(usage, incoming_bytes):
    """A message if storing `incoming_bytes` more would pass the limit, else None."""
    if usage["unlimited"] or usage["storage_used"] + incoming_bytes <= usage["storage_limit"]:
        return None
    return (f"This video doesn't fit in your storage: you're using {usage['storage_gb_used']} of "
            f"{usage['storage_gb_limit']} GB on the {usage['plan']} plan (while it's being prepared, the upload "
            f"needs as much room as the file itself). Delete old videos, or ask the owner "
            f"of this app about a bigger plan.")


def minutes_problem(usage, seconds):
    user = accounts._load_users().get(accounts.current_user()) or {}
    if usage["unlimited"] or minutes_used(user) + seconds / 60 <= usage["minutes_limit"]:
        return None
    left = max(0, usage["minutes_limit"] - usage["minutes_used"])
    return (f"This video is {seconds / 60:.0f} minutes, but you have {left} of your {usage['minutes_limit']} "
            f"analysis minutes left this month on the {usage['plan']} plan. Trim the video, wait until next "
            f"month, or ask the owner of this app about a bigger plan.")


def transfer_problem(usage):
    """A message if this account has used up this month's streaming/downloads."""
    if usage["unlimited"] or usage["transfer_used"] < usage["transfer_limit"]:
        return None
    return (f"You've used this month's {usage['transfer_gb_limit']} GB of video streaming and downloads on the "
            f"{usage['plan']} plan. Your videos and reports are kept; watching and downloading videos works again "
            f"on the 1st of next month, or ask the owner of this app about a bigger plan.")


def server_problem(incoming_bytes, total_stored_bytes, free_disk_bytes):
    """A message if the server as a whole has no room for this upload."""
    limits = server_limits()
    if limits["min_free_disk_gb"] and free_disk_bytes - incoming_bytes < limits["min_free_disk_gb"] * GB:
        return "The server is almost out of space, so uploads are paused. Please try again later."
    if limits["max_total_storage_gb"] and total_stored_bytes + incoming_bytes > limits["max_total_storage_gb"] * GB:
        return "The server has reached its storage limit, so uploads are paused. Please try again later."
    return None


def keep_days_for(user_key):
    """How many days this account's videos are kept (0 = no plan limit)."""
    user = accounts._load_users().get(user_key) or {}
    if user.get("role") == "owner":
        return 0
    return int(_num(plan_for(user)[1], "keep_days", 0))
