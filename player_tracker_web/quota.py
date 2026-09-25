"""
Usage limits per account: storage (GB kept on the laptop) and video minutes
analysed per calendar month.

Plans come from plans.json (edit it with Notepad); these defaults are used
for anything it doesn't set. The owner picks each account's plan on the
Account page. The owner's own account has no limits. Paid plans can be
connected to online payments later; for now the owner assigns them.
"""

import json
import os
import time

import accounts

HERE = os.path.dirname(os.path.abspath(__file__))
PLANS_FILE = os.path.join(HERE, "plans.json")
GB = 1024 ** 3

DEFAULT_PLANS = {
    "free": {"name": "Free", "storage_gb": 5, "minutes_per_month": 120, "price": "Free"},
    "plus": {"name": "Plus", "storage_gb": 25, "minutes_per_month": 600, "price": "Coming soon"},
    "pro": {"name": "Pro", "storage_gb": 100, "minutes_per_month": 3000, "price": "Coming soon"},
}


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


def summary(user_key, storage_bytes):
    """Everything the pages show about one account's usage."""
    user = accounts._load_users().get(user_key) or {}
    all_plans = plans()
    plan_key = user.get("plan") if user.get("plan") in all_plans else "free"
    plan = all_plans[plan_key]
    unlimited = user.get("role") == "owner"
    storage_limit = float(plan.get("storage_gb", 5)) * GB
    minutes_limit = float(plan.get("minutes_per_month", 120))
    used_min = minutes_used(user)
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
    }


def storage_problem(usage, incoming_bytes):
    """A message if storing `incoming_bytes` more would pass the limit, else None."""
    if usage["unlimited"] or usage["storage_used"] + incoming_bytes <= usage["storage_limit"]:
        return None
    return (f"This video doesn't fit in your storage: you're using {usage['storage_gb_used']} of "
            f"{usage['storage_gb_limit']} GB on the {usage['plan']} plan. Delete old videos, or ask the owner "
            f"of this app about a bigger plan.")


def minutes_problem(usage, seconds):
    user = accounts._load_users().get(accounts.current_user()) or {}
    if usage["unlimited"] or minutes_used(user) + seconds / 60 <= usage["minutes_limit"]:
        return None
    left = max(0, usage["minutes_limit"] - usage["minutes_used"])
    return (f"This video is {seconds / 60:.0f} minutes, but you have {left} of your {usage['minutes_limit']} "
            f"analysis minutes left this month on the {usage['plan']} plan. Trim the video, wait until next "
            f"month, or ask the owner of this app about a bigger plan.")
