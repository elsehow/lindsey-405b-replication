#!/usr/bin/env python3
"""Poll RunPod every 60s for any 405B-fitting GPU config (8x H100 SXM, 4x H200, 4x B200)
across all DCs. Fires a Mac notification + log line whenever a slot OPENS (edge-triggered —
no spam if slot stays available). Nick deploys manually via the RunPod console.

Logs to scripts/notify_when_available.log.
"""
import json, os, subprocess, sys, time
from pathlib import Path

REPO = Path("/Users/elsehow/Projects/lindsey-405b-replication")
SCRIPTS = REPO / "scripts"
LOG = SCRIPTS / "notify_when_available.log"

# DCs where Nick already has weights on a network volume — preferred (no fresh download needed)
EXISTING_VOLUMES = {
    "AP-JP-1": "mlnj4f09jl",
    "US-GA-2": "3inqb44opz",
}

PROBE_CONFIGS = [
    {"gpu_id": "NVIDIA H100 80GB HBM3", "count": 8, "max_price": 24.0, "label": "8x H100 SXM"},
    {"gpu_id": "NVIDIA H200",            "count": 4, "max_price": 18.0, "label": "4x H200"},
    {"gpu_id": "NVIDIA B200",            "count": 4, "max_price": 26.0, "label": "4x B200"},
]


def read_env(path, key):
    try:
        for line in open(path):
            line = line.strip()
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        return None
    return None


RUNPOD_KEY = read_env(REPO / ".env", "RUNPOD_API_KEY")
if not RUNPOD_KEY:
    sys.exit("Missing RUNPOD_API_KEY in .env")


def gql(q):
    r = subprocess.run(
        ["curl", "-sS", "-X", "POST", "https://api.runpod.io/graphql",
         "-H", "Content-Type: application/json",
         "-H", "Authorization: Bearer " + RUNPOD_KEY,
         "--data-binary", "@-"],
        input=json.dumps({"query": q}), capture_output=True, text=True, timeout=30,
    )
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {}


def log(msg):
    with open(LOG, "a") as f:
        f.write("[{}] {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def notify(title, msg):
    log("NOTIFY: " + msg)
    safe_msg = msg.replace('"', "'")
    safe_title = title.replace('"', "'")
    subprocess.run(["osascript", "-e",
                    'display notification "{}" with title "{}" sound name "Glass"'.format(safe_msg, safe_title)],
                   capture_output=True)
    # Also speak it briefly
    subprocess.run(["say", "-v", "Samantha", title.replace("(","").replace(")","")[:60]], capture_output=True)


def find_available():
    """Return list of (gpu_id, count, dc, price, label, has_volume) for all currently-bookable fits."""
    out = []
    for cfg in PROBE_CONFIGS:
        # First narrow to DCs where this config is conceivably available
        avail_q = ('{dataCenters{id gpuAvailability(input:{gpuCount:' + str(cfg["count"]) +
                   ',minMemoryInGb:80,minVcpuCount:0}){gpuTypeId available}}}')
        r = gql(avail_q)
        candidate_dcs = []
        for dc in r.get("data", {}).get("dataCenters", []) or []:
            for g in dc.get("gpuAvailability", []) or []:
                if g.get("available") and g.get("gpuTypeId") == cfg["gpu_id"]:
                    candidate_dcs.append(dc["id"])
        # Confirm bookable + check price
        for dc in candidate_dcs:
            r = gql('{gpuTypes(input:{id:"' + cfg["gpu_id"] +
                    '"}){lowestPrice(input:{gpuCount:' + str(cfg["count"]) +
                    ',dataCenterId:"' + dc + '"}){uninterruptablePrice}}}')
            try:
                p = r["data"]["gpuTypes"][0]["lowestPrice"] or {}
            except (KeyError, IndexError, TypeError):
                continue
            price = p.get("uninterruptablePrice")
            if price and price <= cfg["max_price"]:
                out.append({
                    "gpu_id": cfg["gpu_id"],
                    "count": cfg["count"],
                    "dc": dc,
                    "price": price,
                    "label": cfg["label"],
                    "has_volume": dc in EXISTING_VOLUMES,
                })
    return out


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== notify_when_available.py started ===")
    log("Probing: " + ", ".join(c["label"] for c in PROBE_CONFIGS))
    log("Preferred DCs (existing weights): " + ", ".join(EXISTING_VOLUMES.keys()))
    notify("405B poller started", "Will alert when any 405B-fit slot opens")

    seen_keys = set()  # tracks (gpu_id, count, dc) currently available — for edge-triggered notify

    while True:
        try:
            available = find_available()
            current_keys = {(a["gpu_id"], a["count"], a["dc"]) for a in available}
            new_slots = [a for a in available if (a["gpu_id"], a["count"], a["dc"]) not in seen_keys]

            for slot in new_slots:
                marker = "✓existing volume" if slot["has_volume"] else "fresh download needed"
                msg = "{} in {} at ${}/hr ({})".format(
                    slot["label"], slot["dc"], slot["price"], marker)
                title = "RunPod slot OPEN: " + slot["label"]
                log("SLOT OPEN: " + msg)
                notify(title, msg)

            # Update tracking — drop slots that disappeared (so they re-notify when they reappear)
            seen_keys = current_keys

            if not available:
                log("no slot")
            time.sleep(60)
        except KeyboardInterrupt:
            log("interrupted; exiting")
            sys.exit(0)
        except Exception as e:
            log("poll error: " + repr(e))
            time.sleep(60)


if __name__ == "__main__":
    main()
