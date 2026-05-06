#!/usr/bin/env python3
"""Auto-poll RunPod globally for any 405B-fitting GPU config, deploy + run dense-mag sweep + judge + pull + teardown.

Strategy (per Nick's instruction): poll EVERY DC for 8x H100 SXM, 4x H200, 4x B200. Prefer DCs where we already have a volume with weights (US-GA-2, AP-JP-1) — saves the ~6 min HF download. Fall back to any DC, fresh download into container disk.

Notifies via Mac notifications on key transitions.

Logs to scripts/auto_sweep.log; results dropped in scripts/results-<timestamp>/.
"""
import json, os, subprocess, sys, time, socket, traceback
from pathlib import Path

REPO = Path("/Users/elsehow/Projects/lindsey-405b-replication")
SCRIPTS = REPO / "scripts"
SSH_KEY = "/Users/elsehow/.ssh/id_ed25519"
LOG = SCRIPTS / "auto_sweep.log"

# DCs where we already have a volume holding the FP8 weights — prefer these (no download needed)
# Both volumes deleted 2026-05-05 to avoid ongoing storage cost. Every deploy is now fresh-download
# (~6 min on a US/EU pod). Re-add entries here if you re-create persistent volumes.
EXISTING_VOLUMES = {}

# DCs to skip — host driver too old for the runpod/pytorch:1.0.3-cu1290-torch280 image
# (torch 2.8 wants newer than CUDA driver 12090). Burned ~$10 / 3 runs at US-NC-1 on 2026-05-06
# before we pinned this down. Remove from blacklist if RunPod refreshes drivers there.
DC_BLACKLIST = {"US-NC-1"}

# Configs that fit Llama-3.1-405B-Instruct FP8 (~410 GB)
#   8x H100 SXM (640 GB), 4x H200 (564 GB), 4x B200 (720 GB)
PROBE_CONFIGS = [
    {"gpu_id": "NVIDIA H100 80GB HBM3", "count": 8, "max_price": 24.0},
    {"gpu_id": "NVIDIA H200",            "count": 4, "max_price": 18.0},
    {"gpu_id": "NVIDIA H200",            "count": 8, "max_price": 30.0},
    {"gpu_id": "NVIDIA B200",            "count": 4, "max_price": 26.0},
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
ANTHROPIC_KEY = (
    read_env(Path.home() / ".env", "ANTHROPIC_API_KEY")
    or read_env(REPO / ".env", "ANTHROPIC_API_KEY")
    or read_env(Path.home() / "Projects" / "interpretability" / ".env", "ANTHROPIC_API_KEY")
    or os.environ.get("ANTHROPIC_API_KEY")
)
HF_TOKEN = (
    read_env(Path.home() / ".env", "HUGGING_FACE_API_KEY")
    or read_env(Path.home() / "Projects" / "interpretability" / ".env", "HUGGING_FACE_API_KEY")
)

if not RUNPOD_KEY:
    sys.exit("Missing RUNPOD_API_KEY in .env")
if not ANTHROPIC_KEY:
    sys.exit("Missing ANTHROPIC_API_KEY (need it for the judge)")


def gql(query, variables=None, timeout=30):
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    r = subprocess.run(
        ["curl", "-sS", "-X", "POST", "https://api.runpod.io/graphql",
         "-H", "Content-Type: application/json",
         "-H", "Authorization: Bearer " + RUNPOD_KEY,
         "--data-binary", "@-"],
        input=json.dumps(payload), capture_output=True, text=True, timeout=timeout,
    )
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"_raw": r.stdout, "_err": r.stderr}


PUBKEY = gql("query{myself{pubKey}}")["data"]["myself"]["pubKey"]


def log(msg):
    line = "[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def notify(msg):
    log("NOTIFY: " + msg)
    safe = msg.replace('"', "'")
    subprocess.run(["osascript", "-e",
                    'display notification "{}" with title "lindsey-sweep"'.format(safe)],
                   capture_output=True)


def poll_for_slot():
    """Returns the cheapest fits-405B option currently bookable, preferring DCs where we have volumes.
    Returns dict with {gpu_id, count, dc, price, volume_id (None if fresh-download)} or None."""
    candidates = []
    for cfg in PROBE_CONFIGS:
        # gpuAvailability: which DCs *could* host this config
        avail_q = ('{dataCenters{id gpuAvailability(input:{gpuCount:' + str(cfg["count"]) +
                   ',minMemoryInGb:80,minVcpuCount:0}){gpuTypeId available}}}')
        r = gql(avail_q)
        candidate_dcs = []
        for dc in r.get("data", {}).get("dataCenters", []) or []:
            if dc["id"] in DC_BLACKLIST:
                continue
            for g in dc.get("gpuAvailability", []) or []:
                if g.get("available") and g.get("gpuTypeId") == cfg["gpu_id"]:
                    candidate_dcs.append(dc["id"])
        # For each DC where it's potentially available, check actual price (may be null if not bookable now)
        for dc in candidate_dcs:
            price_q = ('{gpuTypes(input:{id:"' + cfg["gpu_id"] +
                       '"}){lowestPrice(input:{gpuCount:' + str(cfg["count"]) +
                       ',dataCenterId:"' + dc +
                       '"}){uninterruptablePrice stockStatus}}}')
            r = gql(price_q)
            try:
                p = r["data"]["gpuTypes"][0]["lowestPrice"] or {}
            except (KeyError, IndexError, TypeError):
                continue
            price = p.get("uninterruptablePrice")
            if price and price <= cfg["max_price"]:
                candidates.append({
                    "gpu_id": cfg["gpu_id"],
                    "count": cfg["count"],
                    "dc": dc,
                    "price": price,
                    "volume_id": EXISTING_VOLUMES.get(dc),
                    "label": "{}x{}-{}".format(cfg["count"], cfg["gpu_id"].split()[1], dc),
                    "stock": p.get("stockStatus"),
                })
    if not candidates:
        return None
    # Prefer DCs with existing volumes (no download), then cheapest
    candidates.sort(key=lambda c: (c["volume_id"] is None, c["price"]))
    return candidates[0]


def deploy(target):
    needs_download = target["volume_id"] is None
    log("deploying {} at ${}/hr ({}-volume)".format(
        target["label"], target["price"], "fresh" if needs_download else "existing"))

    mut = """mutation deploy(
      $cloudType: CloudTypeEnum!, $gpuCount: Int!, $volumeInGb: Int!, $containerDiskInGb: Int!,
      $gpuTypeId: String!, $name: String!, $imageName: String!,
      $networkVolumeId: String, $ports: String!, $dataCenterId: String!,
      $env: [EnvironmentVariableInput!], $volumeMountPath: String
    ){
      podFindAndDeployOnDemand(input: {
        cloudType: $cloudType, gpuCount: $gpuCount,
        volumeInGb: $volumeInGb, containerDiskInGb: $containerDiskInGb,
        gpuTypeId: $gpuTypeId, name: $name, imageName: $imageName,
        networkVolumeId: $networkVolumeId, ports: $ports,
        dataCenterId: $dataCenterId, volumeMountPath: $volumeMountPath,
        env: $env
      }) { id desiredStatus }
    }"""

    env_vars = [{"key": "PUBLIC_KEY", "value": PUBKEY},
                {"key": "ANTHROPIC_API_KEY", "value": ANTHROPIC_KEY}]
    if HF_TOKEN:
        env_vars.append({"key": "HUGGING_FACE_HUB_TOKEN", "value": HF_TOKEN})

    variables = {
        "cloudType": "SECURE",
        "gpuCount": target["count"],
        "volumeInGb": 0,
        "containerDiskInGb": 500 if needs_download else 80,
        "gpuTypeId": target["gpu_id"],
        "name": "lindsey-sweep-" + target["dc"].lower(),
        # Pinned to May-3 canonical-run image. Earlier we tried 1.0.3-cu1290 (CUDA 12.9):
        # (1) US-NC-1 host driver was too old (12090) for torch 2.8 / cu1290 → driver crash;
        # (2) cu1290 image's pip-installed vllm resolved to 0.20.1 (released 2026-05-04) which
        # added a hard dep on `deep_gemm` for FP8. Pinning back to 1.0.2-cu1281-torch280-ubuntu2404
        # (the recipe's known-good image) + vllm==0.20.0 below restores the May-3 canonical stack.
        "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "networkVolumeId": target["volume_id"],
        "ports": "22/tcp",
        "dataCenterId": target["dc"],
        "env": env_vars,
        "volumeMountPath": "/workspace" if not needs_download else None,
    }

    r = gql(mut, variables)
    if r.get("data", {}).get("podFindAndDeployOnDemand"):
        return r["data"]["podFindAndDeployOnDemand"]["id"]
    log("deploy failed: " + json.dumps(r)[:400])
    return None


def teardown(pod_id):
    if not pod_id:
        return
    log("tearing down " + pod_id)
    gql('mutation{podTerminate(input:{podId:"' + pod_id + '"})}')


def get_ssh(pod_id):
    r = gql('{pod(input:{podId:"' + pod_id + '"}){runtime{uptimeInSeconds ports{ip publicPort privatePort}}}}')
    rt = ((r.get("data", {}).get("pod") or {}).get("runtime")) or {}
    for p in (rt.get("ports") or []):
        if p.get("privatePort") == 22:
            return p["ip"], int(p["publicPort"]), rt.get("uptimeInSeconds")
    return None, None, rt.get("uptimeInSeconds")


def wait_for_ssh(pod_id, timeout=1800):
    log("waiting for SSH (timeout " + str(timeout) + "s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        ip, port, up = get_ssh(pod_id)
        if ip:
            try:
                with socket.create_connection((ip, port), timeout=3):
                    r = subprocess.run(
                        ["ssh", "-p", str(port), "-i", SSH_KEY,
                         "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                         "-o", "ConnectTimeout=10",
                         "root@" + ip, "echo SSH_OK"],
                        capture_output=True, text=True, timeout=15,
                    )
                    if "SSH_OK" in r.stdout:
                        log("SSH up at {}:{} (uptime {}s)".format(ip, port, up))
                        return ip, port
            except Exception:
                pass
        time.sleep(15)
    raise RuntimeError("SSH timeout after " + str(timeout) + "s")


def ssh_exec(host, port, cmd, timeout=300):
    return subprocess.run(
        ["ssh", "-p", str(port), "-i", SSH_KEY,
         "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         "-o", "ConnectTimeout=10",
         "root@" + host, cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def scp_to(host, port, local, remote, timeout=120):
    return subprocess.run(
        ["scp", "-P", str(port), "-i", SSH_KEY,
         "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         local, "root@" + host + ":" + remote],
        capture_output=True, text=True, timeout=timeout,
    )


def maybe_download_weights(host, port, target):
    """If weights aren't already on /workspace, fresh-download from HF (~6 min on US/EU pods)."""
    r = ssh_exec(host, port,
        "ls /workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic/*.safetensors 2>/dev/null | wc -l")
    n = int((r.stdout.strip() or "0").split()[0]) if r.stdout.strip() else 0
    if n >= 86:
        log("weights already present ({} shards)".format(n))
        return

    notify("Fresh download — ~6 min for 410 GB FP8 weights")
    log("installing huggingface_hub[cli,hf_transfer] + downloading weights...")
    cmd = (
        "pip install -q --no-cache-dir 'huggingface_hub[cli,hf_transfer]' && "
        "mkdir -p /workspace/hf-cache && "
        "export HF_HUB_ENABLE_HF_TRANSFER=1 && "
        "hf download "
        "  RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic "
        "  --local-dir /workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic "
        "  --max-workers 8"
    )
    r = ssh_exec(host, port, cmd, timeout=900)  # 15 min cap
    log("download rc=" + str(r.returncode))
    if r.returncode != 0:
        log("download stderr: " + r.stderr[-500:])
        raise RuntimeError("HF download failed")
    # Sanity check
    r = ssh_exec(host, port, "ls /workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic/*.safetensors | wc -l")
    n = int((r.stdout.strip() or "0").split()[0])
    if n < 86:
        raise RuntimeError("only got {} shards after download".format(n))
    log("download done, {} shards".format(n))


def setup_pod(host, port):
    log("installing vllm==0.20.0 + vllm-lens==1.1.0 + anthropic + pydantic...")
    # vllm pinned to 0.20.0 (May-3 canonical-run version). 0.20.1 (released 2026-05-04) added
    # a hard FP8 dep on deep_gemm that breaks vllm-lens 1.1.0 paths.
    r = ssh_exec(host, port,
        "pip install --no-cache-dir 'vllm==0.20.0' 'vllm-lens==1.1.0' 'anthropic>=0.40' 'pydantic>=2' 2>&1 | tail -3",
        timeout=900)
    log("pip install rc=" + str(r.returncode))
    if r.returncode != 0:
        log("pip stdout: " + r.stdout[-1500:])
        log("pip stderr: " + r.stderr[-500:])
        raise RuntimeError("pip install failed")

    log("uploading lindsey_full_sweep.py + judge_lindsey_batch.py...")
    for f in ("lindsey_full_sweep.py", "judge_lindsey_batch.py"):
        r = scp_to(host, port, str(REPO / f), "/workspace/" + f)
        if r.returncode != 0:
            raise RuntimeError("scp failed for " + f + ": " + r.stderr)


def run_sweep(host, port, target):
    log("running dense-magnitude sweep (model load ~15 min + sweep ~3 min)...")
    cmd = (
        "set -o pipefail; "
        "mkdir -p /workspace/sweep-results && cd /workspace && "
        "MODEL_PATH=/workspace/hf-cache/Llama-3.1-405B-Instruct-FP8-dynamic "
        "LAYER=84 "
        "TP_SIZE=" + str(target["count"]) + " "
        "MAGNITUDES=10,10.5,11,11.5,12 "
        "CONDITIONS=lindsey "
        "OUT_DIR=/workspace/sweep-results "
        "VLLM_ALLREDUCE_USE_SYMM_MEM=0 "
        "python lindsey_full_sweep.py 2>&1 | tee /workspace/sweep-results/sweep.log"
    )
    r = ssh_exec(host, port, cmd, timeout=2700)  # 45 min cap
    log("sweep rc=" + str(r.returncode))
    log("sweep tail:\n" + r.stdout[-2000:])
    if r.returncode != 0:
        # Pull the full sweep.log for offline debugging before tearing down
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            dbg_dir = SCRIPTS / ("debug-" + ts + "-" + target["label"])
            dbg_dir.mkdir(exist_ok=True)
            subprocess.run(
                ["scp", "-P", str(port), "-i", SSH_KEY,
                 "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                 "root@" + host + ":/workspace/sweep-results/sweep.log",
                 str(dbg_dir / "sweep.log")],
                capture_output=True, text=True, timeout=120,
            )
            log("debug sweep.log pulled to " + str(dbg_dir))
        except Exception as e:
            log("failed to pull debug sweep.log: " + repr(e))
        raise RuntimeError("sweep failed rc=" + str(r.returncode))


def run_judge(host, port):
    log("running judge...")
    if not ANTHROPIC_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not loaded in auto_sweep — judge cannot run")
    cmd = (
        "set -o pipefail; "
        "cd /workspace/sweep-results && "
        "ANTHROPIC_API_KEY=" + ANTHROPIC_KEY + " "
        "python /workspace/judge_lindsey_batch.py 2>&1 | tee judge.log"
    )
    r = ssh_exec(host, port, cmd, timeout=900)
    log("judge rc=" + str(r.returncode))
    log("judge tail:\n" + r.stdout[-2000:])
    if r.returncode != 0:
        raise RuntimeError("judge failed rc=" + str(r.returncode))


def pull_results(host, port, label):
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = SCRIPTS / ("results-" + ts + "-" + label)
    out_dir.mkdir(exist_ok=True)
    log("pulling results to " + str(out_dir))
    r = subprocess.run(
        ["scp", "-r", "-P", str(port), "-i", SSH_KEY,
         "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
         "root@" + host + ":/workspace/sweep-results/",
         str(out_dir) + "/"],
        capture_output=True, text=True, timeout=300,
    )
    log("scp rc=" + str(r.returncode))
    if r.stderr:
        log("scp stderr: " + r.stderr[-300:])
    # Validate: a successful sweep produces a judged JSON. If we only see logs, the run failed silently.
    json_files = list(out_dir.rglob("*.json"))
    log("output JSON files: {}".format(len(json_files)))
    if not json_files:
        raise RuntimeError("pull_results: no .json files in output — sweep produced no real results")
    return out_dir


def run_pipeline(target):
    pod_id = None
    try:
        pod_id = deploy(target)
        if not pod_id:
            notify("deploy failed; will keep polling")
            return False
        notify("Deployed " + target["label"] + " — pod " + pod_id + ", waiting for SSH")
        host, port = wait_for_ssh(pod_id)
        notify("SSH up — installing deps")
        setup_pod(host, port)
        maybe_download_weights(host, port, target)
        notify("Running sweep...")
        run_sweep(host, port, target)
        notify("Sweep done — judging")
        run_judge(host, port)
        notify("Judge done — pulling results")
        out_dir = pull_results(host, port, target["label"])
        notify("DONE — results in " + out_dir.name)
        log("SUCCESS — results at " + str(out_dir))
        return True
    except Exception as e:
        log("PIPELINE FAILED: " + repr(e))
        log(traceback.format_exc())
        notify("Pipeline failed: " + str(e)[:80])
        return False
    finally:
        if pod_id:
            teardown(pod_id)


MAX_PIPELINE_ATTEMPTS = 1


def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log("=== auto_sweep.py started ===")
    log("Probe configs: " + ", ".join("{}x{}".format(c["count"], c["gpu_id"].split()[1]) for c in PROBE_CONFIGS))
    log("Existing volumes (preferred): " + ", ".join(EXISTING_VOLUMES.keys()))
    log("DC blacklist: " + ", ".join(sorted(DC_BLACKLIST)))
    log("SSH wait timeout: 1800s (RunPod boots are slow)")
    notify("Auto-poller started — polling globally for any 405B-fit slot")

    pipeline_attempts = 0

    while True:
        try:
            target = poll_for_slot()
            if target:
                pipeline_attempts += 1
                log("FOUND SLOT [attempt {}/{}]: {} at ${}/hr (volume={})".format(
                    pipeline_attempts, MAX_PIPELINE_ATTEMPTS,
                    target["label"], target["price"],
                    target["volume_id"] or "fresh"))
                ok = run_pipeline(target)
                if ok:
                    log("SUCCESS — exiting")
                    sys.exit(0)
                if pipeline_attempts >= MAX_PIPELINE_ATTEMPTS:
                    log("Hit MAX_PIPELINE_ATTEMPTS={}, giving up".format(MAX_PIPELINE_ATTEMPTS))
                    notify("Auto-poller giving up after {} failed attempts".format(MAX_PIPELINE_ATTEMPTS))
                    sys.exit(1)
                log("Pipeline failed; will keep polling for next slot ({}/{} attempts used)".format(
                    pipeline_attempts, MAX_PIPELINE_ATTEMPTS))
                # Wait a bit before next poll so we don't immediately re-grab a stuck DC
                time.sleep(120)
                continue
            log("no slot; sleeping 60s")
            time.sleep(60)
        except KeyboardInterrupt:
            log("interrupted; exiting")
            sys.exit(2)
        except Exception as e:
            log("poll loop error: " + repr(e))
            time.sleep(60)


if __name__ == "__main__":
    main()
