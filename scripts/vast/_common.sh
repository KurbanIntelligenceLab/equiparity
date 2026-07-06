#!/bin/bash
# ==============================================================
# Shared helpers for the equiparity vast.ai launch/fetch scripts.
# Adapted from E3-GRAND/scripts/vast/_common.sh. Sourced by launch.sh / fetch_results.sh.
#
# Provides:
#   vast_load_api_key             - read VAST_API_KEY from agents-mlip/.env and register it
#   vast_find_offer GPU MAXP DISK - echo "OFFER_ID DPH GPU_NAME" for the cheapest qualifying offer
#   vast_inject_onstart SCRIPT ENV - build an --onstart-cmd that base64-injects a local script
#   vast_create_start ...         - create + force-start an instance, echo its id
#   vast_ssh_hostport ID          - echo "HOST PORT" for an instance's SSH endpoint
# ==============================================================

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARITY_ROOT="$(cd "${_COMMON_DIR}/../.." && pwd)"
# agents-mlip sits beside ParityInMS and holds the shared .env (VAST_API_KEY).
AGENTS_MLIP_ROOT="${AGENTS_MLIP_ROOT:-$(cd "${PARITY_ROOT}/../agents-mlip" 2>/dev/null && pwd || true)}"

# The image bakes the processed data in, so a data URL is only needed to OVERRIDE it at runtime.
DEFAULT_DATA_URL="${DEFAULT_DATA_URL:-}"

# Offer quality gates: reliable, long-lived, single-GPU hosts.
RELIABILITY_MIN="${RELIABILITY_MIN:-0.99}"
DURATION_MIN_DAYS="${DURATION_MIN_DAYS:-7}"
INET_DOWN_MIN="${INET_DOWN_MIN:-200}"   # Mbps; fast enough to pull the ~3.5 GB image without stalling
DLPERF_MIN="${DLPERF_MIN:-200}"         # vast DLPerf compute score (confirmed filter)

vast_load_api_key() {
    if ! command -v vastai &>/dev/null; then
        echo "ERROR: vastai CLI not found. Install with: uv tool install vastai (or: pip install vastai)" >&2
        return 1
    fi
    local env_file="${AGENTS_MLIP_ROOT}/.env"
    if [[ -z "${VAST_API_KEY:-}" && -f "$env_file" ]]; then
        VAST_API_KEY="$(grep -E '^VAST_API_KEY=' "$env_file" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')"
    fi
    if [[ -z "${VAST_API_KEY:-}" ]]; then
        echo "ERROR: VAST_API_KEY not set and not found in ${env_file}" >&2
        return 1
    fi
    vastai set api-key "$VAST_API_KEY" >/dev/null
}

# Expand a friendly GPU choice into candidate vast.ai gpu_name tokens.
# RTX 5090 primary; RTX PRO 6000 (Blackwell 96GB _WS/_S) as the availability fallback. No A100.
_vast_gpu_tokens() {
    case "$1" in
        RTX_5090)       echo "RTX_5090" ;;
        RTX_PRO_6000)   echo "RTX_PRO_6000_WS RTX_PRO_6000_S" ;;
        either|"")      echo "RTX_5090 RTX_PRO_6000_WS RTX_PRO_6000_S" ;;
        *)              echo "$1" ;;
    esac
}

# Echo "OFFER_ID DPH GPU_NAME" of the cheapest offer meeting all gates, or nothing.
vast_find_offer() {
    local gpu_choice="$1" max_price="$2" disk="$3"
    local tmp; tmp="$(mktemp -d)"
    local g
    for g in $(_vast_gpu_tokens "$gpu_choice"); do
        vastai search offers \
            "gpu_name=${g} reliability>${RELIABILITY_MIN} duration>=${DURATION_MIN_DAYS} disk_space>=${disk} inet_down>${INET_DOWN_MIN} dlperf>${DLPERF_MIN} num_gpus=1 rentable=true dph<${max_price}" \
            -o 'dph+' --raw 2>/dev/null > "${tmp}/${g}.json" || echo '[]' > "${tmp}/${g}.json"
    done
    python3 - "$tmp" "$max_price" <<'PY'
import json, sys, glob, os
tmp, max_price = sys.argv[1], float(sys.argv[2])
best = None
for f in glob.glob(os.path.join(tmp, "*.json")):
    try:
        offers = json.load(open(f))
    except Exception:
        continue
    for o in offers or []:
        dph = o.get("dph_total", o.get("dph"))
        if dph is None or dph >= max_price:
            continue
        if best is None or dph < best[1]:
            best = (o["id"], dph, o.get("gpu_name", "?"))
if best:
    print(f"{best[0]} {best[1]:.3f} {best[2]}")
PY
    rm -rf "$tmp"
}

# Build an --onstart-cmd that base64-injects a LOCAL script onto the instance and runs it, so onstart
# logic can change WITHOUT rebuilding the image. Args: script_path, "ENV1=v1 ENV2=v2".
vast_inject_onstart() {
    local script_path="$1" env_str="${2:-}"
    local b64; b64="$(base64 -w0 "$script_path")"
    local cmd=""
    [[ -n "$env_str" ]] && cmd+="export ${env_str} && "
    cmd+="echo '${b64}' | base64 -d > /workspace/onstart_injected.sh && bash /workspace/onstart_injected.sh"
    echo "$cmd"
}

# Create an instance and force it RUNNING. Args: offer_id image disk label onstart_cmd. Echoes the id.
vast_create_start() {
    local offer_id="$1" image="$2" disk="$3" label="$4" onstart_cmd="$5"
    local login_args=()
    [[ -n "${VAST_DOCKER_LOGIN:-}" ]] && login_args=(--login "$VAST_DOCKER_LOGIN")
    local result
    result="$(vastai create instance "$offer_id" --image "$image" --disk "$disk" \
        --ssh --direct --label "$label" --onstart-cmd "$onstart_cmd" "${login_args[@]}" 2>&1)" || true
    echo "$result" >&2
    local id
    id="$(echo "$result" | grep -oP "'new_contract':\s*\K[0-9]+" || true)"
    [[ -z "$id" ]] && return 1
    vastai start instance "$id" >&2 2>&1 || true
    echo "$id"
}

# Echo "HOST PORT" for an instance's SSH endpoint. PREFERS the direct endpoint
# (public_ipaddr + the host port mapped to container port 22): vast's proxy reverse-tunnel
# (sshN.vast.ai) frequently fails ("remote port forwarding failed for listen port N"), whereas
# the direct port mapping is reliable. Tests TCP reachability and falls back to the proxy.
vast_ssh_hostport() {
    local id="$1" raw direct proxy
    raw="$(vastai show instance "$id" --raw 2>/dev/null)"
    direct="$(echo "$raw" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit()
ip=d.get('public_ipaddr'); ports=(d.get('ports') or {}).get('22/tcp') or []
hp=ports[0].get('HostPort') if ports else None
if ip and hp: print(f'{ip.strip()} {hp}')" 2>/dev/null)"
    proxy="$(echo "$raw" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit()
h,p=d.get('ssh_host'),d.get('ssh_port')
if h and p: print(f'{h} {p}')" 2>/dev/null)"
    local cand host port
    for cand in "$direct" "$proxy"; do
        [[ -z "$cand" ]] && continue
        host="${cand% *}"; port="${cand#* }"
        if timeout 4 bash -c "exec 3<>/dev/tcp/${host}/${port}" 2>/dev/null; then
            echo "$host $port"; return 0
        fi
    done
    # Nothing reachable yet — return direct (best guess) so the caller can retry over time.
    [[ -n "$direct" ]] && echo "$direct" || echo "$proxy"
}
