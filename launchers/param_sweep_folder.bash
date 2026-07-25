#!/usr/bin/env bash
# launchers/param_sweep_folder.bash — sweep ONE config parameter across every
# condition (config) in a folder.
#
# Usage:
#   param_sweep_folder.bash <config_folder> <param.path> <value> [<value> ...] \
#       [--no_sweep_tags] [-- | <passthrough flags for exp_file_launcher.bash ...>]
#
#   e.g. sweep the peg break threshold over 3 values for every force-importance condition:
#     param_sweep_folder.bash configs/exp_cfgs/force_imp/clearance_tests \
#         breakable_peg_cfg.break_force 10 12.5 15 --skip_existing
#
# LABEL:VALUE form — when a sweep value is long/ugly (e.g. a gym task id), give it a
# short label with a leading "LABEL:" so the run dirs, wandb groups and tags stay
# readable. The part before the first ':' is the label (used for --exp_name_suffix and
# the wandb tag); the part after is the actual VALUE fed to --set. A value with no ':'
# uses itself as the label (unchanged behavior). E.g.
#     param_sweep_folder.bash <folder> runner_cfg.task \
#         default:Isaac-Forge-PegInsert-Direct-v0 0p5mm:Isaac-Forge-PegInsert-Clear0p5-Direct-v0
#
# For each VALUE it runs the whole folder once via exp_file_launcher.bash with:
#     --set <param.path>=<value>      (baked into every run's copied config.yaml)
#     --exp_name_suffix __<leaf>_<label>   (keeps each value's runs + wandb group separate,
#                                          while each config keeps its own experiment.directory)
#     --wandb_tag sweep_<leaf> --wandb_tag <leaf>=<label>   (filter/group in the wandb UI;
#                                          suppress both with --no_sweep_tags)
# so N configs x M values => N*M runs, laid out as
#     runs/<family>/<config><suffix>/...   (one leaf dir per config per value).
#
# param.path is a dotted path rooted at a config header, exactly as runner.py's --set
# expects (sac_cfg.actor_lr, runner_cfg.num_envs, model_cfg.actor.actor_latent, ...).
# The override lands in the runtime config copy, not just the CLI — verify with:
#     grep -R "<leaf>:" runs/<family>/<config>__<leaf>_<value>/0/config.yaml
#
# Argument parsing: every leading argument after <param.path> is treated as a sweep
# VALUE until the first argument that begins with '-' (or an explicit '--'); from there
# on, all remaining arguments are forwarded verbatim to exp_file_launcher.bash (e.g.
# --skip_existing, --no_eval, --experiment_directory, --wandb_project). Negative-number
# values are therefore not supported positionally — none of the swept params need them.
#
# Runs sweeps sequentially (delegates the per-config loop, failure handling, and summary
# to exp_file_launcher.bash). Env vars pass through: LOGDIR=... and PYTHON=... still work.
set -uo pipefail

# ===== Args =====
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <config_folder> <param.path> <value> [<value> ...] [passthrough flags ...]" >&2
    echo "  e.g. $0 configs/exp_cfgs/force_imp/clearance_tests breakable_peg_cfg.break_force 10 12.5 15 --skip_existing" >&2
    exit 2
fi
CONFIG_FOLDER="$1"; shift
PARAM_PATH="$1"; shift

# param.path must be a dotted path rooted at a header (HEADER.something...), matching
# runner.py --set. Catch the common mistake of passing a bare leaf name here.
if [[ "$PARAM_PATH" != *.* ]]; then
    echo "[sweep] param.path must be a dotted path rooted at a config header, e.g." >&2
    echo "        'breakable_peg_cfg.break_force' or 'sac_cfg.actor_lr' (got '$PARAM_PATH')" >&2
    exit 2
fi

# Collect VALUES (leading non-dash args) then PASSTHROUGH (from first dash arg / '--').
VALUES=()
PASSTHROUGH_RAW=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --) shift; PASSTHROUGH_RAW+=("$@"); break ;;   # explicit end-of-values sentinel
        -*) PASSTHROUGH_RAW+=("$@"); break ;;          # first flag begins passthrough
        *)  VALUES+=("$1"); shift ;;
    esac
done

if [[ ${#VALUES[@]} -eq 0 ]]; then
    echo "[sweep] no sweep values given for '$PARAM_PATH'" >&2
    exit 2
fi

# Pull the sweep-level --no_sweep_tags out of the passthrough (it is NOT a valid
# exp_file_launcher flag); everything else forwards on verbatim.
SWEEP_TAGS=1
PASSTHROUGH=()
for a in "${PASSTHROUGH_RAW[@]}"; do
    if [[ "$a" == "--no_sweep_tags" ]]; then
        SWEEP_TAGS=0
    else
        PASSTHROUGH+=("$a")
    fi
done

# ===== Derived =====
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FOLDER_LAUNCHER="$SCRIPT_DIR/exp_file_launcher.bash"
[[ -f "$FOLDER_LAUNCHER" ]] || { echo "[sweep] folder launcher not found: $FOLDER_LAUNCHER" >&2; exit 1; }
[[ -d "$CONFIG_FOLDER" ]]  || { echo "[sweep] config folder not found: $CONFIG_FOLDER" >&2; exit 1; }

# Leaf name of the param (last dotted segment) — used to build readable, unique
# exp-name suffixes and wandb tags.
LEAF="${PARAM_PATH##*.}"

# Sanitize a value into something safe for a directory / wandb-tag component:
# keep [A-Za-z0-9._-], collapse everything else to '_'.
sanitize() { echo "$1" | sed 's/[^A-Za-z0-9._-]/_/g'; }

echo "[sweep] folder=$CONFIG_FOLDER"
echo "[sweep] param=$PARAM_PATH  values=(${VALUES[*]})  (${#VALUES[@]} value(s))"
[[ ${#PASSTHROUGH[@]} -gt 0 ]] && echo "[sweep] passthrough -> exp_file_launcher: ${PASSTHROUGH[*]}"

# ===== Sweep =====
FAILED_VALUES=()
for entry in "${VALUES[@]}"; do
    # LABEL:VALUE split (first ':' only). No ':' => label is the value itself.
    if [[ "$entry" == *:* ]]; then
        label="${entry%%:*}"
        value="${entry#*:}"
    else
        label="$entry"
        value="$entry"
    fi
    safe_label="$(sanitize "$label")"
    suffix="__${LEAF}_${safe_label}"

    # Auto wandb tags (unless --no_sweep_tags): one marking the sweep, one the value.
    SWEEP_TAG_FLAGS=()
    if [[ "$SWEEP_TAGS" -eq 1 ]]; then
        SWEEP_TAG_FLAGS=(--wandb_tag "sweep_${LEAF}" --wandb_tag "${LEAF}=${safe_label}")
    fi

    echo ""
    echo "[sweep] ####################################################################"
    echo "[sweep] #### VALUE: $PARAM_PATH=$value   (label=$label, exp_name_suffix=$suffix)"
    echo "[sweep] ####################################################################"

    rc=0
    bash "$FOLDER_LAUNCHER" "$CONFIG_FOLDER" \
        --set "$PARAM_PATH=$value" \
        --exp_name_suffix "$suffix" \
        "${SWEEP_TAG_FLAGS[@]}" \
        "${PASSTHROUGH[@]}" || rc=$?

    if [[ "$rc" -eq 0 ]]; then
        echo "[sweep] VALUE $PARAM_PATH=$value completed"
    else
        echo "[sweep] VALUE $PARAM_PATH=$value had failures (folder launcher exit $rc)" >&2
        FAILED_VALUES+=("$value (exit $rc)")
    fi
done

# ===== Summary =====
echo ""
echo "[sweep] ===================================================================="
echo "[sweep] SWEEP DONE. $((${#VALUES[@]} - ${#FAILED_VALUES[@]}))/${#VALUES[@]} value(s) clean"
for f in "${FAILED_VALUES[@]}"; do echo "[sweep]   VALUE WITH FAILURES: $f"; done
echo "[sweep] ===================================================================="

# Nonzero exit if any swept value had a failing run underneath it.
[[ ${#FAILED_VALUES[@]} -eq 0 ]]
