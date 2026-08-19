#!/usr/bin/env bash
# ATOM one-shot setup — installs SYSTEM prerequisites, then ATOM itself, then
# runs a health check and points at a sample test script.
#
# Supported platforms:
#   - macOS               (Homebrew)
#   - RHEL 10 / Rocky / AlmaLinux / Fedora   (dnf)
#   - Debian 13 / Ubuntu  (apt)
#
# System packages install with sudo (Linux) or Homebrew (macOS). The ATOM app
# itself installs per-user under $HOME/atom (no root) via scripts/install.sh.
#
# Usage:
#   bash scripts/setup.sh                 install system deps + ATOM, then health-check
#   bash scripts/setup.sh --yes           non-interactive (assume yes to prompts)
#   bash scripts/setup.sh --no-system     skip OS packages (Python already present)
#   bash scripts/setup.sh --no-rc         don't modify shell rc files
#   bash scripts/setup.sh --prefix DIR    install ATOM somewhere other than ~/atom
#   bash scripts/setup.sh --help
set -euo pipefail

ASSUME_YES=0
DO_SYSTEM=1
DO_RC=1
PREFIX=""
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) ASSUME_YES=1; shift ;;
    --no-system) DO_SYSTEM=0; shift ;;
    --no-rc) DO_RC=0; shift ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

BOLD=$(printf '\033[1m'); CYAN=$(printf '\033[36m'); GREEN=$(printf '\033[32m')
RED=$(printf '\033[31m'); YEL=$(printf '\033[33m'); NC=$(printf '\033[0m')
log()  { printf '%s[setup]%s %s\n' "${CYAN}${BOLD}" "${NC}" "$*"; }
die()  { printf '%s[setup] ERROR:%s %s\n' "${RED}${BOLD}" "${NC}" "$*" >&2; exit 1; }
ask()  { [ "$ASSUME_YES" -eq 1 ] && return 0; printf '%s [y/N] ' "$1"; read -r a; case "$a" in y|Y|yes) return 0;; *) return 1;; esac; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$SRC_ROOT/scripts/install.sh" ] || die "run from an ATOM checkout (scripts/setup.sh)"

# ------------------------------------------------------ detect platform / distro
OS="$(uname -s)"; DISTRO=""; PKG=""
case "$OS" in
  Darwin) PKG="brew" ;;
  Linux)
    [ -r /etc/os-release ] && . /etc/os-release || true
    DISTRO="${ID:-unknown}"
    case "${ID:-}:${ID_LIKE:-}" in
      *rhel*|*fedora*|*centos*|*rocky*|*almalinux*) PKG="dnf" ;;
      *debian*|*ubuntu*)                            PKG="apt" ;;
      *) case "${ID_LIKE:-}" in
           *rhel*|*fedora*) PKG="dnf" ;;
           *debian*)        PKG="apt" ;;
           *) die "unsupported Linux distro '${ID:-?}'. Install python3(>=3.10) + pip + venv manually, then re-run with --no-system." ;;
         esac ;;
    esac ;;
  *) die "unsupported platform: $OS" ;;
esac
log "platform: ${OS}${DISTRO:+ / $DISTRO}   package manager: $PKG"

# ------------------------------------------------------ system prerequisites
sudo_maybe() { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi; }

install_system_deps() {
  case "$PKG" in
    dnf)
      # RHEL 10 ships Python 3.12. libgomp = OpenMP runtime for scikit-learn/BLAS.
      local pkgs=(python3 python3-pip python3-devel gcc gcc-c++ libgomp git)
      log "system packages (dnf): ${pkgs[*]}"
      ask "Install these with sudo dnf?" || { log "skipped system packages"; return 0; }
      sudo_maybe dnf install -y "${pkgs[@]}" || die "dnf install failed"
      ;;
    apt)
      # Debian 13 splits venv into python3-venv; libgomp1 = OpenMP runtime.
      local pkgs=(python3 python3-pip python3-venv python3-dev build-essential libgomp1 git ca-certificates)
      log "system packages (apt): ${pkgs[*]}"
      ask "Install these with sudo apt-get?" || { log "skipped system packages"; return 0; }
      sudo_maybe apt-get update -y || die "apt-get update failed"
      sudo_maybe apt-get install -y "${pkgs[@]}" || die "apt-get install failed"
      ;;
    brew)
      if ! command -v brew >/dev/null 2>&1; then
        ask "Homebrew is not installed. Install it now (requests admin password)?" \
          || die "Homebrew is required on macOS (or install Python 3.10+ yourself and re-run --no-system)"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
          || die "Homebrew install failed"
        eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
      fi
      # libomp = OpenMP runtime some scientific wheels dlopen on macOS.
      log "system packages (brew): python@3.12 git libomp"
      brew install -q python@3.12 git libomp || die "brew install failed"
      ;;
  esac
}

if [ "$DO_SYSTEM" -eq 1 ]; then
  install_system_deps
else
  log "skipping system packages (--no-system)"
fi

# ------------------------------------------------------ install ATOM (per-user)
log "installing ATOM (per-user, no root)"
INSTALL_ARGS=()
[ -n "$PREFIX" ] && INSTALL_ARGS+=(--prefix "$PREFIX")
[ "$DO_RC" -eq 0 ] && INSTALL_ARGS+=(--no-rc)
# empty-array-safe expansion: macOS ships bash 3.2, where "${arr[@]}" on an
# empty array under `set -u` raises "unbound variable".
bash "$SRC_ROOT/scripts/install.sh" ${INSTALL_ARGS[@]+"${INSTALL_ARGS[@]}"}

ATOM_ROOT="${PREFIX:-$HOME/atom}"
ATOM_BIN="$ATOM_ROOT/bin/atom"
[ -x "$ATOM_BIN" ] || die "ATOM launcher not found at $ATOM_BIN after install"

# ------------------------------------------------------ health check
printf '\n%s========================  HEALTH CHECK  ========================%s\n' "${BOLD}" "${NC}"
pass=0; fail=0
HC_LOG="$(mktemp 2>/dev/null || echo "${TMPDIR:-/tmp}/atom_hc.$$")"
trap 'rm -f "$HC_LOG"' EXIT
check() { # name, command...
  local name="$1"; shift
  if "$@" >"$HC_LOG" 2>&1; then
    printf '  %s✔%s  %-34s %s\n' "$GREEN" "$NC" "$name" "$(tail -n1 "$HC_LOG" 2>/dev/null | cut -c1-40)"
    pass=$((pass+1))
  else
    printf '  %sx%s  %-34s %sFAILED%s\n' "$RED" "$NC" "$name" "$RED" "$NC"
    sed 's/^/       /' "$HC_LOG" | tail -n3
    fail=$((fail+1))
  fi
}

PYV="$("$ATOM_ROOT/venv/bin/python" -c 'import sys;print(".".join(map(str,sys.version_info[:3])))' 2>/dev/null || echo '?')"
check "python interpreter ($PYV)"    "$ATOM_ROOT/venv/bin/python" -c 'import sys; assert sys.version_info>=(3,10)'
check "core libraries import"        "$ATOM_ROOT/venv/bin/python" -c 'import numpy,pyarrow,sklearn,onnxruntime,skl2onnx'
check "atom command responds"        "$ATOM_BIN" --help
check "module smoke gate"            "$ATOM_BIN" modules verify

# end-to-end: pack -> inspect -> run -> deployable ONNX
HC_DIR="$ATOM_ROOT/data/.healthcheck"; rm -rf "$HC_DIR"; mkdir -p "$HC_DIR"
"$ATOM_ROOT/venv/bin/python" - "$HC_DIR/hc.csv" <<'PY'
import csv, random, sys
rng = random.Random(0)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["x1", "x2", "label"])
    for i in range(400):
        c = rng.random() < 0.5
        w.writerow([rng.gauss(2 if c else 0, 1), rng.gauss(0, 1), "P" if c else "N"])
PY
check "pack CSV -> dataset package"  "$ATOM_BIN" pack "$HC_DIR/hc.csv" --target label --name hcpkg --out "$HC_DIR"
check "inspect dataset package"      "$ATOM_BIN" inspect "$HC_DIR/hcpkg"
check "end-to-end run -> ONNX AMP"   bash -c "\"$ATOM_BIN\" run \"$HC_DIR/hcpkg\" --time-budget 20 --yes --out \"$HC_DIR/runs\" >/dev/null && ls \"$HC_DIR/runs\"/*/model/pipeline.onnx"
rm -rf "$HC_DIR"

printf '%s================================================================%s\n' "${BOLD}" "${NC}"
if [ "$fail" -eq 0 ]; then
  printf '  %sALL %d CHECKS PASSED%s — ATOM is installed and working.\n' "$GREEN$BOLD" "$pass" "$NC"
else
  printf '  %s%d passed, %d FAILED%s — see messages above.\n' "$RED$BOLD" "$pass" "$fail" "$NC"
fi
printf '================================================================\n\n'

# ------------------------------------------------------ next steps + sample
printf '  %sNext steps%s\n' "$BOLD" "$NC"
printf '    1. reload your shell:   source %s\n' "$([ "$OS" = Darwin ] && echo ~/.zshrc || echo ~/.bashrc)"
printf '    2. verify on PATH:      atom modules list\n'
printf '    3. sample test run:     bash %s/scripts/sample_run.sh\n\n' "$SRC_ROOT"
printf '  Manual: docs/manual.md   ·   Slurm (many datasets): docs/slurm.md\n\n'

[ "$fail" -eq 0 ] || exit 1
