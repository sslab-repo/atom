#!/usr/bin/env bash
# ATOM installer — self-contained, per-user installation.
#
#   Linux (RHEL 10):  no root required. Everything lives under $HOME/atom;
#                     environment lands in ~/.bashrc (bash).
#   macOS:            everything under $HOME/atom; environment in ~/.zshrc
#                     (zsh). Admin authentication is requested ONLY if a
#                     suitable Python must be installed via Homebrew.
#
# Usage:
#   bash scripts/install.sh                 install / upgrade
#   bash scripts/install.sh --prefix DIR    install somewhere else
#   bash scripts/install.sh --no-rc         don't touch shell rc files
#   bash scripts/install.sh --uninstall     remove install + rc block
#
# Layout created:
#   $ATOM_ROOT/app      copy of the ATOM source tree (installed from here)
#   $ATOM_ROOT/venv     private Python virtualenv
#   $ATOM_ROOT/bin      the `atom` launcher (added to PATH)
#   $ATOM_ROOT/config   atom.env — user configuration, sourced by launcher
#   $ATOM_ROOT/home     ATOM_HOME: meta-KB, caches (kept inside the folder)
#   $ATOM_ROOT/data     your dataset packages (ADPs)
#   $ATOM_ROOT/runs     run outputs (provenance + model packages)

set -euo pipefail

ATOM_ROOT="${ATOM_PREFIX:-$HOME/atom}"
DO_RC=1
UNINSTALL=0
MIN_PY="3.10"

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) ATOM_ROOT="$2"; shift 2 ;;
    --no-rc) DO_RC=0; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

BOLD=$(printf '\033[1m'); CYAN=$(printf '\033[36m'); RED=$(printf '\033[31m'); NC=$(printf '\033[0m')
log() { printf '%s[atom]%s %s\n' "${CYAN}${BOLD}" "${NC}" "$*"; }
die() { printf '%s[atom] ERROR:%s %s\n' "${RED}${BOLD}" "${NC}" "$*" >&2; exit 1; }

OS="$(uname -s)"
case "$OS" in
  Linux)  RC_FILE="$HOME/.bashrc"; SHELL_NAME="bash" ;;
  Darwin) RC_FILE="$HOME/.zshrc";  SHELL_NAME="zsh" ;;
  *) die "unsupported platform: $OS (supported: Linux/RHEL, macOS)" ;;
esac

RC_BEGIN="# >>> ATOM environment >>>"
RC_END="# <<< ATOM environment <<<"

remove_rc_block() {
  [ -f "$RC_FILE" ] || return 0
  if grep -qF "$RC_BEGIN" "$RC_FILE"; then
    awk -v b="$RC_BEGIN" -v e="$RC_END" '
      $0==b {skip=1; next} $0==e {skip=0; next} !skip {print}' \
      "$RC_FILE" > "$RC_FILE.atom-tmp" && mv "$RC_FILE.atom-tmp" "$RC_FILE"
  fi
}

# ---------------------------------------------------------------- uninstall
if [ "$UNINSTALL" -eq 1 ]; then
  log "uninstalling: removing rc block from $RC_FILE"
  remove_rc_block
  if [ -d "$ATOM_ROOT" ]; then
    printf 'Delete %s and EVERYTHING in it (incl. data/, runs/, meta-KB)? [y/N] ' "$ATOM_ROOT"
    read -r answer
    case "$answer" in
      y|Y|yes) rm -rf "$ATOM_ROOT"; log "removed $ATOM_ROOT" ;;
      *) log "kept $ATOM_ROOT (only the shell environment block was removed)" ;;
    esac
  fi
  log "uninstall complete"
  exit 0
fi

# ------------------------------------------------------------- find sources
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$SRC_ROOT/pyproject.toml" ] && [ -d "$SRC_ROOT/src/atom" ] \
  || die "run this from an ATOM checkout (scripts/install.sh); pyproject.toml not found at $SRC_ROOT"

log "installing ATOM from $SRC_ROOT"
log "target: $ATOM_ROOT  (platform: $OS, shell: $SHELL_NAME)"

# ------------------------------------------------------------- find python
python_ok() { "$1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; }

PY=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
    PY="$(command -v "$candidate")"
    break
  fi
done

if [ -z "$PY" ] && [ "$OS" = "Darwin" ]; then
  log "no Python >= $MIN_PY found — using Homebrew (this may request ADMIN authentication)"
  if ! command -v brew >/dev/null 2>&1; then
    printf 'Homebrew is not installed. Install it now (requires admin password)? [y/N] '
    read -r answer
    case "$answer" in
      y|Y|yes)
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
          || die "Homebrew installation failed"
        eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || /usr/local/bin/brew shellenv)"
        ;;
      *) die "Python >= $MIN_PY is required. Install it (e.g. from python.org) and re-run." ;;
    esac
  fi
  brew install -q python@3.12 || die "brew install python@3.12 failed"
  for candidate in python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && python_ok "$candidate"; then
      PY="$(command -v "$candidate")"; break
    fi
  done
fi

if [ -z "$PY" ]; then
  if [ "$OS" = "Linux" ]; then
    die "no Python >= $MIN_PY found. RHEL 10 ships python3 3.12 — if this host lacks it, \
ask an administrator to run once: sudo dnf install -y python3  (no root is needed after that)"
  fi
  die "no Python >= $MIN_PY found"
fi
log "using Python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

# ------------------------------------------------------------ create layout
for d in app bin config home data runs; do mkdir -p "$ATOM_ROOT/$d"; done

log "copying application files -> $ATOM_ROOT/app"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'runs' \
    "$SRC_ROOT/pyproject.toml" "$SRC_ROOT/README.md" "$SRC_ROOT/src" "$SRC_ROOT/docs" \
    "$ATOM_ROOT/app/"
else
  rm -rf "$ATOM_ROOT/app"; mkdir -p "$ATOM_ROOT/app"
  cp -R "$SRC_ROOT/pyproject.toml" "$SRC_ROOT/README.md" "$SRC_ROOT/src" "$SRC_ROOT/docs" \
    "$ATOM_ROOT/app/"
fi

# -------------------------------------------------------------- virtualenv
if [ ! -x "$ATOM_ROOT/venv/bin/python" ]; then
  log "creating virtualenv -> $ATOM_ROOT/venv"
  "$PY" -m venv "$ATOM_ROOT/venv"
fi
log "installing ATOM + dependencies into the venv (this can take a few minutes)"
"$ATOM_ROOT/venv/bin/python" -m pip install --quiet --upgrade pip
"$ATOM_ROOT/venv/bin/python" -m pip install --quiet --upgrade "$ATOM_ROOT/app" \
  || die "pip install failed (network required on first install)"

# ------------------------------------------------------------ configuration
CONFIG="$ATOM_ROOT/config/atom.env"
if [ ! -f "$CONFIG" ]; then
  log "writing default configuration -> $CONFIG"
  cat > "$CONFIG" <<EOF
# ATOM user configuration (sourced by the atom launcher and your shell).
# Everything ATOM writes stays under ATOM_ROOT.
export ATOM_ROOT="$ATOM_ROOT"
export ATOM_HOME="\$ATOM_ROOT/home"          # meta-KB + caches
export ATOM_DEFAULT_OUT="\$ATOM_ROOT/runs"   # default --out for runs
# export ATOM_KB=""                          # override meta-KB location
EOF
else
  log "keeping existing configuration: $CONFIG"
fi

# ---------------------------------------------------------------- launcher
cat > "$ATOM_ROOT/bin/atom" <<EOF
#!/usr/bin/env bash
# ATOM launcher — sources user config, then runs the venv entry point.
set -e
. "$ATOM_ROOT/config/atom.env"
default_out=()
if [ "\${1:-}" = "run" ] && [ -n "\${ATOM_DEFAULT_OUT:-}" ]; then
  case " \$* " in *" --out "*) ;; *) default_out=(--out "\$ATOM_DEFAULT_OUT");; esac
fi
exec "$ATOM_ROOT/venv/bin/atom" "\$@" "\${default_out[@]}"
EOF
chmod +x "$ATOM_ROOT/bin/atom"

# ------------------------------------------------------- shell environment
if [ "$DO_RC" -eq 1 ]; then
  log "updating $RC_FILE ($SHELL_NAME)"
  remove_rc_block
  cat >> "$RC_FILE" <<EOF
$RC_BEGIN
. "$ATOM_ROOT/config/atom.env"
export PATH="$ATOM_ROOT/bin:\$PATH"
$RC_END
EOF
else
  log "skipping shell rc update (--no-rc)"
fi

# ------------------------------------------------------------- self test
log "running post-install self-test"
"$ATOM_ROOT/bin/atom" modules verify >/dev/null \
  || die "self-test failed: 'atom modules verify' did not pass"
SELFTEST_CSV="$ATOM_ROOT/data/.selftest.csv"
printf 'x1,x2,label\n1.0,2.0,a\n2.0,1.0,b\n1.5,1.5,a\n0.5,2.5,b\n' > "$SELFTEST_CSV"
"$ATOM_ROOT/bin/atom" pack "$SELFTEST_CSV" --target label \
  --out "$ATOM_ROOT/data" --name .selftest-pkg >/dev/null
"$ATOM_ROOT/bin/atom" inspect "$ATOM_ROOT/data/.selftest-pkg" >/dev/null
rm -rf "$SELFTEST_CSV" "$ATOM_ROOT/data/.selftest-pkg"
log "self-test passed (modules verify + pack + inspect)"

log "ATOM installed successfully."
printf '\n  Next steps:\n'
printf '    1. reload your shell:   source %s\n' "$RC_FILE"
printf '    2. try it:              atom modules list\n'
printf '    3. run on a package:    atom run <package.zip> --target <column>\n'
printf '\n  Everything lives in %s — config: config/atom.env, runs: runs/, meta-KB: home/\n\n' "$ATOM_ROOT"
