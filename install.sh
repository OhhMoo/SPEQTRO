#!/usr/bin/env bash
# speqtro installer
# Usage: curl -fsSL https://raw.githubusercontent.com/OhhMoo/SPEQTRO-Agent/master/install.sh | bash

set -e

REPO="https://github.com/OhhMoo/SPEQTRO-Agent.git"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10
MAX_PYTHON_MINOR=12

# ── colours ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[speqtro]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[speqtro]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[speqtro]${RESET} $*"; }
error()   { echo -e "${RED}${BOLD}[speqtro]${RESET} $*" >&2; exit 1; }

# ── banner ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}  ███████╗██████╗ ███████╗ ██████╗████████╗██████╗  ██████╗ ${RESET}"
echo -e "${CYAN}${BOLD}  ██╔════╝██╔══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗${RESET}"
echo -e "${CYAN}${BOLD}  ███████╗██████╔╝█████╗  ██║  ███╗  ██║   ██████╔╝██║   ██║${RESET}"
echo -e "${CYAN}${BOLD}  ╚════██║██╔═══╝ ██╔══╝  ██║   ██║  ██║   ██╔══██╗██║   ██║${RESET}"
echo -e "${CYAN}${BOLD}  ███████║██║     ███████╗╚██████╔╝  ██║   ██║  ██║╚██████╔╝${RESET}"
echo -e "${CYAN}${BOLD}  ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ${RESET}"
echo ""
echo -e "  Autonomous spectroscopy reasoning agent for chemists."
echo ""

# ── check OS ───────────────────────────────────────────────────────────────────
OS="$(uname -s 2>/dev/null || echo Unknown)"
case "$OS" in
  Linux|Darwin) ;;
  MINGW*|MSYS*|CYGWIN*) warn "Windows detected — use PowerShell or WSL for best results." ;;
  *) warn "Unrecognised OS: $OS — continuing anyway." ;;
esac

# ── find python ────────────────────────────────────────────────────────────────
find_python() {
  for cmd in python3 python python3.12 python3.11 python3.10; do
    if command -v "$cmd" &>/dev/null; then
      ver="$("$cmd" -c 'import sys; print(sys.version_info[:2])' 2>/dev/null)"
      major="$("$cmd" -c 'import sys; print(sys.version_info[0])')"
      minor="$("$cmd" -c 'import sys; print(sys.version_info[1])')"
      if [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && \
         [ "$minor" -ge "$MIN_PYTHON_MINOR" ] && \
         [ "$minor" -le "$MAX_PYTHON_MINOR" ]; then
        echo "$cmd"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON="$(find_python || true)"

if [ -z "$PYTHON" ]; then
  error "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}–${MIN_PYTHON_MAJOR}.${MAX_PYTHON_MINOR} is required but not found.\n  Install it from https://python.org and re-run this script."
fi

PY_VER="$("$PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
info "Using Python $PY_VER  ($PYTHON)"

# ── choose install method ──────────────────────────────────────────────────────
USE_PIPX=false
if command -v pipx &>/dev/null; then
  USE_PIPX=true
  info "pipx detected — will install into an isolated environment."
else
  info "pipx not found — will install with pip."
fi

# ── optional: RDKit ────────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}Optional:[${RESET} RDKit adds cheminformatics support (recommended)."
printf "  Install with RDKit? [Y/n] "
read -r RDKIT_CHOICE </dev/tty 2>/dev/null || RDKIT_CHOICE="y"
RDKIT_CHOICE="${RDKIT_CHOICE:-y}"

EXTRAS=""
if [[ "$RDKIT_CHOICE" =~ ^[Yy]$ ]]; then
  EXTRAS="[chemistry]"
fi

# ── install ────────────────────────────────────────────────────────────────────
echo ""
PACKAGE="git+${REPO}"

if $USE_PIPX; then
  info "Running: pipx install \"${PACKAGE}${EXTRAS}\""
  if [ -n "$EXTRAS" ]; then
    pipx install "$PACKAGE" --pip-args "speqtro${EXTRAS}" || \
    pipx install "speqtro${EXTRAS}" --pip-args "" 2>/dev/null || \
    pipx install "$PACKAGE"
  else
    pipx install "$PACKAGE"
  fi
else
  info "Running: $PYTHON -m pip install \"${PACKAGE}${EXTRAS}\""
  "$PYTHON" -m pip install --upgrade "${PACKAGE}${EXTRAS}"
fi

# ── verify ─────────────────────────────────────────────────────────────────────
echo ""
if command -v speqtro &>/dev/null; then
  INSTALLED_VER="$(speqtro --version 2>/dev/null || echo 'unknown')"
  success "speqtro installed successfully!  (version: $INSTALLED_VER)"
else
  warn "Installation finished but 'speqtro' was not found on PATH."
  warn "You may need to restart your shell or add the install directory to PATH."
  if $USE_PIPX; then
    warn "  Try: pipx ensurepath"
  fi
fi

# ── next steps ─────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
echo -e "  ${CYAN}speqtro setup${RESET}       — configure your Anthropic API key"
echo -e "  ${CYAN}speqtro doctor${RESET}      — verify dependencies and connectivity"
echo -e "  ${CYAN}speqtro${RESET}             — launch the interactive REPL"
echo ""
echo -e "  Docs: ${CYAN}https://github.com/OhhMoo/SPEQTRO-Agent${RESET}"
echo ""
