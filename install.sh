#!/bin/bash
set -e
echo "[Sky360] Installer starting..."

# Ensure root

if [ "$EUID" -ne 0 ]; then
	echo "[ERROR] Please run as root: sudo ./install.sh"
	exit 1
fi

# Check Python availability

if ! command -v python3 >/dev/null 2>&1; then
	echo "[ERROR] python3 is required but not installed."
	exit 1
fi

# Show mode

if [[ "$*" == *"--dry-run"* ]]; then
	echo "[Sky360] Running in DRY-RUN mode (no changes will be made)"
fi

# Execute Python installer and forward ALL arguments

python3 "$(dirname "$0")/install.py" "$@"
echo "[Sky360] Installer finished."
