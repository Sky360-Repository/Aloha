#!/usr/bin/env python3
# coding: utf-8

# \copyright    Sky360.org
#
# \brief        Installer of Sky360 services
#               detects hostname
#               reads INI
#               installs files
#               installs config
#               installs udev rules
#               installs systemd service
#               enables service
#
#               options:
#               --host <hostname> # config selection used for testing with --dry-run option
#               --dry-run # no changes will be made
#
#               example:
#               sudo ./install.sh --host S360asc --dry-run
#               this will read the configuration for the host S360asc and does a dry-run without any changes
#
# ************************************************************************

import argparse
import configparser
import shutil
import socket
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config"

SYSTEMD_DIR = Path("/etc/systemd/system")
UDEV_DIR = Path("/etc/udev/rules.d")

class Installer:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def log(self, msg):
        print(msg)

    def run(self, cmd):
        self.log(f"[CMD] {cmd}")
        if not self.dry_run:
            subprocess.run(cmd, shell=True, check=True)

    def write_file(self, path: Path, content: str):
        self.log(f"[WRITE] {path}")
        if not self.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    def copy_file(self, src: Path, dst: Path):
        self.log(f"[COPY] {src} -> {dst}")
        if not self.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

# ----------------------------
# Helper functions
# ----------------------------

def get_hostname():
    return socket.gethostname()


def load_config(hostname):
    base_ini = CONFIG_DIR / "S360base.ini"
    host_ini = CONFIG_DIR / f"{hostname}.ini"

    if not base_ini.exists():
        raise FileNotFoundError(f"Missing config: {base_ini}")

    if not host_ini.exists():
        raise FileNotFoundError(f"Missing config: {host_ini}")

    parser = configparser.ConfigParser()

    # load base first
    parser.read(base_ini)

    # overlay host config
    parser.read(host_ini)

    return parser, host_ini


def ensure_user(inst, user, group):
    # Ensure primary group exists
    inst.run(f"groupadd -f {group}")

    # Create user if it doesn't exist
    inst.run(f"id -u {user} || useradd -r -g {group} -s /usr/sbin/nologin {user}")

    # Ensure supplementary groups exist
    extra_groups = ["video", "dialout"]
    for g in extra_groups:
        inst.run(f"groupadd -f {g}")

    # Add user to groups (append!)
    inst.run(f"usermod -aG {','.join(extra_groups)} {user}")


def install_sources(inst: Installer, install_root: Path):
    inst.run(
        f'rsync -av --delete '
        f'--include "scripts/***" '
        f'--include "src/***" '
        f'--exclude="*" '
        f'"{REPO_ROOT}/" "{install_root}/"'
    )


def install_config(inst: Installer, install_root: Path):
    config_target = install_root / "config"

    for ini in CONFIG_DIR.glob("*.ini"):
        dst = config_target / ini.name
        inst.copy_file(ini, dst)


def install_udev(inst: Installer):
    for rule in CONFIG_DIR.glob("*.rules"):
        dst = UDEV_DIR / rule.name
        inst.copy_file(rule, dst)

    inst.run("udevadm control --reload-rules")
    inst.run("udevadm trigger")


def install_service(inst: Installer, config, install_root: Path):
    controller = config["startup"]["controller"]

    if controller == "mcp.py":
        service_name = "sky360-mcp.service"
    else:
        service_name = "sky360-main-controller.service"

    src = CONFIG_DIR / service_name
    dst = SYSTEMD_DIR / service_name

    inst.copy_file(src, dst)

    inst.run("systemctl daemon-reload")
    inst.run(f"systemctl enable {service_name}")
    

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--host", type=str, help="Override hostname (for testing)")
    args = parser.parse_args()

    inst = Installer(dry_run=args.dry_run)

    hostname = args.host if args.host else get_hostname()
    inst.log(f"[INFO] Host: {hostname}")

    config, ini_path = load_config(hostname)
    
    install_root = Path(config["global"]["install_root"])

    user = config["global"].get("user", "s360")
    group = config["global"].get("group", "s360")

    ensure_user(inst, user, group)
    install_sources(inst, install_root)
    install_config(inst, install_root)
    install_udev(inst)
    install_service(inst, config, install_root)
    inst.log("[DONE]")

if __name__ == "__main__":
    main()
