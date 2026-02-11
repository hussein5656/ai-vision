"""Bootstrap the Python environment and system dependencies with OS-aware defaults."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


def run_command(command: Iterable[str]):
    command_list = list(command)
    print(f"[setup] $ {' '.join(command_list)}")
    subprocess.run(command_list, check=True)


def ensure_virtualenv() -> Path:
    if not VENV_DIR.exists():
        print("[setup] Creating virtual environment (.venv)")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    python_bin = (
        VENV_DIR / "Scripts" / "python.exe"
        if os.name == "nt"
        else VENV_DIR / "bin" / "python3"
    )
    if not python_bin.exists():
        python_bin = (
            VENV_DIR / "bin" / "python"
        )
    if not python_bin.exists():
        raise RuntimeError("Virtual environment python executable not found")
    return python_bin


def install_requirements(python_bin: Path):
    if not REQUIREMENTS_FILE.exists():
        print("[setup] requirements.txt not found, skipping Python deps")
        return
    print("[setup] Installing Python dependencies")
    run_command([str(python_bin), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])


def ensure_system_dependencies():
    system = platform.system().lower()
    if system == "darwin":
        _ensure_macos_packages()
    elif system == "linux":
        _ensure_linux_packages()
    else:
        print(f"[setup] No extra system dependencies needed for {system}")


def _ensure_macos_packages():
    missing = _missing_commands(["ffmpeg", "gst-launch-1.0"])
    if not missing:
        return
    if not shutil.which("brew"):
        print("[setup] Homebrew not found. Install from https://brew.sh/ to auto-install: " + ", ".join(missing))
        return
    brew_packages = {
        "ffmpeg": "ffmpeg",
        "gst-launch-1.0": "gstreamer"
    }
    for cmd in missing:
        pkg = brew_packages.get(cmd)
        if pkg:
            try:
                run_command(["brew", "install", pkg])
            except subprocess.CalledProcessError as exc:
                print(f"[setup] Failed to install {pkg} via brew: {exc}")


def _ensure_linux_packages():
    missing = _missing_commands(["ffmpeg", "gst-launch-1.0"])
    if not missing:
        return
    manager = _detect_linux_package_manager()
    if not manager:
        print("[setup] No supported package manager detected. Install manually: " + ", ".join(missing))
        return
    commands: List[List[str]] = []
    if manager in {"apt", "apt-get"}:
        pkg_list = ["ffmpeg", "gstreamer1.0-libav", "gstreamer1.0-plugins-good", "gstreamer1.0-plugins-base"]
        commands.append(["sudo", manager, "update"])
        commands.append(["sudo", manager, "install", "-y", *pkg_list])
    elif manager == "dnf":
        pkg_list = ["ffmpeg", "gstreamer1", "gstreamer1-plugins-base", "gstreamer1-plugins-good"]
        commands.append(["sudo", "dnf", "install", "-y", *pkg_list])
    elif manager == "yum":
        pkg_list = ["ffmpeg", "gstreamer1", "gstreamer1-plugins-base", "gstreamer1-plugins-good"]
        commands.append(["sudo", "yum", "install", "-y", *pkg_list])
    elif manager == "pacman":
        pkg_list = ["ffmpeg", "gst-libav", "gst-plugins-good", "gst-plugins-base"]
        commands.append(["sudo", "pacman", "-Sy", "--noconfirm", *pkg_list])
    elif manager == "zypper":
        pkg_list = ["ffmpeg", "gstreamer", "gstreamer-plugins-base", "gstreamer-plugins-good"]
        commands.append(["sudo", "zypper", "install", "-y", *pkg_list])
    else:
        print("[setup] Unsupported package manager. Install manually: " + ", ".join(missing))
        return
    for cmd in commands:
        try:
            run_command(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"[setup] Failed to run {' '.join(cmd)}: {exc}")
            break


def _missing_commands(commands: Iterable[str]):
    return [cmd for cmd in commands if shutil.which(cmd) is None]


def _detect_linux_package_manager() -> str | None:
    for manager in ("apt", "apt-get", "dnf", "yum", "pacman", "zypper"):
        if shutil.which(manager):
            return manager
    return None


def main():
    python_bin = ensure_virtualenv()
    install_requirements(python_bin)
    ensure_system_dependencies()
    print("[setup] Done. Activate the virtualenv: " + (".venv\\Scripts\\activate" if os.name == "nt" else "source .venv/bin/activate"))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[setup] Command failed: {exc}")
        sys.exit(exc.returncode)
    except Exception as exc:
        print(f"[setup] Error: {exc}")
        sys.exit(1)
