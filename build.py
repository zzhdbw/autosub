"""Build standalone AutoSub GUI application with PyInstaller."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
DIST = ROOT / "dist"
BUILD = ROOT / "build"


def main() -> None:
    # Clean previous builds
    for d in [DIST, BUILD]:
        if d.exists():
            subprocess.run(["rm", "-rf", str(d)], check=True)

    is_darwin = platform.system() == "Darwin"
    is_win32 = platform.system() == "Windows"

    pyinstaller = shutil.which("pyinstaller")
    if not pyinstaller:
        print("Error: pyinstaller not found. Run: pip install pyinstaller", flush=True)
        sys.exit(1)

    cmd = [
        pyinstaller,
        "--clean",
        "--noconfirm",
        "--name", "AutoSub",
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
    ]

    if is_darwin:
        cmd += [
            "--windowed",
            "--target-architecture", "arm64" if platform.machine() == "arm64" else "x86_64",
        ]
    elif is_win32:
        cmd += ["--windowed", "--uac-admin"]

    cmd += ["--collect-data", "customtkinter"]
    cmd += [str(SRC / "autosub" / "gui.py")]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    if is_darwin:
        app = DIST / "AutoSub.app"
        if app.exists():
            print(f"\n\xe2\x9c\x93 Built: {app}")
            print(f"  Run: open {app}")
    elif is_win32:
        exe = DIST / "AutoSub" / "AutoSub.exe"
        if exe.exists():
            print(f"\n\xe2\x9c\x93 Built: {exe}")
    else:
        print(f"\n\xe2\x9c\x93 Built: {DIST / 'AutoSub'}")


if __name__ == "__main__":
    main()
