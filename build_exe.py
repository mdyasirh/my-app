"""
LocalROSA – PyInstaller Build Script
======================================
Creates a standalone .exe for Windows (also works on Mac/Linux).

Usage:
    pip install pyinstaller
    python build_exe.py
"""

import os
import subprocess
import sys


def build():
    """Build standalone executable using PyInstaller."""
    print("Building LocalROSA standalone executable...")
    print("This may take several minutes on first build.\n")

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "LocalROSA",
        "--onedir",  # Use onedir for faster startup (vs onefile)
        "--windowed" if sys.platform == "win32" else "--console",
        "--add-data", f"static{os.pathsep}static",
        "--hidden-import", "mediapipe",
        "--hidden-import", "cv2",
        "--hidden-import", "gradio",
        "--hidden-import", "fpdf",
        "--collect-all", "mediapipe",
        "--collect-all", "gradio",
        "--noconfirm",
        "app.py",
    ]

    print(f"Running: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("Build complete!")
        print(f"Executable is in: dist/LocalROSA/")
        if sys.platform == "win32":
            print("Run: dist\\LocalROSA\\LocalROSA.exe")
        else:
            print("Run: dist/LocalROSA/LocalROSA")
        print("=" * 50)
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with error: {e}")
        print("Make sure PyInstaller is installed: pip install pyinstaller")
        sys.exit(1)
    except FileNotFoundError:
        print("\nPyInstaller not found. Install it with: pip install pyinstaller")
        sys.exit(1)


if __name__ == "__main__":
    build()
