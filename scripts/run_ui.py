from __future__ import annotations
import os, subprocess, sys

def main():
    app = os.path.join("src", "ui", "app.py")
    cmd = ["streamlit", "run", app, "--server.runOnSave=true"]
    print(" ".join(cmd))
    subprocess.call(cmd)

if __name__ == "__main__":
    main()
