from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
import os
_DOTENV = find_dotenv(usecwd=True)
load_dotenv(_DOTENV, override=True)
if __name__ == "__main__" or os.getenv("DEBUG_DOTENV") == "1":
    print(f"[.env] Cargado: {_DOTENV or 'NO ENCONTRADO'}")

import subprocess, sys

def main():
    app = os.path.join("src", "ui", "app.py")
    cmd = ["streamlit", "run", app, "--server.runOnSave=true"]
    print(" ".join(cmd))
    subprocess.call(cmd)

if __name__ == "__main__":
    main()
