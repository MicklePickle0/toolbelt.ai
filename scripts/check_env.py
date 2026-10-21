"""Report whether the local environment is ready for the pipeline.

Usage: python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import shutil
import sys

CORE = ["numpy", "scipy", "cv2", "yaml", "pandas", "duckdb"]
OPTIONAL = ["torch", "torchvision", "rfdetr", "supervision", "rtmlib", "onnxruntime",
            "easyocr", "fastapi", "uvicorn", "streamlit", "pytest", "ruff"]


def check_module(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except ImportError:
        return "MISSING"
    return getattr(mod, "__version__", "ok")


def main() -> int:
    ok = True
    py = sys.version_info
    py_ok = (3, 11) <= py[:2] <= (3, 12)
    ok &= py_ok
    print(f"python {py.major}.{py.minor}.{py.micro}: {'ok' if py_ok else 'WANT 3.11 or 3.12'}")

    for tool in ("ffmpeg", "ffprobe", "git"):
        found = shutil.which(tool)
        ok &= found is not None
        print(f"{tool}: {found or 'MISSING'}")

    print("\ncore:")
    for name in CORE:
        status = check_module(name)
        ok &= status != "MISSING"
        print(f"  {name}: {status}")

    print("\noptional (vision / ocr / app / dev):")
    for name in OPTIONAL:
        print(f"  {name}: {check_module(name)}")

    try:
        import torch

        print(f"\ntorch cuda: {torch.cuda.is_available()}  "
              f"mps: {torch.backends.mps.is_available()}")
    except ImportError:
        pass

    print("\nREADY" if ok else "\nNOT READY (see MISSING above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
