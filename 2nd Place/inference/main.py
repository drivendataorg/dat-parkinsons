"""Entrypoint with compact, data-free failure diagnostics."""
import sys
import traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tabicl_lib.zip"))
try:
    from importlib.metadata import version
    for package in ("torch", "numpy", "scipy", "scikit-learn", "pandas", "psutil", "huggingface-hub"):
        print("Runtime dependency: " + package + "=" + version(package), flush=True)
    from inference_main import main
    main()
except Exception as exc:
    print("Inference failure type: " + type(exc).__name__, flush=True)
    for frame in traceback.extract_tb(exc.__traceback__)[-12:]:
        print("Failure location: " + Path(frame.filename).name + ":" + str(frame.lineno) + " in " + frame.name, flush=True)
    sys.exit(1)
