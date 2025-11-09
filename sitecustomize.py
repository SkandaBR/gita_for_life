# sitecustomize (src): redirect caches and keep import behavior when launching scripts under src
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
TARGET_DIR = os.path.join(PROJECT_ROOT, "target")
PYCACHE_DIR = os.path.join(TARGET_DIR, "__pycache__")

os.makedirs(PYCACHE_DIR, exist_ok=True)
os.environ["PYTHONPYCACHEPREFIX"] = PYCACHE_DIR

try:
    sys.pycache_prefix = PYCACHE_DIR
except Exception:
    pass