import os
import sys
import py_compile
import importlib

# Ensure src is importable when running tests from project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def test_pycache_prefix_set():
    expected = os.path.join(PROJECT_ROOT, "target", "__pycache__")
    assert os.path.isdir(expected), "target/__pycache__ directory should exist"
    assert getattr(sys, "pycache_prefix", None) == expected

def test_compilation_writes_into_target_pycache():
    mod = importlib.import_module("bhagavadgita_rag")
    cfile = py_compile.compile(mod.__file__, doraise=True)
    assert cfile.startswith(sys.pycache_prefix), f"pyc should be under {sys.pycache_prefix}, got {cfile}"
    assert os.path.exists(cfile)