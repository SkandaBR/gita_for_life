# Bhagavad Gita RAG System with Bilingual Interface

This project implements a Retrieval Augmented Generation (RAG) system for the Bhagavad Gita with support for both Kannada and English languages. The system includes both a Python library for programmatic access and a Streamlit web application with an intuitive bilingual interface.

## Features

### Core RAG System
- Load and process Kannada JSON data of Bhagavad Gita
- Create embeddings for verses using a multilingual sentence transformer model
- Retrieve the most relevant verses for a given query using cosine similarity
- Support for various JSON structures of Bhagavad Gita data

### Streamlit Web Application
- **Bilingual Interface**: Complete language switching between English and Kannada
- **Dynamic Content**: All UI elements update based on language selection
- **Audio Support**: Text-to-speech functionality in both languages using gTTS
- **Interactive Search**: Real-time verse retrieval with similarity scoring
- **Example Queries**: Pre-built questions about key concepts like Dharma, Karma, and Moksha
- **Responsive Design**: Modern UI with expandable result sections



# Quick Start Guide

## New Folder Layout and Paths (after migration)

- Project root: `e:\Codebase\16thNov2025`
- App entry: `src\app.py`
- Core library: `src\bhagavadgita_rag.py`
- Example script: `src\example.py`
- Data files: `data\bhagavadgita_kannada_sample.json`, `data\bhagavadgita_Chapter_18.json`
- Vector store and utilities: `indexing\build_chroma_store.py`, `indexing\example_chroma_query.py`, `indexing\health_check_chroma.py`
- Requirements: `requirements.txt`
- Auxiliary: `auxillary\create_presentation.py`

## Running the Application


1. Create a virtual environment (Windows):
```bash
py -m venv .venv311
```

2. Activate the virtual environment:
```bash
.venv311\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Chroma DB Verification Workflow

- Build ChromaDB:
  - python indexing\\build_chroma_store.py
  - Confirms: prints “Persisted Chroma DB at: …\target\chroma_db”.
- Query via RAG:
  - python src\\bhagavadgita_rag.py
  - Confirms: prints top results with chapter/verse and text.

5. Run the Streamlit application:
```bash
streamlit run src\app.py
```

6. Access the application:
- Open your browser and go to `http://localhost:8501`

To launch the web interface at any time:
```bash
streamlit run src\app.py
```



## Migration Note (for existing users)

The repository was reorganized after migration from the previous Bitbucket layout. Here are the key path changes:

- `rag\app.py` → `src\app.py`
- `rag\bhagavadgita_rag.py` → `src\bhagavadgita_rag.py`
- `rag\bhagavadgita_kannada_sample.json` → `data\bhagavadgita_kannada_sample.json`
- `rag\requirements.txt` → `requirements.txt` (project root)
- `rag\README.md` → `README.md` (project root)
- New vector store utilities available under `indexing\`:
  - `indexing\build_chroma_store.py`
  - `indexing\example_chroma_query.py`
  - `indexing\health_check_chroma.py`

Actions for existing users:
- Update any scripts to import via `src` (e.g., add `src` to `PYTHONPATH` as shown above).
- Use `streamlit run src\app.py` instead of `streamlit run app.py`.
- Reinstall dependencies from the project root using `pip install -r requirements.txt`.
- Use data files from `data\` rather than the old `rag\` folder.

## File Structure (updated)

```
rag/
├── app.py                              # Streamlit web application
├── bhagavadgita_rag.py                 # Core RAG system implementation
├── bhagavadgita_kannada_sample.json    # Sample Bhagavad Gita data with translations
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## Language Support

The application supports:
- **Kannada**: Native language interface with complete translations
- **English**: Full English interface with translated content
- **Audio**: Text-to-speech in both languages using Google Text-to-Speech (gTTS)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve the system.

## License

This project is open-source and available for educational and research purposes.

# Chroma DB Integration Summary
# - Purpose: Chroma DB provides a persistent vector store for semantic retrieval of Bhagavad Gita content, enabling efficient top‑k search across Kannada/English text.
# - build_chroma_store.py: Ingests Chapter 18 JSON, constructs multilingual documents and metadata, computes normalized embeddings (SentenceTransformer multilingual‑e5‑large), and persists to collection `bhagavadgita_ch18` in `chroma_db`.
# - example_chroma_query.py: Encodes a Kannada query with the same embedding model, queries `bhagavadgita_ch18`, and prints top‑k results with verse metadata and document excerpts.
# - health_check_chroma.py: Executes a diagnostic write/read by adding a 1024‑dim test vector to `health_check`, querying it, and printing collection count and IDs with telemetry disabled.
# - Technical achievements: Stable IDs for deduplication, multilingual concatenation of text and translations for richer semantics, normalized embeddings, and reproducible local persistence (DuckDB/Parquet via Chroma).
# - Performance improvements: Precomputed embeddings and local persistence reduce query latency; embedding normalization improves cosine‑similarity stability; local client avoids network overhead.
# - Challenges overcome: Robust handling of heterogeneous JSON schemas and bilingual content alignment; safe collection reset logic to prevent stale data without native truncate.
# - Embedding method: SentenceTransformer `intfloat/multilingual-e5-large`, generating 1024‑dim normalized vectors for documents and queries.
# - Query capabilities: `collection.query` with configurable `n_results` (e.g., 3–5), returning documents and metadatas suitable for UI display or downstream processing.
# - Integration points: Shared `chroma_db` path reused across scripts; artifacts are ready for consumption by applications (e.g., `app.py`) or CLI utilities.


## Startup Behavior and Pycache Redirection

This project redirects Python’s compiled bytecode (`__pycache__`) away from `src\` into `target\__pycache__` using Python’s supported mechanisms. The change preserves normal import behavior, works across direct script runs, module imports, tests, and compilation, and is cross-platform.

### How Startup Works

- Python auto-imports `sitecustomize` at interpreter startup when it’s importable on `sys.path`.
- Two hooks are provided:
  - `sitecustomize.py` at project root: loads first for root-started processes; sets `PYTHONPYCACHEPREFIX` and `sys.pycache_prefix` to `target\__pycache__`, and prepends `src\` to `sys.path` to enable imports from root.
  - `src\sitecustomize.py`: ensures the same behavior when starting scripts directly under `src\`.
- Entry points (`src\app.py`, `src\bhagavadgita_rag.py`) include a safe fallback that sets the prefix only if not already set.
- Result: All `.pyc` files go under `target\__pycache__` following a parallel directory tree mirroring the original source paths.

Expected behavior:
- `sys.pycache_prefix` resolves to `g:\Codebase\16thNov2025\target\__pycache__`.
- Running `python src\*.py`, importing modules, running tests, or compiling produces `.pyc` under `target\__pycache__`.

### Verification Steps

Run these from the project root in your virtual environment (`.venv311`) to verify correct behavior:

- Check current prefix:
```bash
python -c "import sys; print(sys.pycache_prefix)"
```

- Direct script run:
```bash
python src\example.py
```

- Import and compile from root:
```bash
python -c "import sys, importlib, py_compile; m=importlib.import_module('bhagavadgita_rag'); print(sys.pycache_prefix); print(py_compile.compile(m.__file__, doraise=True))"
```

- Run tests (PowerShell-safe):
```bash
python -m unittest -v tests.test_pycache_redirection
```

- Manual compilation:
```bash
python -m py_compile src\bhagavadgita_rag.py
```

- Optional cleanup of old caches (PowerShell syntax):
```bash
Remove-Item -Recurse -Force src\__pycache__
```

### Verification Command Details

- `python -c "import sys; print(sys.pycache_prefix)"`  
  Prints the active pycache prefix or `None` if unset.
- `python -m unittest -v tests.test_pycache_redirection`  
  Runs the dedicated verification test with verbose output. Alternatives:
  - Discovery:
```bash
python -m unittest discover -s tests -p "test_pycache_redirection.py" -v
```
- Manual override (for diagnostics):  
  You can temporarily override the prefix with `-X pycache_prefix=...` to confirm behavior switches:
```bash
python -X pycache_prefix=target\__pycache__ -c "import sys; print(sys.pycache_prefix)"
```

### Expected Output Examples

Successful prefix check:
```plaintext
g:\Codebase\16thNov2025\target\__pycache__
```

Failed prefix check (sitecustomize not loaded or wrong start dir):
```plaintext
None
```

Successful import and compile:
```plaintext
g:\Codebase\16thNov2025\target\__pycache__
g:\Codebase\16thNov2025\target\__pycache__\Codebase\16thNov2025\src\bhagavadgita_rag.cpython-313.pyc
```

Tests (success):
```plaintext
test_compilation_writes_into_target_pycache (tests.test_pycache_redirection) ... ok
test_pycache_prefix_set (tests.test_pycache_redirection) ... ok

----------------------------------------------------------------------
Ran 2 tests in 0.XXs

OK
```

Tests (failure: 0 tests ran due to module path issues):
```plaintext
----------------------------------------------------------------------
Ran 0 tests in 0.000s

NO TESTS RAN
```

### Troubleshooting

- `sys.pycache_prefix` prints `None`:
  - Ensure you’re running from the project root.
  - Confirm `g:\Codebase\16thNov2025\sitecustomize.py` exists and is importable.
  - Check environment: `PYTHONDONTWRITEBYTECODE` should NOT be set.
  - Verify with:
```bash
python -c "import sys; print(sys.path)"
```

- `ModuleNotFoundError: No module named 'bhagavadgita_rag'`:
  - Root `sitecustomize` adds `src\` to `sys.path`. If you see this error:
    - Check `sys.path` contains `g:\Codebase\16thNov2025\src`.
    - Re-run from project root or explicitly add `src\`:
```bash
python -c "import sys, os; sys.path.insert(0, os.path.join(os.getcwd(), 'src')); import bhagavadgita_rag; print('ok')"
```

- Tests run zero or don’t discover modules:
  - Use the module path invocation:
```bash
python -m unittest -v tests.test_pycache_redirection
```
  - Or discovery:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

- Old `src\__pycache__` persists:
  - Remove it with PowerShell:
```bash
Remove-Item -Recurse -Force src\__pycache__
```

- Prefix path shows nested folders like `Codebase\` and `Users\` under `target\__pycache__`:
  - This is expected: Python mirrors source paths under the prefix. No action needed.

- Overriding behavior:
  - You can test different prefixes with:
```bash
python -X pycache_prefix=target\__pycache__ -c "import sys; print(sys.pycache_prefix)"
```
  - Environment-based override (session only):
```bash
$env:PYTHONPYCACHEPREFIX="g:\Codebase\16thNov2025\target\__pycache__"
python -c "import sys; print(sys.pycache_prefix)"
```

This setup keeps Python’s bytecode behavior intact, maintains cross-platform compatibility, and avoids breaking relative imports by only changing where `.pyc` files are stored.


