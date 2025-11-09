# Bootstrap: ensure caches and ChromaDB persist under ../target before imports
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
TARGET_DIR = os.path.join(PROJECT_ROOT, "target")
PYCACHE_DIR = os.path.join(TARGET_DIR, "__pycache__")
os.makedirs(PYCACHE_DIR, exist_ok=True)
os.environ["PYTHONPYCACHEPREFIX"] = PYCACHE_DIR
import json
from typing import Any, Dict, List
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

def load_json(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_verses(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    verses: List[Dict[str, Any]] = []

    # Handle nested structure with chapters -> verses
    if "chapters" in data:
        for chapter in data["chapters"]:
            if "verses" in chapter:
                for verse in chapter["verses"]:
                    v = dict(verse) if isinstance(verse, dict) else {"text": str(verse)}
                    v["chapter"] = chapter.get("chapter_number", "")
                    verses.append(v)

    # Fallback: flattened structures
    if not verses:
        for chapter_num, chapter_data in data.items():
            if isinstance(chapter_data, dict) and "verses" in chapter_data:
                for verse_num, verse_text in chapter_data["verses"].items():
                    verses.append({
                        "chapter": chapter_num,
                        "verse": verse_num,
                        "text": verse_text
                    })

    return verses

def build_documents(verses: List[Dict[str, Any]]) -> Dict[str, List]:
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for v in verses:
        chapter = str(v.get("chapter", ""))
        verse_num = str(v.get("verse", ""))
        text = v.get("text", "")
        translation = v.get("translation", "")
        english_translation = v.get("english_translation", "")

        # Stable ID for consistent dedup
        doc_id = f"ch{chapter}_v{verse_num}" if chapter and verse_num else f"auto_{len(ids)}"
        ids.append(doc_id)

        # Embed combined content for stronger multilingual semantic coverage
        combined = "\n\n".join(
            [part for part in [
                text,
                f"Translation: {translation}" if translation else "",
                f"English: {english_translation}" if english_translation else ""
            ] if part]
        )
        documents.append(combined if combined else text)

        # Preserve all relevant fields in metadata for display
        metadatas.append({
            "chapter": chapter,
            "verse": verse_num,
            "text": text,
            "translation": translation,
            "english_translation": english_translation
        })

    return {"ids": ids, "documents": documents, "metadatas": metadatas}

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    # Pick the first JSON file found in the data folder
    json_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    json_path = os.path.join(data_dir, json_files[0])
    print(f"Loading JSON: {json_path}")
    data = load_json(json_path)
    verses = extract_verses(data)
    if not verses:
        raise RuntimeError("No verses found in JSON.")

    payload = build_documents(verses)
    print(f"Prepared {len(payload['documents'])} documents.")

    # Multilingual embeddings (default used in repo)
    model_name = "intfloat/multilingual-e5-large"
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(payload["documents"], convert_to_numpy=True, normalize_embeddings=True).tolist()
    print("Embeddings created.")

    # Persist to Chroma (DuckDB + Parquet)
    persist_dir = os.path.join(current_dir, "..", "target", "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    client: PersistentClient = PersistentClient(path=persist_dir)
    collection_name = "bhagavadgita_ch18"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"[verify] PYTHONPYCACHEPREFIX={os.environ.get('PYTHONPYCACHEPREFIX')}")
    print(f"[verify] Chroma persist_dir={persist_dir} (exists={os.path.isdir(persist_dir)})")

    # Clear existing collection documents (optional safety)
    try:
        existing = collection.count()
        if existing:
            print(f"Clearing existing {existing} docs from collection '{collection_name}'.")
            # Chroma doesn’t have a direct truncate; re-create collection for a clean slate
            client.delete_collection(collection_name)
            collection = client.get_or_create_collection(name=collection_name)
    except Exception:
        pass

    print(f"Adding {len(payload['documents'])} docs to collection '{collection_name}'.")
    collection.add(
        ids=payload["ids"],
        documents=payload["documents"],
        metadatas=payload["metadatas"],
        embeddings=embeddings
    )
    print(f"Persisted Chroma DB at: {persist_dir}")

    # Quick verification: query using precomputed query embeddings
    test_query = "ತ್ಯಾಗ ಮತ್ತು ಸಂನ್ಯಾಸದ ವ್ಯತ್ಯಾಸ"
    q_embed = model.encode([test_query], convert_to_numpy=True, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_embed, n_results=3)
    print("Sample query results (top 3):")
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        sim_doc = results["documents"][0][i][:120].replace("\n", " ")
        print(f"- Chapter {meta.get('chapter')}, Verse {meta.get('verse')}: {sim_doc}...")

if __name__ == "__main__":
    main()