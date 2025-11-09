from chromadb import PersistentClient
from chromadb.config import Settings

def main():
    import os
    from pathlib import Path

    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "..", "target")    

    client = PersistentClient(
        path=os.path.join(target_dir, "chroma_db"),
        settings=Settings(anonymized_telemetry=False)
    )
    # Reset health_check collection
    try:
        client.delete_collection("health_check")
    except Exception:
        pass
    col = client.get_or_create_collection("health_check")

    # Add a single test vector (1024 dims for multilingual-e5-large)
    col.add(
        ids=["hc1"],
        documents=["diagnostic"],
        metadatas=[{"source": "health_check"}],
        embeddings=[[0.0] * 1024]
    )

    print("Count:", col.count())
    res = col.query(query_embeddings=[[0.0] * 1024], n_results=1)
    print("Query OK:", res["ids"][0])

if __name__ == "__main__":
    main()