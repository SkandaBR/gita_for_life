from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

client = PersistentClient(path="target/chroma_db")
col = client.get_collection("bhagavadgita_ch18")

model = SentenceTransformer("intfloat/multilingual-e5-large")
q = "ಸ್ವಧರ್ಮದ ಮಹತ್ವವೇನು?"
q_embed = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).tolist()

res = col.query(query_embeddings=q_embed, n_results=5)
for meta, doc in zip(res["metadatas"][0], res["documents"][0]):
    print(f"Chapter {meta.get('chapter')}, Verse {meta.get('verse')}")
    print(doc[:200], "\n")