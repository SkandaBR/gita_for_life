# Bootstrap: redirect pycache to ../target early when directly executed
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
TARGET_DIR = os.environ.get("PROJECT_ROOT", os.path.join(PROJECT_ROOT, "target"))
PYCACHE_DIR = os.path.join(TARGET_DIR, "__pycache__")
try:
    os.makedirs(PYCACHE_DIR, exist_ok=True)
except Exception as e:
    # If target is inaccessible, log and continue with default cache behavior
    print(f"[warn] Unable to create {PYCACHE_DIR}: {e}")

# Fallback: only set if not already handled by sitecustomize
if not getattr(sys, "pycache_prefix", None):
    os.environ["PYTHONPYCACHEPREFIX"] = PYCACHE_DIR
    try:
        # Python 3.8+ supports runtime assignment
        sys.pycache_prefix = PYCACHE_DIR
    except Exception:
        pass

# Top-level imports and logger setup
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import glob
import chromadb
from chromadb.utils import embedding_functions
import glob
import logging
import sys

# Initialize a module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger.setLevel(logging.INFO)
logger.info(f"bhagavadgita_rag module loaded from {__file__}")

def _log_pycache_setup():
    """
    Log the active pycache configuration and expected cache path for this module.
    This does not force compilation and preserves standard bytecode behavior.
    """
    try:
        import importlib.util
        expected_cache = importlib.util.cache_from_source(__file__)
    except Exception as e:
        expected_cache = f"<unavailable: {e}>"
    logger.info(
        "pycache_setup",
        extra={
            "pycache_prefix": getattr(sys, "pycache_prefix", None),
            "env_PYTHONPYCACHEPREFIX": os.environ.get("PYTHONPYCACHEPREFIX"),
            "expected_cache_path": expected_cache,
            "target_accessible": os.path.isdir(PYCACHE_DIR),
        }
    )

_log_pycache_setup()

class BhagavadGitaRAG:
    """
    Retrieval Augmented Generation system for Bhagavad Gita in Kannada.
    This class loads Kannada JSON data of Bhagavad Gita, indexes it using embeddings,
    and provides retrieval capabilities.
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 is used for embeddings.
    """
    
    def __init__(self, json_path: str, model_name: str = 'intfloat/multilingual-e5-large'):
        """
        Initialize the RAG system.

        Args:
            json_path: Path to the Kannada JSON file of Bhagavad Gita
            model_name: Name of the sentence transformer model to use for embeddings
        """
        logger.info(f"BhagavadGitaRAG.__init__ enter | json_path={json_path} | model_name={model_name}")
        self.json_path = json_path
        self.model = SentenceTransformer(model_name)
        self.verses = []
        self.embeddings = None
        self.load_data()
        self.create_embeddings()
        logger.info(f"BhagavadGitaRAG.__init__ exit | verses_loaded={len(self.verses)} | embeddings_ready={self.embeddings is not None}")
    
    def load_data(self) -> None:
        """
        Load the Bhagavad Gita data from JSON file.
        """
        logger.info(f"load_data enter | json_path={self.json_path}")
        if not os.path.exists(self.json_path):
            logger.error(f"JSON file not found at {self.json_path}")
            raise FileNotFoundError(f"JSON file not found at {self.json_path}")
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"JSON loaded | type={type(data)}")
            
            if isinstance(data, list):
                self.verses = data
                logger.debug("Detected list structure for verses.")
            elif isinstance(data, dict) and 'verses' in data:
                self.verses = data['verses']
                logger.debug("Detected dict with 'verses' key.")
            else:
                logger.debug("Unknown structure; extracting verses via _extract_verses.")
                self.verses = self._extract_verses(data)
                
            logger.info(f"load_data exit | verses_count={len(self.verses)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file at {self.json_path} | error={e}")
            raise ValueError(f"Invalid JSON file at {self.json_path}") from e
    
    def _extract_verses(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract verses from a nested JSON structure.
        
        Args:
            data: The JSON data
            
        Returns:
            A list of verse dictionaries
        """
        logger.info("_extract_verses enter")
        verses = []
        
        if 'chapters' in data:
            logger.debug("Found 'chapters' in JSON; iterating for verses.")
            for chapter in data['chapters']:
                if 'verses' in chapter:
                    for verse in chapter['verses']:
                        if isinstance(verse, dict):
                            verse['chapter'] = chapter.get('chapter_number', '')
                        else:
                            verse = {'chapter': chapter.get('chapter_number', ''), 'text': str(verse)}
                        verses.append(verse)
        
        if not verses:
            logger.debug("No verses found under 'chapters'; trying flattened structure.")
            for chapter_num, chapter_data in data.items():
                if isinstance(chapter_data, dict) and 'verses' in chapter_data:
                    for verse_num, verse_text in chapter_data['verses'].items():
                        verses.append({
                            'chapter': chapter_num,
                            'verse': verse_num,
                            'text': verse_text
                        })
        
        logger.info(f"_extract_verses exit | verses_extracted={len(verses)}")
        return verses
    
    def create_embeddings(self) -> None:
        """
        Create embeddings for all verses.
        """
        logger.info("create_embeddings enter")
        if not self.verses:
            logger.warning("No verses to embed")
            return
        
        texts = []
        for verse in self.verses:
            if isinstance(verse, dict) and 'text' in verse:
                texts.append(verse['text'])
            elif isinstance(verse, str):
                texts.append(verse)
            else:
                text = self._extract_text(verse)
                texts.append(text if text else str(verse))
        logger.debug(f"Prepared {len(texts)} texts for embedding.")
        
        self.embeddings = self.model.encode(texts)
        logger.info(f"create_embeddings exit | embeddings_count={len(texts)} | embeddings_type={type(self.embeddings)}")
    
    def _extract_text(self, verse: Dict[str, Any]) -> str:
        """
        Extract text from a verse dictionary with unknown structure.
        
        Args:
            verse: A verse dictionary
            
        Returns:
            The extracted text
        """
        # Try common field names for text
        for field in ['text', 'verse_text', 'content', 'kannada', 'translation']:
            if field in verse:
                return verse[field]
        
        # If no text field found, convert the whole verse to string
        return str(verse)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most relevant verses for the query.

        Args:
            query: The query text
            top_k: Number of top results to return

        Returns:
            List of top_k most relevant verses with similarity scores
        """
        logger.info(f"retrieve enter | query={query} | top_k={top_k}")
        if not self.verses or self.embeddings is None:
            logger.error("No verses or embeddings available")
            raise ValueError("No verses or embeddings available")

        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'verse': self.verses[idx],
                'similarity': float(similarities[idx])
            })
        logger.info(f"retrieve exit | results_count={len(results)} | top_indices={list(map(int, top_indices))}")
        return results


def main():
    """
    Example usage of the BhagavadGitaRAG class.
    """
    logger.info("main enter")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "..", "data")
        abs_data_dir = os.path.abspath(data_dir)
        logger.info(f"Resolved data_dir={abs_data_dir} | cwd={os.getcwd()}")

        json_files = glob.glob(os.path.join(abs_data_dir, "*.json"))
        logger.info(f"Found JSON files: {json_files}")
        if not json_files:
            logger.error(f"No JSON file found in {abs_data_dir}")
            raise FileNotFoundError(f"No JSON file found in {abs_data_dir}")

        # Prefer Chapter 18 if multiple JSONs exist
        preferred = [p for p in json_files if "Chapter_18" in os.path.basename(p)]
        json_path = preferred[0] if preferred else json_files[0]
        logger.info(f"Selected JSON file: {json_path}")

        rag = BhagavadGitaRAG(json_path)
        query = "ಕರ್ಮದ ಬಗ್ಗೆ ಕೃಷ್ಣನು ಏನು ಹೇಳಿದನು?"
        logger.info(f"Executing query: {query}")
        results = rag.retrieve(query, top_k=3)

        print(f"\nQuery: {query}")
        print("\nRelevant verses:")
        for i, result in enumerate(results, 1):
            verse = result['verse']
            similarity = result['similarity']
            chapter = verse.get('chapter', 'Unknown')
            verse_num = verse.get('verse', 'Unknown')
            text = verse.get('text', str(verse))
            print(f"\n{i}. Chapter {chapter}, Verse {verse_num} (Similarity: {similarity:.4f})")
            print(f"   {text}")

        logger.info("main exit | success")
    except Exception as e:
        logger.error(f"main error | {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()