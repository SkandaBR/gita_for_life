#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating how to use the Bhagavad Gita RAG system with Kannada text.
"""

import os
import sys

# Redirect Python bytecode caches to target/__pycache__.
# - Uses PYTHONPYCACHEPREFIX (Python 3.8+) and sys.pycache_prefix (3.8+).
# - On Python < 3.8: logs a warning and preserves default behavior.
# - Fails gracefully if target is inaccessible, without breaking imports or execution.
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    TARGET_DIR = os.path.join(PROJECT_ROOT, "target")
    PYCACHE_DIR = os.path.join(TARGET_DIR, "__pycache__")
    os.makedirs(PYCACHE_DIR, exist_ok=True)
    # Environment variable (effective from Python 3.8+)
    os.environ["PYTHONPYCACHEPREFIX"] = PYCACHE_DIR
    # Runtime prefix (Python 3.8+); older versions will skip
    if sys.version_info >= (3, 8):
        try:
            sys.pycache_prefix = PYCACHE_DIR
        except Exception as e:
            print(f"[warn] sys.pycache_prefix not set: {e}")
    else:
        print("[warn] Python < 3.8: PYTHONPYCACHEPREFIX/sys.pycache_prefix unavailable; using default __pycache__.")
except Exception as e:
    print(f"[warn] Could not initialize target/__pycache__: {e}")
from bhagavadgita_rag import BhagavadGitaRAG


def main():
    # Get the current directory
    PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
    # Use the project's data directory; do not change functional behavior
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    # Path to the sample JSON file
    json_path = os.path.join(DATA_DIR, "bhagavadgita_Chapter_18.json")
    
    print("Initializing Bhagavad Gita RAG system...")
    # Initialize the RAG system
    rag = BhagavadGitaRAG(json_path)
    
    # Example queries in Kannada
    queries = [
        "ತ್ಯಾಗ ಮತ್ತು ಸಂನ್ಯಾಸದ ನಡುವಿನ ನಿಜವಾದ ವ್ಯತ್ಯಾಸವೇನು?", # What is the true difference between Tyaga and Sannyasa?
        "ಮೂರು ಗುಣಗಳು ನಮ್ಮ ಕರ್ಮ ಮತ್ತು ಜ್ಞಾನದ ಮೇಲೆ ಹೇಗೆ ಪ್ರಭಾವ ಬೀರುತ್ತವೆ?", # How do the three Gunas influence our actions and knowledge?
        "ಸ್ವಧರ್ಮವನ್ನು ಆಚರಿಸುವುದರ ಮಹತ್ವವೇನು?", # What is the importance of performing one's own duty (Svadharma)?
        "ಕರ್ಮ ಬಂಧನದಿಂದ ಮುಕ್ತರಾಗಿ ಮೋಕ್ಷವನ್ನು ಸಾಧಿಸುವುದು ಹೇಗೆ?", # How can one attain liberation (Moksha) from the bondage of karma?
        "ಅರ್ಜುನನಿಗೆ ಶ್ರೀಕೃಷ್ಣನು ನೀಡಿದ ಅಂತಿಮ ಉಪದೇಶವೇನು?", # What is the final advice given by Sri Krishna to Arjuna?
    ]
    
    # Process each query
    for query in queries:
        print("\n" + "-"*50)
        print(f"Query: {query}")
        
        # Retrieve relevant verses
        results = rag.retrieve(query, top_k=2)
        
        # Print results
        print("\nRelevant verses:")
        for i, result in enumerate(results, 1):
            verse = result['verse']
            similarity = result['similarity']
            
            # Extract verse information
            chapter = verse.get('chapter', 'Unknown')
            verse_num = verse.get('verse', 'Unknown')
            text = verse.get('text', str(verse))
            translation = verse.get('translation', '')
            
            print(f"\n{i}. Chapter {chapter}, Verse {verse_num} (Similarity: {similarity:.4f})")
            print(f"   Text: {text}")
            if translation:
                print(f"   Translation: {translation}")


if __name__ == "__main__":
    main()