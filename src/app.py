# Module-level logging and helpers
# Redirect __pycache__ to target early in process startup
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
TARGET_DIR = os.environ.get("TARGET_DIR", os.path.join(PROJECT_ROOT, "target"))
PYCACHE_DIR = os.path.join(TARGET_DIR, "__pycache__")
os.makedirs(PYCACHE_DIR, exist_ok=True)

# Fallback: only set if not already handled by sitecustomize
if not getattr(sys, "pycache_prefix", None):
    os.environ["PYTHONPYCACHEPREFIX"] = PYCACHE_DIR
    try:
        sys.pycache_prefix = PYCACHE_DIR
    except Exception:
        pass

import streamlit as st
import os
from bhagavadgita_rag import BhagavadGitaRAG
from gtts import gTTS
import tempfile
import base64
import logging
from datetime import datetime, date, time as dt_time
import uuid
import time
import json

import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
TARGET_DIR = os.path.join(PROJECT_ROOT, "target")

current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = TARGET_DIR

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SLOW_THRESHOLD_MS = 5000

def now_ms():
    return int(time.time() * 1000)

def get_system_metrics():
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        net = psutil.net_io_counters()
        return {
            "cpu_pct": cpu,
            "mem_pct": vm.percent,
            "mem_used_mb": round(vm.used / (1024 * 1024), 2),
            "net_sent_mb": round(net.bytes_sent / (1024 * 1024), 2),
            "net_recv_mb": round(net.bytes_recv / (1024 * 1024), 2),
        }
    except Exception:
        return {
            "cpu_pct": None,
            "mem_pct": None,
            "mem_used_mb": None,
            "net_sent_mb": None,
            "net_recv_mb": None,
        }

def log_event(event, level=logging.INFO, **fields):
    payload = {"event": event, "ts": datetime.utcnow().isoformat(), **fields}
    try:
        logger.log(level, json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.log(level, f"{event} | {fields}")

def verify_runtime_paths():
    import sys
    chroma_target = os.path.join(TARGET_DIR, "chroma_db")
    src_chroma = os.path.join(os.path.dirname(__file__), "chroma_db")
    log_event(
        "runtime_paths_check",
        pycache_prefix=getattr(sys, "pycache_prefix", None),
        expected_pycache=PYCACHE_DIR,
        chroma_target_dir=chroma_target,
        src_chroma_exists=os.path.isdir(src_chroma),
    )

verify_runtime_paths()

@st.cache_resource(show_spinner=False)
def init_chroma():
    try:
        start = now_ms()
        from chromadb import PersistentClient
        from chromadb.config import Settings
        persist_dir = os.path.join(TARGET_DIR, "chroma_db")
        os.makedirs(persist_dir, exist_ok=True)
        client = PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        collection = client.get_or_create_collection("chat_history")
        dur = now_ms() - start
        # Runtime path validation
        import sys
        db_file = os.path.join(persist_dir, "chroma.sqlite3")
        log_event(
            "chroma_init_success",
            duration_ms=dur,
            persist_dir=persist_dir,
            db_exists=os.path.isfile(db_file),
            pycache_prefix=getattr(sys, "pycache_prefix", None),
            chroma_ready=True
        )
        if dur > SLOW_THRESHOLD_MS:
            log_event("chroma_init_slow", level=logging.WARNING, duration_ms=dur)
        return client, collection
    except Exception as e:
        log_event("chroma_init_error", level=logging.ERROR, error=str(e), chroma_ready=False)
        try:
            st.warning(f"Chroma initialization error: {e}")
        except Exception:
            pass
        return None, None

@st.cache_resource(show_spinner=False)
def get_embed_model():
    try:
        start = now_ms()
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("intfloat/multilingual-e5-large")
        dur = now_ms() - start
        log_event("embed_model_loaded", duration_ms=dur, device="cpu", model="intfloat/multilingual-e5-large")
        if dur > SLOW_THRESHOLD_MS:
            log_event("embed_model_load_slow", level=logging.WARNING, duration_ms=dur)
        return model
    except Exception as e:
        log_event("embed_model_error", level=logging.ERROR, error=str(e))
        try:
            st.error(f"Embedding model load error: {e}")
        except Exception:
            pass
        return None

def ensure_services_ready():
    try:
        # Lazy load services; avoid heavy work at import-time
        if "chroma_client" not in st.session_state or "chat_col" not in st.session_state or st.session_state.chat_col is None:
            st.session_state.chroma_client, st.session_state.chat_col = init_chroma()
        if "embed_model" not in st.session_state or st.session_state.embed_model is None:
            st.session_state.embed_model = get_embed_model()
    except Exception as e:
        log_event("ensure_services_error", level=logging.ERROR, error=str(e))
        try:
            st.warning(f"Service initialization error: {e}")
        except Exception:
            pass

def persist_chat(chat_col, model, query_text, results, language):
    try:
        if chat_col is None or model is None or not query_text:
            log_event("persist_chat_skipped", reason="missing_col_or_model_or_query")
            return
        # Build combined document
        doc = query_text
        if results:
            top = results[0].get("verse", {})
            doc = f"Query ({language}): {query_text}\nTop verse [Ch {top.get('chapter','')}, V {top.get('verse','')}]: {top.get('text','')}"
        ids = [str(uuid.uuid4())]

        t0 = now_ms()
        embeds = model.encode([doc], convert_to_numpy=True, normalize_embeddings=True).tolist()
        t1 = now_ms()
        chat_col.add(ids=ids, documents=[doc], metadatas=[{
            "type": "chat",
            "language": language,
            "timestamp": datetime.utcnow().isoformat(),
            "query": query_text
        }], embeddings=embeds)
        t2 = now_ms()

        log_event(
            "persist_chat_success",
            request_id=ids[0],
            embed_ms=t1 - t0,
            chroma_add_ms=t2 - t1,
            total_ms=t2 - t0
        )
        if (t2 - t0) > SLOW_THRESHOLD_MS:
            log_event("persist_chat_slow", level=logging.WARNING, total_ms=t2 - t0)
    except Exception as e:
        log_event("persist_chat_error", level=logging.ERROR, error=str(e))
        try:
            st.warning(f"Failed to persist chat: {e}")
        except Exception:
            pass

# Set page config
st.set_page_config(
    page_title="ಭಗವದ್ಗೀತೆ ಸರ್ಚ್ | Bhagavad Gita Search",
    page_icon="🕉️",
    layout="wide"
)

# Language selection
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Language selection radio buttons
language = st.radio(
    "Select Language / ಭಾಷೆ ಆಯ್ಕೆ ಮಾಡಿ:",
    options=['English', 'Kannada'],
    index=0 if st.session_state.language == 'English' else 1,
    horizontal=True
)

# Update session state
st.session_state.language = language

# Language-specific content dictionary
LANG_CONTENT = {
    'English': {
        'title': '🕉️ Bhagavad Gita Knowledge Repository',
        'subtitle': 'Semantic Search and Information Retrieval from Bhagavad Gita',
        'data_loaded': '✅ Bhagavad Gita data loaded successfully',
        'example_queries_title': '📝 Example Queries',
        'example_queries': [
            'What is the true difference between Tyaga and Sannyasa?',
            'How do the three Gunas influence our actions and knowledge?',
            "What is the importance of performing one's own duty (Svadharma)?",
            'How can one attain liberation (Moksha) from the bondage of karma?',
            'What is the final advice given by Sri Krishna to Arjuna?',
        ],
        'query_placeholder': 'Enter your question here',
        'query_example': 'Example: What did Krishna say about karma?',
        'results_slider': 'How many results to show?',
        'search_button': '🔍 Search',
        'searching': 'Searching...', 
        'results_title': '📖 Results',
        'chapter': 'Chapter',
        'verse': 'Verse',
        'similarity': 'Similarity',
        'original_verse': 'Original Verse',
        'translation': 'English Translation',
        'generating_audio': 'Generating audio...',
        'generating_original_audio': 'Generating original verse audio...',
        'generating_translation_audio': 'Generating translation audio...'
    },
    'Kannada': {
        'title': '🕉️ ಭಗವದ್ಗೀತೆ ಜ್ಞಾನ ಭಂಡಾರ',
        'subtitle': 'ಭಗವದ್ಗೀತೆಯಲ್ಲಿ ಸಂದರ್ಭೋಚಿತ ಮಾಹಿತಿ ಪುನರ್ಪ್ರಾಪ್ತಿ',
        'data_loaded': '✅ ಭಗವದ್ಗೀತೆಯ ಮಾಹಿತಿ ಈಗ ಲಭ್ಯ',
        'example_queries_title': '📝 ಉದಾಹರಣೆ ಪ್ರಶ್ನೆಗಳು',
        'example_queries': [
            'ತ್ಯಾಗ ಮತ್ತು ಸಂನ್ಯಾಸದ ನಡುವಿನ ನಿಜವಾದ ವ್ಯತ್ಯಾಸವೇನು?', # What is the true difference between Tyaga and Sannyasa?
            'ಮೂರು ಗುಣಗಳು ನಮ್ಮ ಕರ್ಮ ಮತ್ತು ಜ್ಞಾನದ ಮೇಲೆ ಹೇಗೆ ಪ್ರಭಾವ ಬೀರುತ್ತವೆ?', # How do the three Gunas influence our actions and knowledge?
            'ಸ್ವಧರ್ಮವನ್ನು ಆಚರಿಸುವುದರ ಮಹತ್ವವೇನು?', # What is the importance of performing one's own duty (Svadharma)?
            'ಕರ್ಮ ಬಂಧನದಿಂದ ಮುಕ್ತರಾಗಿ ಮೋಕ್ಷವನ್ನು ಸಾಧಿಸುವುದು ಹೇಗೆ?', # How can one attain liberation (Moksha) from the bondage of karma?
            'ಅರ್ಜುನನಿಗೆ ಶ್ರೀಕೃಷ್ಣನು ನೀಡಿದ ಅಂತಿಮ ಉಪದೇಶವೇನು?', # What is the final advice given by Sri Krishna to Arjuna?
        ],
        'query_placeholder': 'ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಇಲ್ಲಿ ಬರೆಯಿರಿ',
        'query_example': 'ಉದಾ: ಕರ್ಮದ ಬಗ್ಗೆ ಕೃಷ್ಣನು ಏನು ಹೇಳಿದನು?',
        'results_slider': 'ಎಷ್ಟು ಫಲಿತಾಂಶಗಳನ್ನು ತೋರಿಸಬೇಕು?',
        'search_button': '🔍 ಹುಡುಕಿ',
        'searching': 'ಹುಡುಕುತ್ತಿದೆ...', 
        'results_title': '📖 ಫಲಿತಾಂಶಗಳು',
        'chapter': 'ಅಧ್ಯಾಯ',
        'verse': 'ಶ್ತು',
        'similarity': 'ಹೊಂದಾಣಿಕೆ',
        'original_verse': 'ಮೂಲ ಶ್ತು',
        'translation': 'ಕನ್ನಡ ಅರ್ಥ',
        'generating_audio': 'ಧ್ವನಿ ತಯಾರಿಸುತ್ತಿದೆ...',
        'generating_original_audio': 'ಮೂಲ ಶ್ತುದ ಧ್ವನಿ ತಯಾರಿಸುತ್ತಿದೆ...', 
        'generating_translation_audio': 'ಅನುವಾದದ ಧ್ವನಿ ತಯಾರಿಸುತ್ತಿದೆ...'
    }
}

# Get current language content
content = LANG_CONTENT[st.session_state.language]

# Initialize Chroma and embedding model in session
if "chroma_client" not in st.session_state or "chat_col" not in st.session_state:
    try:
        st.session_state.chroma_client, st.session_state.chat_col = init_chroma()
    except Exception as e:
        log_event("chroma_session_init_error", level=logging.ERROR, error=str(e))
        st.session_state.chroma_client = None
        st.session_state.chat_col = None
        st.warning(f"Chroma unavailable: {e}")

if "embed_model" not in st.session_state or st.session_state.embed_model is None:
    st.session_state.embed_model = get_embed_model()

# Remove heavy init at import-time; call ensure_services_ready() right before usage
# (delete any previous top-level calls to init_chroma/get_embed_model here)

# Sidebar: Chat history semantic search
with st.sidebar:
    st.markdown("### 🔎 Search Chat History")
    hist_query = st.text_input("Search past chats")
    hist_k = st.slider("Results", 1, 10, 3)
    if hist_query:
        ensure_services_ready()
        hs_start = now_ms()
        net_before = get_system_metrics()
        try:
            if st.session_state.embed_model is None or st.session_state.chat_col is None:
                log_event("history_search_skipped", reason="missing_model_or_collection")
                st.warning("Embedding model or Chroma not initialized.")
            else:
                q_emb = st.session_state.embed_model.encode(
                    [hist_query], convert_to_numpy=True, normalize_embeddings=True
                ).tolist()
                enc_ms = now_ms() - hs_start
                res = st.session_state.chat_col.query(query_embeddings=q_emb, n_results=hist_k)
                hs_end = now_ms()
                net_after = get_system_metrics()
                log_event(
                    "history_search_done",
                    duration_ms=hs_end - hs_start,
                    encode_ms=enc_ms,
                    chroma_query_ms=hs_end - hs_start - enc_ms,
                    params={"query": hist_query, "k": hist_k},
                    metrics={"before": net_before, "after": net_after},
                    results_count=len(res.get("ids", [[]])[0]) if res and res.get("ids") else 0
                )
                if (hs_end - hs_start) > SLOW_THRESHOLD_MS:
                    log_event("history_search_slow", level=logging.WARNING, duration_ms=hs_end - hs_start)
                if res and res.get("documents"):
                    for i in range(len(res["ids"][0])):
                        st.write(res["documents"][0][i])
                        ts = res["metadatas"][0][i].get("timestamp")
                        st.caption(ts if ts else "")
                else:
                    st.info("No matches found in chat history.")
        except Exception as e:
            log_event("history_search_error", level=logging.ERROR, error=str(e))
            st.warning(f"Chat history search failed: {e}")

# Function to create audio player HTML
def get_audio_player_html(audio_path, label=""):
    audio_file = open(audio_path, 'rb')
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_file.close()
    return f'''
    <div class="audio-section">
        <span class="audio-icon">🔊</span>
        <div class="audio-label">{label}</div>
        <div class="audio-player">
            <audio controls>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
        </div>
    </div>
    '''

# Function to generate speech
def generate_speech(text, lang='kn'):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stMarkdown {
        text-align: center;
    }
    .audio-section {
        display: flex;
        align-items: center;
        gap: 10px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .audio-icon {
        font-size: 1.5em;
        color: #1f77b4;
    }
    .audio-label {
        font-weight: bold;
        color: #1f77b4;
        min-width: 120px;
    }
    .audio-player {
        flex-grow: 1;
    }
    .audio-player audio {
        width: 100%;
        margin: 5px 0;
    }
    .audio-player audio::-webkit-media-controls-panel {
        background-color: #ffffff;
    }
    .audio-player audio::-webkit-media-controls-play-button {
        background-color: #1f77b4;
        border-radius: 50%;
    }
    .audio-player audio::-webkit-media-controls-timeline {
        background-color: #e6e9ef;
        border-radius: 25px;
        margin: 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title(content['title'])
st.markdown(f"""
### {content['subtitle']}
---
""")


@st.cache_resource
def load_rag_system():
    """Load the RAG system with caching"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    json_path = os.path.join(data_dir, "bhagavadgita_Chapter_2.json")
    return BhagavadGitaRAG(json_path)

# Initialize session state for query
if 'query' not in st.session_state:
    st.session_state.query = ""

# Initialize the RAG system
try:
    rag = load_rag_system()
    st.success(content['data_loaded'])

    # Example queries in sidebar
    with st.sidebar:
        st.markdown(f"### {content['example_queries_title']}")
        for example_query in content['example_queries']:
            if st.button(example_query):
                st.session_state.query = example_query

    # Ensure query is initialized in session state
    if 'query' not in st.session_state:
        st.session_state.query = ""

    # Search interface
    with st.form(key="search_form"):
        # Search query input
        query = st.text_input(
            content['query_placeholder'],
            value=st.session_state.query,
            placeholder=content['query_example']
        )

        # Number of results slider
        num_results = st.slider(
            content['results_slider'],
            min_value=1,
            max_value=10,
            value=3
        )

        # Search button
        search_button = st.form_submit_button(content['search_button'])

    # Perform search
    with st.spinner(content['searching']):
        # Only run search on submit with a non-empty query
        if search_button and query.strip():
            search_query = query.strip()

            req_id = str(uuid.uuid4())
            start_ms = now_ms()
            metrics_before = get_system_metrics()
            log_event(
                "request_start",
                request_id=req_id,
                params={"query": search_query, "language": st.session_state.language, "top_k": num_results},
                metrics=metrics_before,
                chroma_ready=st.session_state.chat_col is not None
            )

            # Use cached RAG instance and ensure services are ready
            rag_init_start = now_ms()
            rag = load_rag_system()
            rag_init_ms = now_ms() - rag_init_start
            ensure_services_ready()

            retrieve_start = now_ms()
            results = rag.retrieve(search_query, top_k=num_results)
            retrieve_ms = now_ms() - retrieve_start

            try:
                persist_chat(
                    st.session_state.chat_col,
                    st.session_state.embed_model,
                    search_query,
                    results,
                    st.session_state.language
                )
            except Exception as e:
                log_event("persist_chat_wrapper_error", level=logging.ERROR, error=str(e))
                st.warning(f"Could not persist chat: {e}")

            end_ms = now_ms()
            metrics_after = get_system_metrics()
            total_ms = end_ms - start_ms

            log_event(
                "request_end",
                request_id=req_id,
                total_ms=total_ms,
                rag_init_ms=rag_init_ms,
                retrieve_ms=retrieve_ms,
                metrics={"before": metrics_before, "after": metrics_after}
            )
            if total_ms > SLOW_THRESHOLD_MS:
                log_event("request_slow", level=logging.WARNING, request_id=req_id, total_ms=total_ms)

            # Display results
            st.markdown(f"### {content['results_title']}")
            for i, result in enumerate(results, 1):
                verse = result['verse']
                similarity = result['similarity']
                with st.expander(f"{content['chapter']} {verse.get('chapter', 'Unknown')}, {content['verse']} {verse.get('verse', 'Unknown')} ({content['similarity']}: {similarity:.2%})"):
                    st.markdown(f"**{content['original_verse']}:**")
                    st.markdown(f"*{verse.get('text', '')}*")
                    with st.spinner(content['generating_original_audio']):
                        original_audio_path = generate_speech(verse.get('text', ''))
                        if original_audio_path:
                            st.markdown(get_audio_player_html(original_audio_path, content['original_verse']), unsafe_allow_html=True)
                            try:
                                os.unlink(original_audio_path)
                            except:
                                pass
                    if 'translation' in verse:
                        st.markdown(f"**{content['translation']}:**")
                        if st.session_state.language == 'English':
                            english_translation = verse.get('english_translation', verse['translation'])
                            st.markdown(english_translation)
                            with st.spinner(content['generating_translation_audio']):
                                translation_audio_path = generate_speech(english_translation, lang='en')
                                if translation_audio_path:
                                    st.markdown(get_audio_player_html(translation_audio_path, content['translation']), unsafe_allow_html=True)
                                    try:
                                        os.unlink(translation_audio_path)
                                    except:
                                        pass
                        else:
                            st.markdown(verse['translation'])
                            with st.spinner(content['generating_translation_audio']):
                                translation_audio_path = generate_speech(verse['translation'])
                                if translation_audio_path:
                                    st.markdown(get_audio_player_html(translation_audio_path, content['translation']), unsafe_allow_html=True)
                                    try:
                                        os.unlink(translation_audio_path)
                                    except:
                                        pass
                    st.progress(similarity)

            # Clear the session state query after successful search
            st.session_state.query = ""

except Exception as e:
    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit and Sentence Transformers</p>
</div>
""", unsafe_allow_html=True)
