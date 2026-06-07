import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Use ChromaDB's built-in default embedding function (no torch/sentence-transformers needed)
EMBED_FN = embedding_functions.DefaultEmbeddingFunction()

# Lazy globals — initialized on first use to avoid module-level crashes
_client = None
_collection = None

def _get_collection():
    """Return the ChromaDB collection, creating it lazily on first call."""
    global _client, _collection
    if _collection is None:
        try:
            _client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
            _collection = _client.get_or_create_collection(
                name=Config.COLLECTION_NAME,
                embedding_function=EMBED_FN,  # type: ignore[arg-type]
            )
        except Exception as e:
            # If the persistent client fails (e.g., incompatible on-disk format),
            # fall back to an ephemeral in-memory client so the rest of the app works.
            from loguru import logger
            logger.warning(f"ChromaDB PersistentClient failed ({e}), falling back to in-memory client.")
            _client = chromadb.EphemeralClient()
            _collection = _client.get_or_create_collection(
                name=Config.COLLECTION_NAME,
                embedding_function=EMBED_FN,  # type: ignore[arg-type]
            )
    return _collection

@tool
def search_knowledge_base(query: str, n_results: int = 3) -> list:  # type: ignore[type-arg]
    """
    Search the support knowledge base for relevant Q&A pairs.
    Use this for general questions about policies, shipping, returns, etc.
    Returns top N most relevant answers from past support conversations.
    """
    try:
        collection = _get_collection()
        # If the collection is empty, return nothing gracefully
        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count())
        )

        ids:       list = results["ids"][0]        # type: ignore[index]
        metadatas: list = results["metadatas"][0]  # type: ignore[index]
        distances: list = results["distances"][0]  # type: ignore[index]

        matches = []
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 1.0
            matches.append({
                "question":        str(meta.get("question", "")),
                "answer":          str(meta.get("answer", "")),
                "intent":          str(meta.get("intent", "")),
                "relevance_score": round(1 - dist, 4)
            })
        return matches
    except Exception as e:
        from loguru import logger
        logger.warning(f"Knowledge base search failed: {e}")
        return []
