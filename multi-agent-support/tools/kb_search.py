import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Use ChromaDB's built-in default embedding function (no torch/sentence-transformers needed)
EMBED_FN = embedding_functions.DefaultEmbeddingFunction()

_client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=Config.COLLECTION_NAME,
    embedding_function=EMBED_FN  # type: ignore[arg-type]
)

@tool
def search_knowledge_base(query: str, n_results: int = 3) -> list:  # type: ignore[type-arg]
    """
    Search the support knowledge base for relevant Q&A pairs.
    Use this for general questions about policies, shipping, returns, etc.
    Returns top N most relevant answers from past support conversations.
    """
    results = _collection.query(
        query_texts=[query],
        n_results=n_results
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
