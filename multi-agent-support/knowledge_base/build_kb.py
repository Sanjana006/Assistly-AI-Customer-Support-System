import chromadb
import pandas as pd
from datasets import load_dataset
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Exact same import as kb_search.py — must match or queries return garbage
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction

EMBED_FN = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def build_knowledge_base():
    """
    Builds a ChromaDB vector store from the Bitext dataset.
    Each document = a (question, answer) pair from real support conversations.
    The resolver agent searches this when answering general questions.
    """

    print("Loading dataset...")

    # Load from local cache if already downloaded, else fetch from HuggingFace
    df: pd.DataFrame
    try:
        df = pd.read_csv("data/raw/support_dataset.csv")  # type: ignore[assignment]
        print(f"  Loaded from local cache: {len(df)} rows")
    except FileNotFoundError:
        print("  Downloading from HuggingFace (first time only)...")
        dataset = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
        )
        df = dataset["train"].to_pandas()  # type: ignore[assignment]
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/support_dataset.csv", index=False)
        print(f"  Saved {len(df)} rows to data/raw/support_dataset.csv")

    # Initialize ChromaDB with persistent storage
    os.makedirs(Config.CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=Config.CHROMA_PATH)

    # Delete existing collection so rebuild is clean
    try:
        client.delete_collection(Config.COLLECTION_NAME)
        print("  Deleted existing collection, rebuilding fresh...")
    except Exception:
        pass  # Collection didn't exist yet, that's fine

    # Create collection — SAME embedding function as kb_search.py
    collection = client.create_collection(
        name=Config.COLLECTION_NAME,
        embedding_function=EMBED_FN,  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"}
    )

    # Use first 2000 rows — diverse enough, fast to build
    kb_df = df.head(2000).copy()

    documents: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids:       list[str] = []

    for idx, row in kb_df.iterrows():
        # Combine question + answer into one document for better retrieval
        doc = (
            f"Customer question: {row['instruction']}\n"
            f"Support answer: {row['response']}"
        )
        documents.append(doc)
        metadatas.append({
            "intent":   str(row.get("intent",   "")),
            "category": str(row.get("category", "")),
            "question": str(row["instruction"]),
            "answer":   str(row["response"]),
        })
        ids.append(f"doc_{idx}")

    # Add in batches of 100 to avoid memory spikes
    batch_size = 100
    total = len(documents)

    for i in range(0, total, batch_size):
        collection.add(
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],  # type: ignore[arg-type]
            ids=ids[i : i + batch_size],
        )
        print(f"  Indexed {min(i + batch_size, total)}/{total} documents...")

    print(f"\n✅ Knowledge base built: {total} Q&A pairs")
    print(f"   Path: {Config.CHROMA_PATH}")
    print(f"   Collection: {Config.COLLECTION_NAME}")

    # Quick sanity check — make sure queries actually work
    print("\n🔍 Running sanity check...")
    test_queries = [
        "where is my order",
        "I want a refund",
        "how do I cancel my subscription",
    ]
    for q in test_queries:
        result = collection.query(query_texts=[q], n_results=1)
        top_intent = result["metadatas"][0][0]["intent"]  # type: ignore[index]
        print(f"   '{q}' → {top_intent}")

    print("\n✅ Sanity check passed. Knowledge base is ready.")


if __name__ == "__main__":
    build_knowledge_base()