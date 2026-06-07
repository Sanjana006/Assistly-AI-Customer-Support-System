import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import pandas as pd
from datasets import load_dataset, Dataset
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def build_knowledge_base():
    """
    Builds a ChromaDB vector store from the Bitext dataset.
    Each document = a (question, answer) pair from real support conversations.
    The agent searches this when answering general questions.
    """
    
    print("Loading dataset...")
    
    # Load from local if available, else HuggingFace
    df: pd.DataFrame
    try:
        df = pd.read_csv("data/raw/support_dataset.csv")
    except:
        dataset = load_dataset(
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            streaming=False
        )
        train_split: Dataset = dataset['train']  # type: ignore[assignment]
        df = train_split.to_pandas()  # type: ignore[assignment]
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
    
    # Delete collection if rebuilding
    try:
        client.delete_collection(Config.COLLECTION_NAME)
    except:
        pass
    
    # Create collection with default embeddings (no API key needed!)
    # Uses sentence-transformers locally
    collection = client.create_collection(
        name=Config.COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"}
    )
    
    # Use first 2000 rows as knowledge base (diverse, fast to build)
    kb_df = df.head(2000).copy()
    
    # Each document combines question + answer for better retrieval
    documents = []
    metadatas = []
    ids = []
    
    for idx, row in kb_df.iterrows():
        r: dict = row.to_dict()  # convert Series → dict for unambiguous type access
        doc = f"Customer question: {r['instruction']}\nSupport answer: {r['response']}"
        documents.append(doc)
        metadatas.append({
            "intent": str(r.get('intent', '')),
            "category": str(r.get('category', '')),
            "question": str(r['instruction']),
            "answer": str(r['response'])
        })
        ids.append(f"doc_{idx}")
    
    # Add in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],  # type: ignore[arg-type]
            ids=ids[i:i+batch_size]
        )
        if i % 500 == 0:
            print(f"  Added {i}/{len(documents)} documents...")
    
    print(f"\n✅ Knowledge base built: {len(documents)} Q&A pairs")
    print(f"   Path: {Config.CHROMA_PATH}")
    
    # Test it works
    results: dict = collection.query(  # type: ignore[assignment]
        query_texts=["where is my order"],
        n_results=2
    )
    print(f"\n🔍 Test query: 'where is my order'")
    print(f"   Top result intent: {results['metadatas'][0][0]['intent']}")  # type: ignore[index]

if __name__ == "__main__":
    build_knowledge_base()