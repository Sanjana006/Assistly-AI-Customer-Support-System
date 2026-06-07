from datasets import load_dataset, Dataset
import pandas as pd
import os

def download_and_prepare():
    """
    Downloads Bitext Customer Support dataset.
    27,000 rows, 27 intent categories.
    Columns: instruction (customer message), category, intent, response
    """
    
    print("Downloading Bitext Customer Support Dataset...")
    
    # Load from HuggingFace (free, no login needed)
    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        streaming=False  # ensures Dataset (not IterableDataset) is returned
    )
    
    # Convert to pandas — explicitly typed so Pyrefly knows .to_pandas() is valid
    train_split: Dataset = dataset['train']  # type: ignore[assignment]
    df: pd.DataFrame = train_split.to_pandas()  # type: ignore[assignment]

    print(f"Downloaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"\nIntent categories:")
    intent_col: pd.Series = df['intent']  # type: ignore[assignment]
    print(intent_col.value_counts().head(10))

    # Save locally so we don't download every time
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/support_dataset.csv", index=False)

    # Create a smaller test set (500 rows) for quick experiments
    test_df: pd.DataFrame = df.sample(n=500, random_state=42)  # type: ignore[assignment]
    test_df.to_csv("data/raw/test_set.csv", index=False)
    
    print("\n✅ Dataset saved to data/raw/support_dataset.csv")
    print("✅ Test set (500 samples) saved to data/raw/test_set.csv")
    
    return df

if __name__ == "__main__":
    download_and_prepare()