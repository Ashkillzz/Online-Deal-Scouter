"""
Seed ChromaDB Vector Store with Curated Products for RAG Valuation.
Pulls products from Hugging Face, computes MiniLM embeddings, and indexes them into ChromaDB.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATASET_NAME = "Ashkillzz/curated_product_reviews"
FALLBACK_DATASET = "ed-donner/items_lite"
DB_PATH = "products_vectorstore"
COLLECTION_NAME = "products"
BATCH_SIZE = 100
MAX_ITEMS = 1500


def seed():
    print(f"Connecting to ChromaDB at: {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    current_count = collection.count()
    print(f"Current document count in '{COLLECTION_NAME}' collection: {current_count}")

    if current_count >= MAX_ITEMS:
        print(f"Vector store already contains {current_count} items (>= {MAX_ITEMS}). Seeding skipped.")
        return

    print(f"Loading dataset: {DATASET_NAME}...")
    try:
        ds = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"Could not load {DATASET_NAME} ({e}), falling back to {FALLBACK_DATASET}...")
        ds = load_dataset(FALLBACK_DATASET, split="train")

    total_to_seed = min(MAX_ITEMS - current_count, len(ds))
    subset = ds.select(range(total_to_seed))
    print(f"Seeding {total_to_seed} reference items into ChromaDB...")

    print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for i in tqdm(range(0, total_to_seed, BATCH_SIZE), desc="Indexing embeddings"):
        batch = subset.select(range(i, min(i + BATCH_SIZE, total_to_seed)))
        
        # Support text, summary, prompt, or title
        if 'text' in batch.column_names:
            texts = [str(t) for t in batch['text']]
        elif 'summary' in batch.column_names:
            texts = [str(s) if s else str(t) for s, t in zip(batch['summary'], batch['title'])]
        elif 'prompt' in batch.column_names:
            texts = [str(p) for p in batch['prompt']]
        else:
            texts = [str(t) for t in batch['title']]

        prices = [float(p) for p in batch['price']]
        categories = (
            [str(c) if c else "Retail" for c in batch['category']]
            if 'category' in batch.column_names
            else ["Retail"] * len(texts)
        )

        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False).astype(float).tolist()
        ids = [f"prod_{current_count + i + j}" for j in range(len(texts))]
        metadatas = [
            {"price": price, "category": cat}
            for price, cat in zip(prices, categories)
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print(f"\nSeeding complete! Total items now in ChromaDB '{COLLECTION_NAME}': {collection.count()}", flush=True)


if __name__ == "__main__":
    seed()
