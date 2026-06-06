import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import os

def prepare_database():
    """
    Loads raw CSV data, enriches it with sentiment scores, 
    and indexes the content into a persistent ChromaDB vector store.
    """
    # 1. Define Paths
    data_path = "./data/cabin_reports.csv"
    db_path = "./data/cabin_vector_db"
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} records found.")

    # 2. Sentiment Enrichment
    # Calculating polarity scores (-1 to 1) for the analytics dashboard
    df['sentiment_score'] = df['report_content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df.to_csv(data_path, index=False) 
    print("Sentiment analysis completed. CSV updated with scores.")

    # 3. Initialize Embedding Model
    # Using 'all-MiniLM-L6-v2' for optimal speed/performance balance
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Initialize Vector DB (ChromaDB)
    client = chromadb.PersistentClient(path=db_path)
    
    # Re-create collection to ensure a clean start
    try:
        client.delete_collection("cabin_reports")
    except Exception:
        pass
    collection = client.create_collection(name="cabin_reports")

    # 5. Batch Indexing with Unique IDs and Upsert
    documents = df['report_content'].tolist()
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    metadatas = df[['category', 'route', 'urgency']].to_dict(orient='records')
    
    # Creating guaranteed unique IDs using row index to prevent DuplicateIDError
    ids = [f"ID-{i}" for i in range(len(df))]

    # Using 'upsert' to handle potential ID overlaps gracefully
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Process complete. {collection.count()} records indexed in ChromaDB.")

if __name__ == "__main__":
    prepare_database()