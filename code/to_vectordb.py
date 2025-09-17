import os
import sys
import shutil
import chromadb
from paths import VECTOR_DB_DIR

def initialize_db(
    persist_directory, 
    collection_name, 
    delete_existing=False
  ) -> chromadb.Collection:
    """
    Initialize or load a ChromaDB collection.

    Args:
        persist_directory (str): Directory to persist the database - "./exploit_db".
        collection_name (str): Name of the collection to create or load - "exploit_db".
        delete_existing (bool): If True, deletes existing data in the directory.
    
    Returns:
        chromadb.Collection: The initialized or loaded collection.
    """
    if os.path.exists(persist_directory) and delete_existing:
      shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
       collection = client.get_collection(name=collection_name)
       print(f"Loaded existing collection: {collection_name}")
    except Exception:
       collection = client.create_collection(
          name=collection_name,
          metadata={
            "hnsw:space": "cosine",
            "hnsw:batch_size": 10000,
          }
        )
       print(f"Created new collection: {collection_name}")
    
    print(f"ChromaDB initialized at {persist_directory}")
    return collection

def main():
  collection = initialize_db(
    persist_directory=VECTOR_DB_DIR
    collection_name="exploit_db",
    delete_existing=True
  ) 
  
if __name__ == "__main__":
  sys.exit(main())