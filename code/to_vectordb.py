import os
import sys
import torch
import shutil
import chromadb
from paths import VECTOR_DB_DIR
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import load_all_exploits


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


def chunk_exploit(
    exploit: str, chunk_size: int = 1000, overlap: int = 200
) -> list[str]:
  """
  Chunk the given exploit text into smaller pieces using RecursiveCharacterTextSplitter.
  Args:
      exploit (str): The exploit text to chunk.
      chunk_size (int): The size of each chunk. Defaults to 1000.
      overlap (int): The number of overlapping characters between chunks. Defaults to 200.
  Returns:
      list[str]: A list of chunked text pieces.
  """
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=overlap
  )

  return text_splitter.split_text(exploit)


def embed_documents(documents: list[str]) -> list[list[float]]:
  """
  Generate embeddings for a list of documents using HuggingFaceEmbeddings.
  Args:
      documents (list[str]): List of document texts to embed.
  Returns:
      list[list[float]]: A list of embeddings corresponding to the input documents.
  """
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  mode = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
  embeddings = mode.embed_documents(documents)
  return embeddings


def insert_exploits_to_db(collection: chromadb.Collection, exploits: list[str]):
  """
  Insert chunked exploits into the ChromaDB collection with embeddings.
  Args:
      collection (chromadb.Collection): The ChromaDB collection to insert into.
      exploits (list[str]): List of exploit texts to insert.
  Returns:
      None
  """
  next_id = collection.count()

  for exploit in exploits:
    chunk_exploit = chunk_exploit(exploit)
    embeddings = embed_documents(chunk_exploit)
    ids = list(range(next_id, next_id + len(chunk_exploit)))
    ids = [f"exploit_{i}" for i in ids]
    collection.add(
      documents=chunk_exploit,
      embeddings=embeddings,
      ids=ids
    )
    next_id += len(chunk_exploit)


def main():
  collection = initialize_db(
    persist_directory=VECTOR_DB_DIR
    collection_name="exploit_db",
    delete_existing=True
  ) 

  exploits = load_all_exploits()
  insert_exploits_to_db(collection, exploits)

  print(f"Database setup complete with {collection.count()} exploits.")
  

if __name__ == "__main__":
  sys.exit(main())