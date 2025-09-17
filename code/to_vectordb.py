import os
import sys
import torch
import json
import shutil
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from utils import load_all_exploits
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

    try:
        client = chromadb.PersistentClient(path=persist_directory)
    except Exception:
        client = chromadb.Client(Settings(chroma_db_impl="chromadb.db.impl.sqlite.ChromaDB", persist_directory=persist_directory))

    try:
        collection = client.get_collection(name=collection_name)
        print(f"Loaded existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:batch_size": 10000,
            },
        )
        print(f"Created new collection: {collection_name}")

    print(f"ChromaDB initialized at {persist_directory}")
    return collection


def chunk_exploit_text(
  exploit: str, chunk_size: int = 1000, overlap: int = 200
) -> List[str]:
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


def embed_documents(documents: List[str]) -> List[List[float]]:
  """
  Generate embeddings for a list of documents using HuggingFaceEmbeddings.
  Args:
      documents (list[str]): List of document texts to embed.
  Returns:
      list[list[float]]: A list of embeddings corresponding to the input documents.
  """
  device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
  try:
    hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
  except Exception as e:
    # Likely missing sentence-transformers or related dependency — provide actionable instruction
    raise ImportError(
      "Could not initialize HuggingFaceEmbeddings, means 'sentence-transformers' package is not installed. "
      f"Original error: {e}"
    )

  embeddings = hf.embed_documents(documents)
  return embeddings


def insert_exploits_to_db(collection: chromadb.Collection, exploits: List[dict], dry_run: bool = False, batch_size: int = 5000):
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
    if isinstance(exploit, dict):
      exploit_text = json.dumps(exploit)
    else:
      exploit_text = str(exploit)

    chunks = chunk_exploit_text(exploit_text)
    ids = list(range(next_id, next_id + len(chunks)))
    ids = [f"exploit_{i}" for i in ids]

    if dry_run:
      print(f"Dry run: would add {len(chunks)} chunks to collection starting at id {next_id}")
    else:
      embeddings = embed_documents(chunks)
      
      for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        batch_docs = chunks[start:end]
        batch_embeds = embeddings[start:end]
        batch_ids = ids[start:end]
        collection.add(
          documents=batch_docs,
          embeddings=batch_embeds,
          ids=batch_ids
        )

    next_id += len(chunks)


def main():
  collection = initialize_db(
    persist_directory=VECTOR_DB_DIR,
    collection_name="exploit_db",
    delete_existing=True
  ) 

  exploits = load_all_exploits()
  insert_exploits_to_db(collection, exploits)

  print(f"Database setup complete with {collection.count()} exploits.")
  

if __name__ == "__main__":
  sys.exit(main())