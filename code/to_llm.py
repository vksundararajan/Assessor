import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from paths import OUTPUT_DIR, VECTOR_DB_DIR, PROMPT_CONFIG_PATH, REASONING_CONFIG_PATH
from utils import load_yaml_config, save_text_to_file
from to_vectordb import initialize_db, embed_documents

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_chat")


class LLMClient:
  """
  mall wrapper to call a supported LLM provider.
  Currently supports:
  - Google Gemini via langchain_google_genai (default) 
  """

  def __init__(self, provider: str = None, model: str | None = None, temperature: float = 0.1):
    self.provider = provider or os.getenv("LLM_PROVIDER", "google")
    self.model = model
    self.temperature = temperature
    self._client = None
    # lazy imports

  def _init_google(self):
    """
    Initialize Google Gemini LLM client.
    Requires the environment variable GOOGLE_API_KEY to be set.    
    """
    try:
      from langchain_google_genai import ChatGoogleGenerativeAI

      api_key = os.getenv("GOOGLE_API_KEY")
      model = self.model or os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
      self._client = ChatGoogleGenerativeAI(model=model, temperature=self.temperature, api_key=api_key)
      return True
    except Exception as e:
      logger.debug("Google LLM init failed: %s", e)
      return False

  def invoke(self, prompt: str) -> str:
    """
    Invoke the LLM with the given prompt.
    Args:
      prompt (str): The prompt to send to the LLM.
    Returns:
      str: The LLM's response.
    Raises:
      RuntimeError: If no supported LLM provider is available or invocation fails.
    """
    if self._client is None:
      if self.provider == "google":
        ok = self._init_google()
        if not ok:
          if not self._init_groq():
            raise RuntimeError("No supported LLM providers available (Google/Groq).")
      else:
        if not self._init_google():
          raise RuntimeError("No supported LLM providers available (Google).")

    try:
      if self._client.__class__.__name__.lower().startswith("chatgoogle"):
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [SystemMessage(content="You are a security assistant."), HumanMessage(content=prompt)]
        resp = self._client.invoke(messages)
        return getattr(resp, "content", str(resp))
      else:
        resp = self._client.invoke(prompt)
        return getattr(resp, "content", str(resp))
    except Exception as e:
      logger.exception("LLM invocation failed: %s", e)
      raise


def get_collection(collection_name: str = "exploit_db"):
  """
  Get or initialize the ChromaDB collection for the vector database.
  Args:
    collection_name (str): Name of the collection to create or load. Defaults to "exploit_db".
  Returns:
    chromadb.Collection: The initialized or loaded collection.
  """
  coll = initialize_db(persist_directory=VECTOR_DB_DIR, collection_name=collection_name, delete_existing=False)
  return coll


def retrieve_relevant_documents(collection, query: str, n_results: int = 6, threshold: float = 0.5) -> List[Dict[str, Any]]:
  """
  Retrieve relevant documents from the vector database based on the query.
  Args:
      collection (chromadb.Collection): The ChromaDB collection to query.
      query (str): The user's query string.
      n_results (int): Number of top results to retrieve. Defaults to 6.
      threshold (float): Distance threshold for filtering results. Defaults to 0.5.
  Returns:
      list: A list of dictionaries containing the retrieved documents and their metadata.
  """
  logger.info("Embedding query")
  q_embed = embed_documents([query])[0]

  logger.info("Querying vector DB for top %d results", n_results)
  res = collection.query(query_embeddings=[q_embed], n_results=n_results, include=["documents", "metadatas", "distances"]) 

  docs = res.get("documents", [[]])[0]
  metas = res.get("metadatas", [[]])[0]
  dists = res.get("distances", [[]])[0]
  ids = res.get("ids", [[]])[0]

  results = []
  for i, doc in enumerate(docs):
    distance = dists[i] if i < len(dists) else None
    if distance is None or distance <= threshold:
      results.append({
        "id": ids[i] if i < len(ids) else None,
        "document": doc,
        "metadata": metas[i] if i < len(metas) else {},
        "distance": distance,
      })

  return results


def build_rag_prompt(prompt_template: str, question: str, retrieved: List[Dict[str, Any]], max_chars_per_doc: int = 1200) -> str:
  """
  Build the RAG prompt by combining the retrieved documents and the user's question.
  Args:
    prompt_template (str): The prompt template with placeholders for context and question.
    question (str): The user's question.
    retrieved (list): List of retrieved documents with metadata.
    max_chars_per_doc (int): Maximum characters to include from each document. Defaults to 1200.
  Returns:
    str: The final prompt to send to the LLM.
  """
  context_parts = []
  for r in retrieved:
    doc = r.get("document", "")
    snippet = doc.strip()
    if len(snippet) > max_chars_per_doc:
      snippet = snippet[:max_chars_per_doc] + "..."
    md = r.get("metadata", {})
    src = md.get("fullname") or md.get("exploit_id") or md.get("name")
    context_parts.append(f"Source: {src}\n{snippet}\n")

  context = "\n---\n".join(context_parts)

  if prompt_template:
    try:
      return prompt_template.format(context=context, question=question)
    except Exception:
      # Template mismatch; fall back
      pass

  # default prompt
  return (
    "You are a security research assistant. Use the context below from a vulnerability knowledge-base to answer the question. "
    "Cite the Source (exploit fullname or id) for statements. If the answer is uncertain, say so and reference the snippets.\n\n"
    f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer concisely and include sources."
  )


def persist_chat(turn: Dict[str, Any], file_path: str = os.path.join(OUTPUT_DIR, "chat_history.json")):
  """
  Persist the chat turn to a JSON file.
  Args:
    turn (dict): The chat turn containing the query, retrieved ids, metadata, and answer.
    file_path (str): Path to the JSON file to append the chat history. Defaults to "outputs/chat_history.json".
  Returns:
    None
  """
  try:
    if os.path.exists(file_path):
      with open(file_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    else:
      history = []
  except Exception:
    history = []

  history.append(turn)
  try:
    save_text_to_file(json.dumps(history, indent=2, ensure_ascii=False), file_path)
  except Exception as e:
    logger.warning("Failed to persist chat history: %s", e)


def interactive_chat(collection_name: str = "exploit_db"):
  """
  Start an interactive chat session with the RAG assistant.
  Args:
    collection_name (str): Name of the ChromaDB collection to use. Defaults to "exploit_db".
  Returns:
    None
  """
  collection = get_collection(collection_name)

  ### Load system prompt if available
  try:
    cfg = load_yaml_config(PROMPT_CONFIG_PATH)
    sys_prompt_entry = cfg.get("cybersecurity_research_assistant_prompt") or {}
    system_role_text = None
    role = sys_prompt_entry.get("role") or sys_prompt_entry.get("description") or "You are a helpful assistant."
    constraints = "\n".join(sys_prompt_entry.get("output_constraints", [])) if sys_prompt_entry.get("output_constraints") else ""
    output_fmt = sys_prompt_entry.get("output_format")
    examples = sys_prompt_entry.get("examples")
    prompt_parts = [
      role, "\n\n",
      constraints, "\n\n",
      "### Output Format\n", 
      json.dumps(output_fmt, indent=2), "\n\n",
      "### Examples\n",
      json.dumps(examples, indent=2)
    ]
    system_role_text = "".join(prompt_parts)
  except Exception:
    prompt_template = None
    system_role_text = None

  ### Load reasoning strategies if available
  reasoning_cfg = None
  reasoning_choice = None
  try:
    reasoning_cfg = load_yaml_config(REASONING_CONFIG_PATH)
    strategies = reasoning_cfg.get("reasoning_strategies", {})
    if strategies:
      # let user pick via interactive prompt
      print("Available reasoning strategies:")
      for i, k in enumerate(strategies.keys()):
        print(f"  {i+1}) {k}")
      try:
        pick = input("Choose reasoning strategy number (or press Enter for none): ")
        if pick:
          pick_idx = int(pick) - 1
          reasoning_choice = list(strategies.keys())[pick_idx]
      except Exception:
        reasoning_choice = None
  except Exception:
    reasoning_cfg = None
    reasoning_choice = None

  ### Initialize LLM client
  llm_client = LLMClient(provider="google")

  print("RAG assistant (type 'exit' to quit, 'config' to change threshold/topk)")
  print("\n---\n")
  
  n_results = 6  # Number of top documents to retrieve. Increase for more context, decrease for less noise. (Range: 1-20)
  threshold = 0.5  # Distance threshold for filtering. Lower is stricter (more relevant), higher is more lenient. (Range: 0.0-1.0)

  ### Chat loop has begun
  while True:
    try:
      query = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
      print("\nExiting")
      break

    if not query:
      continue
    if query.lower() in ("exit", "quit"):
      break
    if query.lower() == "config":
      try:
        n_results = int(input(f"Top K (current {n_results}): ") or n_results)
        threshold = float(input(f"Distance threshold (current {threshold}): ") or threshold)
      except Exception:
        print("Invalid input, keeping previous config")
      continue

    retrieved = retrieve_relevant_documents(collection, query, n_results=n_results, threshold=threshold)

    logger.info("Retrieved %d items", len(retrieved))
    for i, r in enumerate(retrieved):
      logger.info("%d) id=%s distance=%.4f source=%s", i + 1, r.get("id"), r.get("distance") or 0.0, r.get("metadata", {}).get("fullname") or r.get("metadata", {}).get("exploit_id"))

    # Default prompt
    main_prompt = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
    if reasoning_choice and reasoning_cfg:
      reasoning_text = reasoning_cfg.get("reasoning_strategies", {}).get(reasoning_choice)
      if reasoning_text:
        main_prompt = ("""{context}\n\nReasoning Strategy: """ + reasoning_choice + "\n" + reasoning_text + "\n\n{question}")

    # Combine the system role with the main prompt
    effective_template = system_role_text + "\n\n" + main_prompt

    prompt = build_rag_prompt(effective_template, question=query, retrieved=retrieved)

    try:
      answer = llm_client.invoke(prompt)
    except Exception as e:
      print("LLM error:", e)
      continue

    print("\nAssistant:\n")
    print(answer)
    print("\n---\n")

    # persist
    turn = {
      "query": query,
      "retrieved_ids": [r.get("id") for r in retrieved],
      "retrieved_metadata": [r.get("metadata") for r in retrieved],
      "answer": answer,
    }
    persist_chat(turn)


if __name__ == "__main__":
  interactive_chat()
