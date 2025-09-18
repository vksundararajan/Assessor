# [Kestrel](https://github.com/vksundararajan/Kestrel)

A cybersecurity research assistant built using Retrieval-Augmented Generation (RAG). Kestrel enables efficient question-answering on cybersecurity knowledge by combining vector search with large language models, making research faster and more precise.

🔗 [<u>Watch a demo of Kestrel in action</u>](assets/demo.mov)

### [Features]()
- Choose reasoning strategy: Chain of Thought, ReAct, Self-Ask, or skip for direct answers
- Cybersecurity-focused RAG pipeline for precise and reliable responses
- Converts knowledge bases into vector embeddings for fast retrieval
- Retrieves relevant documents from an indexed vector database
- Builds contextual prompts by combining retrieved content, reasoning mode, and system instructions
- Generates finite, grounded answers using LLMs
- Modular design to swap datasets, vector databases, or LLM providers
- Stores responses and logs in the outputs/ directory for review

### [Repo Structure]()
```
📦 Kestrel
├─ .env.example
├─ .gitignore
├─ LICENSE
├─ README.md
├─ assets
│  └─ demo.mov
├─ code
│  ├─ config
│  │  ├─ prompt_config.yaml
│  │  └─ reasoning_config.yaml
│  ├─ paths.py
│  ├─ to_llm.py
│  ├─ to_vectordb.py
│  └─ utils.py
├─ data
│  └─ 1dfc5bee07ff.json
├─ outputs
│  ├─ .gitignore
│  └─ vector_db
│     └─ .gitignore
└─ requirements.txt
```

### [Installation]()
```
git clone https://github.com/vksundararajan/Kestrel.git
cd Kestrel
python3 -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
python code/to_llm.py
```

### [cybersecurity Research Queries]()

```
> What mitigation steps are recommended for an unauthenticated directory traversal vulnerability in a web appliance.
> List any Metasploit modules in the DB that mention CVE-2023-20198 and summarize what they exploit and which versions are affected.
> Which modules in the DB target Active Directory Certificate Services (ADCS) template misconfigurations or certificate issuance attacks? Provide module fullnames and short purpose.
> Explain the full TLS 1.3 handshake exchange (messages and purpose) and show how a server implements key update. Cite sources.
```

### [Kestrel will]()
_Kestrel_ first asks the user to choose a reasoning mode like CoT (Chain of Thought), ReAct, Self-Ask, or simply press Enter to continue without selecting. The user provides their query. Kestrel fetches the most relevant documents from the indexed vector database. The retrieved documents are combined with the chosen reasoning instructions and the system prompt. The complete prompt is sent to the LLM, which produces a grounded, finite answer.

### [Roadmap]()
- Add support for multiple vector DBs (FAISS, Pinecone, Weaviate).
- Expand cybersecurity corpus (NVD, MITRE ATT&CK, CWE, etc.).
- Web interface for interactive research.
- Evaluation framework for measuring answer quality.

### [License]()
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
