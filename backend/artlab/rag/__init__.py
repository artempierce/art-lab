"""
rag/ — the knowledge base: getting documents in (ingest), and finding the right pieces (search).

    embeddings.py  text → numbers that capture meaning, computed locally (free)
    web.py         safely fetch a web page and turn it into plain text
    ingest.py      load → split → scan for injections → embed → store   (a command you run)
    knowledge.py   the Chroma vector database, plus search over it
"""
