# rag/indexer.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
KNOWLEDGE_BASE_PATH: str = os.getenv("KNOWLEDGE_BASE_PATH", "./knowledge_base")
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"


def _build_embeddings() -> HuggingFaceEmbeddings:
    """Initialise the local sentence-transformer embedding model (CPU-safe)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},       # CPU-safe; change to 'cuda' if GPU available
        encode_kwargs={"normalize_embeddings": True},
    )


def build_or_load_vectorstore() -> Chroma:
    """
    Build the ChromaDB vectorstore from the knowledge base or load from disk.

    Behaviour:
    - If CHROMA_PATH exists and is non-empty, load the persisted index (fast, ~2 sec).
    - Otherwise, load every .txt file under KNOWLEDGE_BASE_PATH recursively,
      chunk and embed them, persist to CHROMA_PATH, and return the new store.

    Returns
    -------
    Chroma
        A LangChain Chroma vectorstore instance backed by all-MiniLM-L6-v2 embeddings.
    """
    embeddings = _build_embeddings()

    #  Load path: persisted index already exists 
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print("Loading existing ChromaDB...")
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        print(f"ChromaDB loaded: {count} chunks available")
        return vectorstore

    #  Build path: embed knowledge base from scratch 
    print("Building ChromaDB from knowledge base...")

    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise FileNotFoundError(
            f"Knowledge base directory not found: '{KNOWLEDGE_BASE_PATH}'. "
            "Set KNOWLEDGE_BASE_PATH in your .env file."
        )

    loader = DirectoryLoader(
        KNOWLEDGE_BASE_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
        use_multithreading=False,
    )

    docs = loader.load()
    print(f"Loaded {len(docs)} source documents from {KNOWLEDGE_BASE_PATH}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(docs)
    print(f"Embedding {len(chunks)} chunks...")

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH,
    )

    print(f"ChromaDB built: {len(chunks)} chunks indexed")
    return vectorstore
