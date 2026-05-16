# rag/retriever.py

"""
Helper utilities for querying the ChromaDB vectorstore.

All public functions return plain page_content strings (List[str]),
not LangChain Document objects, so callers never need to import Document.
"""

from typing import List

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever


#  Retriever factory 

def get_retriever(vectorstore: Chroma, k: int = 4) -> VectorStoreRetriever:
    """
    Return a LangChain retriever configured for similarity search.

    Parameters
    ----------
    vectorstore : Chroma
        An initialised Chroma vectorstore (from indexer.build_or_load_vectorstore).
    k : int
        Number of nearest-neighbour chunks to return per query.

    Returns
    -------
    VectorStoreRetriever
        Ready-to-use retriever; call .invoke(query) or .ainvoke(query) on it.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})


#  Synchronous helper 

def retrieve_chunks(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
) -> List[str]:
    """
    Synchronously retrieve the top-k most relevant text chunks for *query*.

    Parameters
    ----------
    vectorstore : Chroma
        An initialised Chroma vectorstore.
    query : str
        The semantic search query (e.g. 'market size competition trends fintech').
    k : int
        Number of chunks to return.

    Returns
    -------
    List[str]
        Plain text content of the retrieved chunks; order is by descending similarity.
    """
    retriever = get_retriever(vectorstore, k=k)
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


#  Asynchronous helper 

async def retrieve_chunks_async(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
) -> List[str]:
    """
    Asynchronously retrieve the top-k most relevant text chunks for *query*.

    Designed for use inside LangGraph async nodes where calling the sync
    version would block the event loop.

    Parameters
    ----------
    vectorstore : Chroma
        An initialised Chroma vectorstore.
    query : str
        The semantic search query.
    k : int
        Number of chunks to return.

    Returns
    -------
    List[str]
        Plain text content of the retrieved chunks; order is by descending similarity.
    """
    retriever = get_retriever(vectorstore, k=k)
    docs = await retriever.ainvoke(query)
    return [doc.page_content for doc in docs]
