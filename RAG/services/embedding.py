"""
Embedding Service

負責建立 BGE-M3 Embedding

整個 API 只建立一次。
"""

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        print("Loading Embedding Model...")

        _embeddings = HuggingFaceEmbeddings(

            model_name=EMBEDDING_MODEL

        )

        print("Embedding Ready")

    return _embeddings