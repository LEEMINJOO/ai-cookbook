from langchain.vectorstores import FAISS

from ragatouille import RAGPretrainedModel

from src.config import (
    RERANKER_MODEL_NAME,
    NUM_RETREIVERS,
    NUM_RETREIVERS_FINAL,
)

reranker = None


def load_reranker():
    global reranker
    if reranker is None:
        reranker = RAGPretrainedModel.from_pretrained(RERANKER_MODEL_NAME)
    return reranker


def retrive(query, knowledge_db: FAISS, rerank=True, verbose=False):
    if verbose:
        print(f"\nStarting retrieval for {query=}...")

    relevant_docs = knowledge_db.similarity_search(query=query, k=NUM_RETREIVERS)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    if rerank:
        reranker = load_reranker()
        relevant_docs = reranker.rerank(query, relevant_docs, k=NUM_RETREIVERS_FINAL)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    if verbose:
        print("========================Top document========================")
        print(relevant_docs[0])

    return relevant_docs
