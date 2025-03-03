from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from ragatouille import RAGPretrainedModel

from src.config import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    NUM_RETREIVERS,
    NUM_RETREIVERS_FINAL,
)
from src.dataset import load_documents

knowledge_db = None
reranker = None


def load_knowledge_db(verbose):
    global knowledge_db

    if knowledge_db is None:
        docs = load_documents(
            dataset_name="m-ric/huggingface_doc",
            verbose=verbose,
        )
        knowledge_db = build_knowledge_db(docs)

    return knowledge_db


def build_knowledge_db(docs):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        # model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True  # Set `True` for cosine similarity
        },
    )

    knowledge_db = FAISS.from_documents(
        docs, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    return knowledge_db


def recontruct_embeddings(knowledge_db):
    ntotal = knowledge_db.index.ntotal
    return [list(knowledge_db.index.reconstruct_n(idx, 1)[0]) for idx in range(ntotal)]


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
