from functools import partial
import getpass

import fire
from langchain.vectorstores import FAISS
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from ragatouille import RAGPretrainedModel

from rag_base.knowledge import load_knowledge_db, visualize_knowledge_db
from config import (
    READER_MODEL_NAME,
    RERANKER_MODEL_NAME,
    PROMPT_TEMPLATE,
    NUM_RETREIVERS,
    NUM_RETREIVERS_FINAL,
)

HF_TOKEN = getpass.getpass("HF_TOKEN:")
reranker = None


def run(query, verbose=False):
    knowledge_db = load_knowledge_db(verbose=verbose)
    if verbose:
        visualize_knowledge_db(knowledge_db, query)

    contexts = retrive(query, knowledge_db, verbose=verbose)
    message = build_message(query, contexts)

    reader = load_reader()
    answer = reader(message)
    print(answer)


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


def load_reader():
    client = InferenceClient(READER_MODEL_NAME, token=HF_TOKEN)
    reader = partial(client.text_generation, max_new_tokens=500)
    return reader


def build_message(question, contexts):
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"\nDocument {str(i)}:::\n" + doc for i, doc in enumerate(contexts)]
    )

    template = build_prompt_template()
    message = template.format(question=question, context=context)
    return message


def build_prompt_template():
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    prompt_template = tokenizer.apply_chat_template(
        PROMPT_TEMPLATE, tokenize=False, add_generation_prompt=True
    )
    return prompt_template


if __name__ == "__main__":
    fire.Fire(run)
