from functools import partial
import getpass

from langchain_core.vectorstores import VectorStore
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer
from ragatouille import RAGPretrainedModel

from config import (
    READER_MODEL_NAME,
    RERANKER_MODEL_NAME,
    PROMPT_TEMPLATE,
    NUM_RETREIVERS,
    NUM_RETREIVERS_FINAL,
)

HF_TOKEN = getpass.getpass("HF_TOKEN:")


class RAG:
    def __init__(self, vectordb: VectorStore, rerank=True):
        self.vectordb = vectordb
        self.reader = self._load_reader()

        self.reranker = None
        if rerank:
            self.reranker = RAGPretrainedModel.from_pretrained(RERANKER_MODEL_NAME)

    def run(self, query, verbose=False):
        contexts = self.retrive(query, verbose=verbose)
        message = self.build_message(query, contexts)

        answer = self.reader(message)
        return answer

    def retrive(self, query, verbose=True):
        if verbose:
            print(f"\nStarting retrieval for {query=}...")

        relevant_docs = self.vectordb.similarity_search(query=query, k=NUM_RETREIVERS)
        relevant_docs = [
            doc.page_content for doc in relevant_docs
        ]  # Keep only the text

        if self.reranker:
            relevant_docs = self.reranker.rerank(
                query, relevant_docs, k=NUM_RETREIVERS_FINAL
            )
            relevant_docs = [doc["content"] for doc in relevant_docs]

        if verbose:
            print("========================Top document========================")
            print(relevant_docs[0])
        return relevant_docs

    def build_message(self, question, contexts):
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"\nDocument {str(i)}:::\n" + doc for i, doc in enumerate(contexts)]
        )

        template = self._build_prompt_template()
        message = template.format(question=question, context=context)
        return message

    def _build_prompt_template():
        tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
        prompt_template = tokenizer.apply_chat_template(
            PROMPT_TEMPLATE, tokenize=False, add_generation_prompt=True
        )
        return prompt_template

    def _load_reader(self):
        client = InferenceClient(READER_MODEL_NAME, token=HF_TOKEN)
        reader = partial(client.text_generation, max_new_tokens=500)
        return reader
