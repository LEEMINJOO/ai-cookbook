import datasets

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt

from rag_base.config import EMBEDDING_MODEL_NAME, MARKDOWN_SEPARATORS


def load_documents(
    dataset_name, embedding_moel_name=EMBEDDING_MODEL_NAME, verbose=False
):
    ds = load_dataset(dataset_name)
    docs = convert_documents(ds)
    docs = split_documents(docs, embedding_moel_name, verbose)
    return docs


def load_dataset(dataset_name):
    ds = datasets.load_dataset(dataset_name, split="train")
    return ds


def convert_documents(ds):
    docs = []
    for data in ds:
        doc = Document(page_content=data["text"], metadata={"source": data["source"]})
        docs.append(doc)
    return docs


def split_documents(docs, embedding_moel_name=EMBEDDING_MODEL_NAME, verbose=False):
    chunk_size = SentenceTransformer(embedding_moel_name).max_seq_length
    if verbose:
        print(f"Model's maximum sequence length: {chunk_size}")

    tokenizer = AutoTokenizer.from_pretrained(embedding_moel_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in docs:
        docs_processed += text_splitter.split_documents([doc])

    unique_texts = set()
    docs_unique = []
    for doc in docs_processed:
        content = doc.page_content
        if content not in unique_texts:
            unique_texts.add(content)
            docs_unique.append(doc)

    if verbose:
        lengths = [len(tokenizer.encode(doc.page_content)) for doc in docs_processed]
        fig = pd.Series(lengths).hist()
        plt.title("Distribution of document lengths in the docs (in count of tokens)")
        plt.show()

    return docs_unique
