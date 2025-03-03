from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

import numpy as np
import pacmap
import pandas as pd
import plotly.express as px

from src.config import EMBEDDING_MODEL_NAME
from src.dataset import load_documents

knowledge_db = None


def load_knowledge_db(verbose):
    global knowledge_db

    if knowledge_db is None:
        docs = load_documents(
            dataset_name="m-ric/huggingface_doc",
            verbose=verbose,
        )
        knowledge_db = build_knowledge_db(docs)

    return knowledge_db


def build_knowledge_db(docs) -> FAISS:
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


def recontruct_embeddings(knowledge_db: FAISS):
    ntotal = knowledge_db.index.ntotal
    return [list(knowledge_db.index.reconstruct_n(idx, 1)[0]) for idx in range(ntotal)]


def visualize_knowledge_db(knowledge_db: FAISS, query):
    query_vector = knowledge_db._embed_query(query)

    embedding_projector = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=None,
        MN_ratio=0.5,
        FP_ratio=2.0,
        random_state=1,
    )

    embegddings = recontruct_embeddings(knowledge_db) + [query_vector]
    embegddings_2d = embedding_projector.fit_transform(
        np.array(embegddings),
        init="pca",
    )

    df = _build_df_knowledge(knowledge_db, query, embegddings_2d)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        hover_data="extract",
        size="size_col",
        symbol="symbol",
        color_discrete_map={"User query": "black"},
        width=1000,
        height=700,
    )
    fig.update_traces(
        marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        legend_title_text="<b>Chunk source</b>",
        title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
    )
    fig.show()


def _build_df_knowledge(knowledge_db: FAISS, query, embegddings_2d):
    ntotal = knowledge_db.index.ntotal

    data = []
    for i in range(ntotal):
        key = knowledge_db.index_to_docstore_id[i]
        doc = knowledge_db.docstore.search(key)

        d = {
            "x": embegddings_2d[i, 0],
            "y": embegddings_2d[i, 1],
            "source": doc[i].metadata["source"].split("/")[1],
            "extract": doc[i].page_content[:100] + "...",
            "symbol": "circle",
            "size_col": 4,
        }
        data.append(d)

    data += [
        {
            "x": embegddings_2d[-1, 0],
            "y": embegddings_2d[-1, 1],
            "source": "User query",
            "extract": query,
            "size_col": 100,
            "symbol": "star",
        }
    ]

    df = pd.DataFrame.from_dict(data)
    return df
