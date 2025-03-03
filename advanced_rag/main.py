import fire

from rag_base.knowledge import load_knowledge_db, visualize_knowledge_db
from modules import RAG


def run(query, verbose=False):
    knowledge_db = load_knowledge_db(verbose=verbose)
    if verbose:
        visualize_knowledge_db(knowledge_db, query)

    rag = RAG(vectordb=knowledge_db)
    answer = rag.run(query)
    print(answer)


if __name__ == "__main__":
    fire.Fire(run)
