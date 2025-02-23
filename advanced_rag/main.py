from src.dataset import load_documents
from src.retrieval import build_knowledge_db, retrive
from src.llm import get_reader, build_message
from src.visualization import visualize_knowledge_db

knowledge_db = None


def load_knowledge_db(verbose):
    global knowledge_db

    if knowledge_db is None:
        docs = load_documents(verbose=verbose)
        knowledge_db = build_knowledge_db(docs)

    return knowledge_db


def run(query, verbose=False):
    knowledge_db = load_knowledge_db(verbose)

    if verbose:
        visualize_knowledge_db(knowledge_db, query)

    contexts = retrive(query, knowledge_db, verbose=verbose)
    message = build_message(query, contexts)

    reader = get_reader()
    answer = reader(message)

    print(answer)
