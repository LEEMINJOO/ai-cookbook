import fire

from src.retrieval import load_knowledge_db, retrive
from src.llm import get_reader, build_message
from src.visualization import visualize_knowledge_db


def run(query, verbose=False):
    knowledge_db = load_knowledge_db(verbose)
    if verbose:
        visualize_knowledge_db(knowledge_db, query)

    contexts = retrive(query, knowledge_db, verbose=verbose)
    message = build_message(query, contexts)

    reader = get_reader()
    answer = reader(message)

    print(answer)


if __name__ == "__main__":
    fire.Fire(run)
