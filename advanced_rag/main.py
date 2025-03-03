import fire

from src.knowledge import load_knowledge_db, visualize_knowledge_db
from src.retrieval import retrive
from src.llm import get_reader, build_message


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
