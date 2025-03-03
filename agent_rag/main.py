import fire

from rag_base.knowledge import load_knowledge_db
from modules import RAGAgent


def run(query):
    knowledge_db = load_knowledge_db()

    agent = RAGAgent(vectordb=knowledge_db)
    answer = agent.run(query)
    print(answer)


if __name__ == "__main__":
    fire.Fire(run)
