import fire

from database import Database
from modules import Text2SQLAgent


def run(query=None):
    if query is None:
        query = (
            "Can you give me the name of the client who got the most expensive receipt?"
        )
        # or "Which waiter got more total money from tips?"

    db = Database()
    agent = Text2SQLAgent(db)
    answer = agent.run(query)
    print(answer)


if __name__ == "__main__":
    fire.Fire(run)
