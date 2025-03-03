import getpass

from smolagents import Tool, HfApiModel, ToolCallingAgent
from langchain_core.vectorstores import VectorStore

from config import AGENT_MODEL_NAME


HF_TOKEN = getpass.getpass("HF_TOKEN:")


class RAGAgent:
    def __init__(self, vectordb: VectorStore):
        retriever_tool = RetrieverTool(vectordb)
        model = HfApiModel(model_id=AGENT_MODEL_NAME, token=HF_TOKEN)
        self.agent = ToolCallingAgent(tools=[retriever_tool], model=model)

    def run(self, query):
        answer = self.agent.run(query)
        return answer


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": (
                "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question."
                # "You must call this tool first before generating any answer."
            ),
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [
                f"===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
