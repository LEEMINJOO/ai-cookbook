from smolagents import Tool, CodeAgent, HfApiModel, tool
from sqlalchemy import inspect, text

from config import AGENT_MODEL_NAME
from database import Database


class Text2SQLAgent:
    def __init__(self, db: Database):
        sql_engine_tool = SQLEngineTool(db)
        model = HfApiModel(model_id=AGENT_MODEL_NAME)
        self.agent = CodeAgent(tools=[sql_engine_tool], model=model)

    def run(self, query):
        answer = self.agent.run(query)
        return answer


class SQLEngineTool(Tool):
    name = "sql_engine"
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be correct SQL.",
        }
    }
    output_type = "string"

    def __init__(self, db: Database, **kwargs):
        super().__init__(**kwargs)
        self.engine = db.get_engine()
        self._update_description()

    def _update_description(self):
        description = (
            "Allows you to perform SQL queries on the table. "
            "Returns a string representation of the result.\n"
            "It can use the following tables:"
        )

        inspector = inspect(self.engine)
        for table in inspector.get_table_names():
            columns_info = [
                (col["name"], col["type"]) for col in inspector.get_columns(table)
            ]

            table_description = f"Table '{table}':\n"
            table_description += "Columns:\n" + "\n".join(
                [f"  - {name}: {col_type}" for name, col_type in columns_info]
            )
            description += "\n\n" + table_description
        self.description = description

    def forward(self, query: str) -> str:
        output = ""
        with self.engine.connect() as con:
            rows = con.execute(text(query))
            for row in rows:
                output += "\n" + str(row)
        return output
