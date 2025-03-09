# https://huggingface.co/docs/smolagents/en/tutorials/tools#tool-collection-from-any-mcp-server

from contextlib import ExitStack

import fire
from smolagents import ToolCollection, CodeAgent, HfApiModel
from mcp import StdioServerParameters

from config import CLIENT_MODEL_NAME


class MCPClientAgent:
    def __init__(self, server_params):
        self.exit_stack = ExitStack()

        self.tool_collection = None
        self.connect_to_server(server_params)

        self.agent = self.get_agent()

    def connect_to_server(self, server_params: StdioServerParameters):
        self.tool_collection = self.exit_stack.enter_context(
            ToolCollection.from_mcp(server_params)
        )

    def get_agent(self):
        model = HfApiModel(model_id=CLIENT_MODEL_NAME)
        agent = CodeAgent(
            tools=[*self.tool_collection.tools],
            model=model,
            add_base_tools=True,
        )
        return agent

    def get_tools(self):
        return {tool.name: tool for tool in self.tool_collection.tools}

    def call_tool(self, name, inputs):
        tools = self.get_tools()
        tool = tools[name]
        return tool.forward(**inputs)

    def run(self, query):
        answer = self.agent.run(query)
        return answer


def run(server_script_path, query="What are the active weather alerts in Texas?"):

    server_params = StdioServerParameters(
        command="python", args=[server_script_path]
    )  # command="uv", args=["run", server_script_path]

    mcp_client_agent = MCPClientAgent(server_params)

    tools = mcp_client_agent.get_tools()
    print("Tools:")
    print(list(tools.keys()))

    print(f"\nQuery: {query}\n")

    answer = mcp_client_agent.run(query)
    print(f"Result: \n {answer}")


if __name__ == "__main__":
    fire.Fire(run)
