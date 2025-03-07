# https://modelcontextprotocol.io/quickstart/client

from typing import Optional
from contextlib import AsyncExitStack

import fire
from pprint import pprint
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from huggingface_hub import InferenceClient

from config import CLIENT_MODEL_NAME, SYSTEM_PROMPT


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_params: StdioServerParameters):
        read, write = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await self.session.initialize()

    async def get_tools(self):
        response = await self.session.list_tools()
        descs = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
            for tool in response.tools
        ]
        tools = [{"type": "function", "function": desc} for desc in descs]
        return tools

    async def call_tool(self, name, inputs):
        result = await self.session.call_tool(name, inputs)
        return result.content


async def run(server_script_path, query="What are the active weather alerts in Texas?"):

    server_params = StdioServerParameters(
        command="python", args=[server_script_path]
    )  # command="uv", args=["run", server_script_path]

    mcp_client = MCPClient()
    await mcp_client.connect_to_server(server_params)

    tools = await mcp_client.get_tools()
    print("Tools:")
    pprint(tools)

    client = InferenceClient(CLIENT_MODEL_NAME)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    response = client.chat_completion(
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=500,
    )
    function = response.choices[0].message.tool_calls[0].function

    print(f"\nQuery: {query}\n")
    print(f"Selected Tool: {function.name} - {function.arguments}\n")

    result = await mcp_client.call_tool(function.name, function.arguments)
    print(f"Result: \n {result[0].text}")


if __name__ == "__main__":
    fire.Fire(run)
