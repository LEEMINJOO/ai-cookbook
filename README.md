# ai-cookbook

## Basic
[Agent 와 MCP](./agent-mcp-basic.ipynb)


## Huggingface-Learn
| name | source | updated |
|------|------|------|
| [advanced_rag](./advanced_rag/) | [Advanced RAG on Hugging Face documentation using LangChain](https://huggingface.co/learn/cookbook/advanced_rag) | 2025.02.23 |
| [agent_rag](./agent_rag/) | [Agentic RAG: turbocharge your RAG with query reformulation and self-query!](https://huggingface.co/learn/cookbook/agent_rag) | 2025.03.03 |
| [agent_text_to_sql](./agent_text_to_sql) | [Agent for text-to-SQL with automatic error correction](https://huggingface.co/learn/cookbook/agent_text_to_sql) | 2025.03.09 |


## ETC
| name | source | updated |
|------|------|------|
| [mcp-quickstart](./mcp-quickstart/) | [Model Context Protocol](https://modelcontextprotocol.io/introduction) | 2025.03.09 |
| [structured_generation](./etc/structured_generation.ipynb) | [RAG with source highlighting using Structured generation](https://huggingface.co/learn/cookbook/structured_generation) | 2025.04.05 |
| [serverless-api](./etc/serverless_api.ipynb) | [Serverless Inference API](https://huggingface.co/learn/cookbook/enterprise_hub_serverless_inference_api) | 2025.04.12 |
| [effective-agent](./etc/effective-agent.ipynb) | [파이썬으로 AI 에이전트 만들기](https://github.com/dabidstudio/python-ai-agents) | 2025.04.12 |


### Evironment

* 첫 환경 설정
   ```bash
   mkdir {project}

   uv init

   # requirements.txt
   uv -r requirements.txt
   # add
   uv add ipykernel

   source .venv/bin/activate

   python -m ipykernel install --user --name {name} --display-name {name}
   ```

## 가이드라인
* rag 기본 개념을 알고 싶다? advanced_rag
* rag 를 이용한 agent 를 구현하고 싶다? agent_rag
* mcp 서버를 이해하고 싶다? mcp-quickstart
* text2sql 기본 개념?
* text to sql mcp 서버를 하려면?
  * db 정보는 rag 로 가져오고 