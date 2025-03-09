# ai-cookbook

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
