EMBEDDING_MODEL_NAME = "thenlper/gte-small"

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

KNOWLEDGE_SAVE_PATH = "resources/knowledge"


# evaluator
EVALUATION_MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

QUERY_PROMPT = """Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
give a comprehensive answer to the question below.
Respond only to the question asked, response should be concise and relevant to the question.
If you cannot find information, do not give up and try calling your retriever again with different arguments!
Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
Your queries should not be questions but affirmative form sentences: e.g. rather than "How do I load a model from the Hub in bf16?", query should be "load a model from the Hub bf16 weights".

Question:
{question}"""

EVALUATION_PROMPT = """You are a fair evaluator language model.

You will be given an instruction, a response to evaluate, a reference answer that gets a score of 3, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 3. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 3}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.
5. Do not score conciseness: a correct answer that covers the question should receive max score, even if it contains additional useless information.

The instruction to evaluate:
{instruction}

Response to evaluate:
{response}

Reference Answer (Score 3):
{reference_answer}

Score Rubrics:
[Is the response complete, accurate, and factual based on the reference answer?]
Score 1: The response is completely incomplete, inaccurate, and/or not factual.
Score 2: The response is somewhat complete, accurate, and/or factual.
Score 3: The response is completely complete, accurate, and/or factual.

Feedback:"""
