READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
RERANKER_MODEL_NAME = "colbert-ir/colbertv2.0"

PROMPT_TEMPLATE = [
    {
        "role": "system",
        "content": (
            "Using the information contained in the context,"
            "give a comprehensive answer to the question."
            "Respond only to the question asked, response should be concise and relevant to the question."
            "Provide the number of the source document when relevant."
            "If the answer cannot be deduced from the context, do not give an answer."
        ),
    },
    {
        "role": "user",
        "content": (
            "Context:\n{context}\n"
            "---"
            "Now here is the question you need to answer.\n"
            "Question: {question}"
        ),
    },
]


NUM_RETREIVERS = 30
NUM_RETREIVERS_FINAL = 5
