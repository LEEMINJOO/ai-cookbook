from functools import partial
import getpass

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

from src.config import READER_MODEL_NAME, PROMPT_TEMPLATE

HF_TOKEN = getpass.getpass("HF_TOKEN:")


def get_reader():
    client = InferenceClient(READER_MODEL_NAME, token=HF_TOKEN)
    reader = partial(client.text_generation, max_new_tokens=500)
    return reader


def build_message(question, contexts):
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"\nDocument {str(i)}:::\n" + doc for i, doc in enumerate(contexts)]
    )

    template = build_prompt_template()
    message = template.format(question=question, context=context)
    return message


def build_prompt_template():
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    prompt_template = tokenizer.apply_chat_template(
        PROMPT_TEMPLATE, tokenize=False, add_generation_prompt=True
    )
    return prompt_template
