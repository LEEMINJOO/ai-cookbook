import pandas as pd

from huggingface_hub import InferenceClient

from rag_base.dataset import load_dataset
from config import QUERY_PROMPT, EVALUATION_PROMPT, EVALUATION_MODEL_NAME

DEFAULT_SCORE = 2


class Evaluator:
    def __init__(self, model_id=EVALUATION_MODEL_NAME):
        self.client = load_evaluation_client(model_id)

    def run(self, module, dataset_name="m-ric/huggingface_doc_qa_eval", n=10):
        dataset = load_dataset(dataset_name)

        outputs = []
        for example in dataset[:n]:
            experiment = self._evaluate_one(module, example)
        outputs.append(experiment)

        scores = self._compute_scores(outputs)
        print(f"Average score for RAG: {scores.mean()*100:.1f}%")
        return scores

    def _evaluate_one(self, module, example):
        question = example["question"]
        query = QUERY_PROMPT.format(question=question)
        answer = module.run(query)
        experiment = {
            "question": question,
            "true_answer": example["answer"],
            "source_doc": example["source_doc"],
            "generated_answer": answer,
        }

        eval_prompt = EVALUATION_PROMPT.format(
            instruction=question,
            response=answer,
            reference_answer=example["answer"],
        )

        result = self.client.text_generation(eval_prompt, max_new_tokens=1000)
        try:
            feedback, score = [item.strip() for item in result.split("[RESULT]")]
            experiment.update({"eval_score": score, "eval_feedback": feedback})
        except:
            print(f"Parsing failed - output was: {result}")

        return experiment

    def _compute_scores(self, outputs):
        results = pd.DataFrame.from_dict(outputs)
        results = results.loc[~results["generated_answer"].str.contains("Error")]

        scores = results["eval_score"].fillna(DEFAULT_SCORE).apply(fill_score)
        scores = (scores - 1) / 2
        return scores


class OpenAIInferenceClient:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()

    def text_generation(self, prompt, max_new_tokens=1000):
        messages = [
            {"role": "system", "content": "You are a fair evaluator language model."},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=max_new_tokens
        )
        return response.choices[0].message.content


def load_evaluation_client(model_id=EVALUATION_MODEL_NAME):
    if model_id == "open_ai":
        return OpenAIInferenceClient()
    return InferenceClient(model_id)


def fill_score(x):
    try:
        return int(x)
    except:
        return DEFAULT_SCORE
