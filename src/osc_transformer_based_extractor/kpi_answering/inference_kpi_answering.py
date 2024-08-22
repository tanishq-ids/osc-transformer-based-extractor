"""
## Inference
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline


def validate_path_exists(path: str, which_path: str):
    """
    Validate if the given path exists.

    Args:
        path (str): The path to validate.

    Returns:
        ValueError: If the path does not exist.
    """
    if not os.path.exists(path):
        raise ValueError(f"{which_path}: {path} does not exist.")


def get_inference_kpi_answering(question: str, context: str, model_path: str):
    """
    Performs kpi-answering inference using a specified model.

    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.
        model_path (str): Path to the pre-trained model to be used for inference.

    Returns:
        tuple: A tuple containing:
            - answer (str): The predicted answer.
            - score (float): The confidence score of the prediction.
            - start (int): The start position of the answer in the context.
            - end (int): The end position of the answer in the context.
    """
    question_answerer = pipeline("question-answering", model=model_path)
    result = question_answerer(question=question, context=context)
    return result["answer"], result["score"], result["start"], result["end"]


def run_full_inference_kpi_answering(
    data_file_path: str, output_path: str, model_path: str
):
    """
    Runs full inference on a dataset of questions and contexts, and saves the results.

    Args:
        data_file_path (str): Path to the input CSV file containing the dataset.
            The dataset should have columns 'question' and 'context'.
        output_path (str): Path to the directory where the output Excel file will be saved.
        model_path (str): Path to the pre-trained model to be used for inference.

    Returns:
        None: The function saves the resulting DataFrame to an Excel file and prints a success message.
    """
    data_file_path = str(Path(data_file_path))
    output_path = str(Path(output_path))
    model_path = str(Path(model_path))

    data = pd.read_csv(data_file_path)

    result = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing Rows"):
        question = row["question"]
        context = row["context"]
        answer, score, start, end = get_inference_kpi_answering(
            question, context, model_path
        )
        result.append({"answer": answer, "start": start, "end": end, "score": score})

    df = pd.DataFrame(result)

    combined_df = pd.concat([data, df], axis=1)

    file_name = Path(output_path) / "output.xlsx"
    combined_df.to_excel(file_name, index=False)
    print(f"Successfully SAVED resulting file at {file_name}")
