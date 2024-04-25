"""
Module: inference_utils
Author: Tanishq-ids

This module contains utility functions for performing inference
using pre-trained sequence classification models.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def get_inference(question: str, paragraph: str, model_path: str, tokenizer_path: str) -> int:
    """
    Perform inference using a pre-trained sequence classification model.

    Parameters:
        question (str): The question for inference.
        paragraph (str): The paragraph to be analyzed.
        model_path (str): Path to the pre-trained model.
        tokenizer_path (str): Path to the tokenizer to the pre-trained model.

    Returns:
        int: Predicted label ID.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Tokenize inputs
    inputs = tokenizer(question, paragraph, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted label
    predicted_label_id = torch.argmax(outputs.logits, dim=1).item()

    return predicted_label_id
