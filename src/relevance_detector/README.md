# Relevance Detector

This folder contains a set of scripts and notebooks designed to process data, train a sentence transformer model, and perform inferences to detect the relevance of folder contents. Below is a detailed description of each file and folder included in this repository.

## Repository Contents

### Python Scripts

1. **`inference.py`**
   - This script contains the function to make inferences using the trained model.
   - **Usage**: Import this script and use the provided function to predict the relevance of new data.
   - **Example**: 
     ```python
     from inference import get_inference
     result = get_inference(question="What is the relevance?", paragraph="This is a sample paragraph.", model_path="path/to/model", tokenizer_path="path/to/tokenizer")
     ```
     **Parameters**:
       - `question (str)`: The question for inference.
       - `paragraph (str)`: The paragraph to be analyzed.
       - `model_path (str)`: Path to the pre-trained model.
       - `tokenizer_path (str)`: Path to the tokenizer of the pre-trained model.

2. **`make_training_data_from_curator.py`**
   - This script processes CSV data obtained from a module named `curator` to make it suitable for training the model.
   - **Usage**: Run this script to generate training data from the curator's output and save it in the `Data` folder.

### Jupyter Notebooks

1. **`inference_demo.ipynb`**
   - A notebook to demonstrate how to perform inferences using a custom model and tokenizer.
   - **Features**: Allows specifying model and tokenizer paths, which can be local paths or HuggingFace paths.
   - **Usage**: Open this notebook and follow the instructions to test inference with your own models.

2. **`make_sample_training_data.ipynb`**
   - This notebook was used to create sample training data from a sample CSV file.
   - **Usage**: Open and run this notebook to understand the process of creating sample data for training.

3. **`train_sentence_transformer.ipynb`**
   - A notebook to train a sentence transformer model and save the trained model locally.
   - **Usage**: Open and execute this notebook to train your model using the prepared data and save the trained model for inference.

### Data Folder

- **`Data/`**
  - This folder contains the processed training data obtained from the `curator` module. It serves as the input for training the sentence transformer model.

## How to Use This Repository

1. **Prepare Training Data**:
   - If you have CSV data from the curator module, run `make_training_data_from_curator.py` to process and save it in the `Data` folder.
   - Alternatively, you can use `make_sample_training_data.ipynb` to generate sample data from a sample CSV file.

2. **Train the Model**:
   - Use `train_sentence_transformer.ipynb` to train a sentence transformer model with the processed data from the `Data` folder and save it locally. Follow the steps in the notebook to configure and start the training process. 

3. **Perform Inference**:
   - Use `inference_demo.ipynb` to perform inferences with your trained model. Specify the model and tokenizer paths (either local or from HuggingFace) and run the notebook cells to see the results.
   - For programmatic inference, you can use the function provided in `inference.py`:
     ```python
     from inference import get_inference
     result = get_inference(question="What is the relevance?", paragraph="This is a sample paragraph.", model_path="path/to/model", tokenizer_path="path/to/tokenizer")
     ```

## Setting Up the Environment

To set up the working environment for this repository, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/folder-relevance-detector.git
   cd folder-relevance-detector
   ```

2. **Create a new virtual environment and activate it**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install PDM**:
   ```bash
   pip install pdm
   ```

4. **Sync the environment using PDM**:
   ```bash
   pdm sync
   ```

5. **Add any new library**:
   ```bash
   pdm add <library-name>
   ```

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python packages (install via `pdm` as described above)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

For further details and documentation, please refer to the individual scripts and notebooks provided in this repository.# Relevance Detector

This folder contains a set of scripts and notebooks designed to process data, train a sentence transformer model, and perform inferences to detect the relevance of folder contents. Below is a detailed description of each file and folder included in this repository.

## Repository Contents

### Python Scripts

1. **`inference.py`**
   - This script contains the function to make inferences using the trained model.
   - **Usage**: Import this script and use the provided function to predict the relevance of new data.
   - **Example**: 
     ```python
     from inference import get_inference
     result = get_inference(question="What is the relevance?", paragraph="This is a sample paragraph.", model_path="path/to/model", tokenizer_path="path/to/tokenizer")
     ```
     **Parameters**:
       - `question (str)`: The question for inference.
       - `paragraph (str)`: The paragraph to be analyzed.
       - `model_path (str)`: Path to the pre-trained model.
       - `tokenizer_path (str)`: Path to the tokenizer of the pre-trained model.

2. **`make_training_data_from_curator.py`**
   - This script processes CSV data obtained from a module named `curator` to make it suitable for training the model.
   - **Usage**: Run this script to generate training data from the curator's output and save it in the `Data` folder.

### Jupyter Notebooks

1. **`inference_demo.ipynb`**
   - A notebook to demonstrate how to perform inferences using a custom model and tokenizer.
   - **Features**: Allows specifying model and tokenizer paths, which can be local paths or HuggingFace paths.
   - **Usage**: Open this notebook and follow the instructions to test inference with your own models.

2. **`make_sample_training_data.ipynb`**
   - This notebook was used to create sample training data from a sample CSV file.
   - **Usage**: Open and run this notebook to understand the process of creating sample data for training.

3. **`train_sentence_transformer.ipynb`**
   - A notebook to train a sentence transformer model and save the trained model locally.
   - **Usage**: Open and execute this notebook to train your model using the prepared data and save the trained model for inference.

### Data Folder

- **`Data/`**
  - This folder contains the processed training data obtained from the `curator` module. It serves as the input for training the sentence transformer model.

## How to Use This Repository

1. **Prepare Training Data**:
   - If you have CSV data from the curator module, run `make_training_data_from_curator.py` to process and save it in the `Data` folder.
   - Alternatively, you can use `make_sample_training_data.ipynb` to generate sample data from a sample CSV file.

2. **Train the Model**:
   - Use `train_sentence_transformer.ipynb` to train a sentence transformer model with the processed data from the `Data` folder and save it locally. Follow the steps in the notebook to configure and start the training process. 

3. **Perform Inference**:
   - Use `inference_demo.ipynb` to perform inferences with your trained model. Specify the model and tokenizer paths (either local or from HuggingFace) and run the notebook cells to see the results.
   - For programmatic inference, you can use the function provided in `inference.py`:
     ```python
     from inference import get_inference
     result = get_inference(question="What is the relevance?", paragraph="This is a sample paragraph.", model_path="path/to/model", tokenizer_path="path/to/tokenizer")
     ```

## Setting Up the Environment

To set up the working environment for this repository, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/folder-relevance-detector.git
   cd folder-relevance-detector
   ```

2. **Create a new virtual environment and activate it**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install PDM**:
   ```bash
   pip install pdm
   ```

4. **Sync the environment using PDM**:
   ```bash
   pdm sync
   ```

5. **Add any new library**:
   ```bash
   pdm add <library-name>
   ```

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python packages (install via `pdm` as described above)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

For further details and documentation, please refer to the individual scripts and notebooks provided in this repository.