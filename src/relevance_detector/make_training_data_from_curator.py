"""Generate training data based on curator data and KPI mapping."""

import os

import pandas as pd


def make_training_data(curator_data_path: str, kpi_mapping_path: str) -> None:
    """
    Generate training data based on curator data and KPI mapping.

    Args:
        curator_data_path (str): Path to the curator data CSV file.
        kpi_mapping_path (str): Path to the KPI mapping CSV file.

    Returns:
        None
    """
    data = pd.read_csv(curator_data_path)

    kpi_mapping_df = pd.read_csv(kpi_mapping_path)
    kpi_mapping_df = kpi_mapping_df.iloc[:, :5]

    kpi_dict = {}
    for _, row in kpi_mapping_df.iterrows():
        kpi_id = row["kpi_id"]
        if isinstance(kpi_id, float) and kpi_id.is_integer():
            kpi_id = int(kpi_id)
        kpi_dict[kpi_id] = {
            "question": row["question"],
            "sectors": row["sectors"],
            "add_year": row["add_year"],
            "kpi_category": row["kpi_category"],
        }

    # Initialize lists to store data for new DataFrame
    question_list = []
    paragraph_list = []
    label_list = []

    for i in range(data.shape[0]):
        paragraph = data["context"][i]
        kpi_id = data["kpi_id"][i]

        label = data["label"][i]
        question = kpi_dict[kpi_id]["question"]

        # Append values to lists
        question_list.append(question)
        paragraph_list.append(paragraph)
        label_list.append(label)

    # Create new DataFrame
    train_data = pd.DataFrame(
        {
            "question": question_list,
            "paragraph": paragraph_list,
            "label": label_list,
        }
    )

    # Define the directory path
    directory = r"src\relevance_detector\data"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = "train_data.csv"
    save_dir = os.path.join(directory, file_name)
    train_data.to_csv(save_dir)

    print(f"Data saved at {save_dir}")


# make_training_data(r"src\relevance_detector\OSC\output_curator.csv", r"src\relevance_detector\OSC\kpi_mapping.csv")
