import typer
from .train_kpi_answering import (
    train_kpi_answering,
    check_output_dir,
)
from .inference_kpi_answering import (
    run_full_inference_kpi_answering,
    validate_path_exists,
)

# Subcommand app for kpi_answering
kpi_answering_app = typer.Typer()


@kpi_answering_app.callback(invoke_without_command=True)
def kpi_answering(ctx: typer.Context):
    """
    Commands for kpi Answering tasks.

    Available commands:
    - fine-tune
    - inference
    """
    if ctx.invoked_subcommand is None:
        typer.echo(kpi_answering.__doc__)
        raise typer.Exit()


@kpi_answering_app.command("fine-tune")
def fine_tune_qna(
    data_path: str = typer.Argument(
        ..., help="Path to the CSV file containing training data."
    ),
    model_name: str = typer.Argument(
        ...,
        help="Name of the pre-trained model on HuggingFace OR path to local model directory.",
    ),
    max_length: int = typer.Argument(..., help="Maximum length of the sequences."),
    epochs: int = typer.Argument(..., help="Number of training epochs."),
    batch_size: int = typer.Argument(..., help="Batch size for training."),
    output_dir: str = typer.Argument(
        ..., help="Directory to save the fine-tuned model."
    ),
    save_steps: int = typer.Argument(
        ..., help="Number of steps between saving model checkpoints."
    ),
):
    """Fine-tune a pre-trained Hugging Face model on a custom dataset."""
    check_csv_columns_qna(data_path)
    check_output_dir(output_dir)

    train_kpi_answering(
        data_path=data_path,
        model_name=model_name,
        max_length=max_length,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        save_steps=save_steps,
    )

    typer.echo(f"Model '{model_name}' trained and saved successfully at {output_dir}")


@kpi_answering_app.command("inference")
def inference_qna(
    data_file_path: str = typer.Argument(
        ..., help="Path to the input CSV file containing the dataset."
    ),
    output_path: str = typer.Argument(
        ..., help="Path to the directory where the output Excel file will be saved."
    ),
    model_path: str = typer.Argument(
        ..., help="Path to the pre-trained model directory OR name on huggingface."
    ),
):
    """Perform inference using a pre-trained model on a dataset of kpis and contexts, saving an output Excel file."""
    try:
        validate_path_exists(data_file_path, "data_file_path")
        validate_path_exists(output_path, "output_path")
        validate_path_exists(model_path, "model_path")

        run_full_inference_kpi_answering(
            data_file_path=data_file_path,
            output_path=output_path,
            model_path=model_path,
        )

        typer.echo("Inference completed successfully!")

    except ValueError as ve:
        typer.echo(f"Error: {str(ve)}")
        raise typer.Exit(code=1)
