import typer
from .relevance_detector.cli_relevance_detector import relevance_detector_app
from .kpi_answering.cli_kpi_answering import kpi_answering_app

# Main Typer app
app = typer.Typer(name="osc-transformer-based-extractor")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    OSC Transformer Based Extractor CLI.

    Available commands:
    - relevance-detector
    - kpi-answering
    """
    if ctx.invoked_subcommand is None:
        typer.echo(main.__doc__)
        raise typer.Exit()


# Add the relevance_detector and question_answering apps as command groups under the main app
app.add_typer(relevance_detector_app, name="relevance-detector")
app.add_typer(kpi_answering_app, name="kpi-answering")
