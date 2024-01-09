from pathlib import Path

import click

from ..converters import convert_model


@click.command(name="convert")
@click.option("--name", type=str, required=True, help="The model name.")
@click.option(
    "--out",
    type=click.Path(file_okay=False, writable=True),
    default=".",
    help="Output directory where to store the converted models.",
)
@click.option(
    "--compression", type=str, default=None, help="Model weights compression method."
)
def main(name: str, out: str, compression: str | None):
    out_dir = Path(out).expanduser().resolve()

    convert_model(model_name=name, out_dir=out_dir, compression=compression)
