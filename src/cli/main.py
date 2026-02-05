import click

# Import subcommands from command modules
from .commands import data, ingestion # Will be defined in src/cli/commands/data.py
# from .commands import ingestion # Will be defined in src/cli/commands/ingestion.py

# @click.group()
# def cli():
#     """A command-line tool for TechDoc Genie operations."""
#     pass

# # Register subcommands
# cli.add_command(data.data_group)
# # cli.add_command(ingestion.ingestion_group)

@click.group()
@click.version_option(version="0.1.0", prog_name="techdoc-genie")
def cli():
    """TechDoc Genie - Manage technical documentation."""
    pass


# Register command groups
cli.add_command(data.data_group)
cli.add_command(ingestion.ingestion_group)


if __name__ == "__main__":
    cli()
