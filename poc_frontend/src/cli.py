import click
from src import create_app
from src.config import FLASK_RUN_PORT

@click.command()
def run():
    """Start everything: HTTP + Discord + OBS relay."""
    # python -m src.cli
    app = create_app()
    app.run(host="0.0.0.0", port=FLASK_RUN_PORT, debug=True)

if __name__ == "__main__":
    run()
