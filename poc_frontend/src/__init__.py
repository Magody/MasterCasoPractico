from flask import Flask
from threading import Thread
from src.bots.discord_bot import run_discord
from src.bots.obs_relay    import run_obs_relay
from src.api.routes        import api_bp

def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="ui/static",
        template_folder="ui/templates"
    )
    app.register_blueprint(api_bp)

    # start background workers immediately
    # Thread(target=run_discord, daemon=True).start()
    # Thread(target=run_obs_relay, daemon=True).start()

    return app
