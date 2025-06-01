import asyncio
from src import config
from src.utils.vtuber_studio import VTubeStudioController

class VTubeService:
    def __init__(self):
        # Initialize controller with env-driven config
        self.ctrl = VTubeStudioController(
            plugin_name=config.VTS_PLUGIN_NAME,
            developer=config.VTS_DEVELOPER,
            token_path=str(config.TOKEN_PATH),
            host=config.VTS_HOST,
            port=config.VTS_PORT
        )

    def trigger(self, index: int):
        self.ctrl.trigger(index)
