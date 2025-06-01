import asyncio
import pyvts
import threading

class VTubeStudioController:
    def __init__(self,
                 plugin_name="MirAI",
                 developer="You",
                 token_path="./token.txt",
                 host="127.0.0.1",
                 port=8001):
        # 1) Prepare the controller
        self.vt = pyvts.vts(
            plugin_info={
                "plugin_name": plugin_name,
                "developer": developer,
                "authentication_token_path": token_path
            },
            vts_api_info={
                "version":"1.0",
                "name":"VTubeStudioPublicAPI",
                "host":host,
                "port":port
            }
        )

        # 2) New loop + background thread
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

        # 3) Connect & authenticate once
        asyncio.run_coroutine_threadsafe(self._connect_and_auth(), self.loop).result()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _connect_and_auth(self):
        await self.vt.connect()
        await self.vt.request_authenticate_token()
        await self.vt.request_authenticate()

    def trigger(self, idx: int):
        """
        Fire a hotkey by index. Automatically reconnects if the WS has died.
        """
        async def _do():
            # 1) Try to list hotkeys
            try:
                resp = await self.vt.request(self.vt.vts_request.requestHotKeyList())
            except Exception:
                # reconnect & retry once
                await self._connect_and_auth()
                resp = await self.vt.request(self.vt.vts_request.requestHotKeyList())

            names = [h["name"] for h in resp["data"]["availableHotkeys"]]
            if 0 <= idx < len(names):
                name = names[idx]
                try:
                    await self.vt.request(self.vt.vts_request.requestTriggerHotKey(name))
                except Exception:
                    # reconnect & retry
                    await self._connect_and_auth()
                    await self.vt.request(self.vt.vts_request.requestTriggerHotKey(name))

        # schedule it on the background loop
        asyncio.run_coroutine_threadsafe(_do(), self.loop)

    def close(self):
        # gracefully shut down
        asyncio.run_coroutine_threadsafe(self.vt.close(), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)