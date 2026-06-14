import asyncio
from dotenv import load_dotenv

from server import start_websocket_server
from background import start_background_tasks


async def main():
    print("[main] loading environment variables...")
    load_dotenv()

    print("[main] starting background tasks...")
    start_background_tasks()

    print("[main] starting WebSocket server...")
    await start_websocket_server()


if __name__ == "__main__":
    asyncio.run(main())
