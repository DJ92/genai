import asyncio


async def worker_loop() -> None:
    while True:
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(worker_loop())
