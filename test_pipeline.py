import asyncio
import logging

from chain import process_text


async def main() -> None:
    text = (
        "La API usa FastAPI, Redis para caché y PostgreSQL para persistencia. "
        "El pool de conexiones está agotado y el servicio está caído."
    )
    result = await process_text(text)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(main())
