import time

from loguru import logger


async def log_requests_middleware(request, call_next):
    start = time.perf_counter()
    logger.info(f"Request received: {request.method} {request.url.path}")

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration_ms:.2f}ms"
    )
    return response
