from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Rejected invalid payload on {request.method} {request.url.path}: {exc.errors()}")
    # Delegates to FastAPI's default handler so the 422 response shape is untouched.
    return await request_validation_exception_handler(request, exc)


async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.opt(exception=exc).error(f"Unhandled server error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )
