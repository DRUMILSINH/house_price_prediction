from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from api.exception_handlers import unhandled_exception_handler, validation_exception_handler
from api.middleware import log_requests_middleware
from api.routes import router
from logging_config import configure_logging

configure_logging()

app = FastAPI(title="California House Price Prediction API")
app.middleware("http")(log_requests_middleware)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
app.include_router(router)
