from pythonjsonlogger import jsonlogger

from dbmeta_app.config import get_settings

settings = get_settings()
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": jsonlogger.JsonFormatter,
             "format": (
                "%(asctime)s %(name)s %(levelname)s "
                "%(module)s %(funcName)s %(lineno)d %(message)s"
            ),
        }
    },
    "handlers": {
        "default": {
            "level": settings.log_level,
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
        "uvicorn.access": {
            "level": settings.log_level,
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
    },
    "loggers": {
        "": {"handlers": ["default"], "level": settings.log_level},
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {
            "handlers": ["default"],
            "level": "ERROR",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["uvicorn.access"],
            "level": "INFO",
            "propagate": False,
        },
    },
}
