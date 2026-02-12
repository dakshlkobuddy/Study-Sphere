import json
import logging
import os
import uuid
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "error_id", "source_mode", "session_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload)


def get_logger(name="study_sphere"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/app.log", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def new_error_id():
    return uuid.uuid4().hex[:10]

