from __future__ import annotations

import logging
import os
import threading

from rich.logging import RichHandler
from rich.text import Text

_SET_UP_LOGGERS = set()
_ADDITIONAL_HANDLERS = []


def _interpret_level_from_env(level: str | None, *, default=logging.DEBUG) -> int:
    if not level:
        return default
    if level.isnumeric():
        return int(level)
    return getattr(logging, level.upper())


_STREAM_LEVEL = _interpret_level_from_env(os.environ.get("SWE_REX_LOG_STREAM_LEVEL"))
_INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER = False

_THREAD_NAME_TO_LOG_SUFFIX: dict[str, str] = {}


def register_thread_name(name: str) -> None:
    thread_name = threading.current_thread().name
    _THREAD_NAME_TO_LOG_SUFFIX[thread_name] = name


class _RichHandlerWithEmoji(RichHandler):
    def __init__(self, emoji: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not emoji.endswith(" "):
            emoji += " "
        self.emoji = emoji

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname
        return Text.styled((self.emoji + level_name).ljust(10), f"logging.level.{level_name.lower()}")


def get_logger(name: str, *, emoji: str = "🦖") -> logging.Logger:
    """Get logger. Use this instead of `logging.getLogger` to ensure
    that the logger is set up with the correct handlers.
    """
    thread_name = threading.current_thread().name
    if thread_name != "MainThread":
        name = name + "-" + _THREAD_NAME_TO_LOG_SUFFIX.get(thread_name, thread_name)
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Already set up
        return logger
    handler = _RichHandlerWithEmoji(
        emoji=emoji,
        show_time=bool(os.environ.get("SWE_AGENT_LOG_TIME", False)),
        show_path=False,
    )
    handler.setLevel(_STREAM_LEVEL)
    logger.setLevel(_STREAM_LEVEL)
    logger.addHandler(handler)
    logger.propagate = False
    _SET_UP_LOGGERS.add(name)
    for handler in _ADDITIONAL_HANDLERS:
        logger.addHandler(handler)
    return logger
