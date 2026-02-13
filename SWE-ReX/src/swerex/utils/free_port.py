import socket
import time
from threading import Lock

from swerex.utils.log import get_logger

_REGISTERED_PORTS = set()
_REGISTERED_PORTS_LOCK = Lock()


def find_free_port(max_attempts: int = 10, sleep_between_attempts: float = 0.1) -> int:
    logger = get_logger("free_port")
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            with _REGISTERED_PORTS_LOCK:
                s.bind(("", 0))
                port = s.getsockname()[1]
                if port not in _REGISTERED_PORTS:
                    _REGISTERED_PORTS.add(port)
                    logger.debug(f"Found free port {port}")
                    return port
            logger.debug(f"Port {port} already registered, trying again after {sleep_between_attempts}s")
        time.sleep(sleep_between_attempts)
    msg = f"Failed to find a unique free port after {max_attempts} attempts"
    raise RuntimeError(msg)
