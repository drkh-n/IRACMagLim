import logging
import os
from logging.handlers import RotatingFileHandler


def configure_logging(level: str = 'DEBUG', log_file: str = 'logs/photometrypy.log') -> None:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    numeric_level = getattr(logging, level.upper(), logging.DEBUG)
    root.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding='utf-8', mode='w')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Suppress verbose matplotlib logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


