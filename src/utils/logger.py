import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from typing import Optional

class Logger:
    def __init__(self, name: Optional[str] = None, level: int = logging.INFO):
        self.theme = Theme({
            "info": "bold green",
            "warning": "bold yellow",
            "error": "bold red",
            "critical": "bold white on red",
        })

        self.console = Console(theme=self.theme)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if self.logger.handlers: self.logger.handlers.clear()

        # Rich handler for console
        rich_handler = RichHandler(console=self.console, rich_tracebacks=True, markup=True, show_time=False, show_path=False)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        rich_handler.setFormatter(formatter)
        self.logger.addHandler(rich_handler)

    def debug(self, message: str): self.logger.debug(message)

    def info(self, message: str): self.logger.info(message)

    def warning(self, message: str): self.logger.warning(message)

    def error(self, message: str): self.logger.error(message)

    def critical(self, message: str): self.logger.critical(message)

    def exception(self, message: str): self.logger.exception(message)