from logging.handlers import TimedRotatingFileHandler
import logging
import os
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG':    '\033[90m',
        'INFO':     '\033[92m',
        'WARNING':  '\033[93m',
        'ERROR':    '\033[91m',
        'CRITICAL': '\033[1;91m'
    }
    RESET = '\033[0m'

    def format(self, record):
        log_fmt = (
            "%(asctime)-19s | %(levelname)-8s | %(module)-15s | "
            "%(funcName)-30s | line %(lineno)-4d | %(message)s"
        )
        color = self.COLORS.get(record.levelname, self.RESET)
        formatter = logging.Formatter(color + log_fmt + self.RESET, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class LoggerService:
    def __init__(self, log_file: str = "app.log", level: int = logging.INFO):
        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        self.logger.propagate = False

        log_dirs = "./logs"
        os.makedirs(log_dirs, exist_ok = True)

        log_path = os.path.join(log_dirs, log_file)
        if not self.logger.handlers:
            plain_formatter = logging.Formatter(
                fmt="%(asctime)-19s | %(levelname)-8s | %(module)-15s | %(funcName)-30s | line %(lineno)-4d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            file_handler = TimedRotatingFileHandler(
                filename=log_path,
                when='midnight',
                interval=1,
                backupCount=7,
                encoding='utf-8'
            )
            file_handler.setFormatter(plain_formatter)
            file_handler.setLevel(level)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(ColorFormatter())
            console_handler.setLevel(logging.DEBUG)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger