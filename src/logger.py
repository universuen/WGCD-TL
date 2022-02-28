import logging

from . import config


class Logger(logging.Logger):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        sh = logging.StreamHandler()
        sh.setFormatter(config.logging_config.stream_formatter)
        sh.setLevel(config.logging_config.level)
        self.addHandler(sh)

        fh = logging.FileHandler(config.path_config.logs / f'{name}.log')
        fh.setFormatter(config.logging_config.file_formatter)
        fh.setLevel(config.logging_config.level)
        self.addHandler(fh)
