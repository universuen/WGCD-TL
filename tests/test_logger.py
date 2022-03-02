import context

from src import config
from src.logger import Logger


if __name__ == '__main__':
    config.logging_config.level = 'DEBUG'
    logger = Logger('test')
    logger.debug('debug test')
    logger.info('info test')
    logger.warning('warning test')
    logger.error('error test')
    logger.critical('critical test')
    logger.turn_off()
    logger.debug("""you can't see me""")
