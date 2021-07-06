import logging
import logging.config


def enable_logging(level='INFO'):
    '''Enables global logging
    Parameters:
    level: string
        Sets the level of global logging.
        One of 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'
    '''

    logging_config = dict(
        version=1,
        disable_existing_loggers=False,
        formatters={
            'f': {
                'format':
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
            }
        },
        handlers={
            'sh': {
                'class': 'logging.StreamHandler',
                'formatter': 'f',
                'stream': 'ext://sys.stdout'
            }
        },
        root={
            'handlers': ['sh'],
            'level': logging.getLevelName(level)
        },
    )
    logging.config.dictConfig(logging_config)

    logger.info("Logging enabled...")


# Create shared logger object
logger = logging.getLogger(__name__)
enable_logging()
