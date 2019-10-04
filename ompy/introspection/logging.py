import logging
import sys
from typing import List


def get_logger(name: str, level=logging.INFO):
    """ Set the logger of file `name` to the `info` level

    Args:
        name: The name of the file or module to retrieve
            the logger from.
        level: The log level to set. Can be any argument
            that logging.setLevel accepts.
    Returns:
        The logger requested
    """
    # root = logging.getLogger()
    # root.handlers = []
    # logging.basicConfig(level=logging.DEBUG)
    if not name.startswith("ompy"):
        name = "ompy." + name
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmtstring = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmtstring)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def available_loggers() -> List[str]:
    """ Get all available OMpy loggers """
    # Is the code from cython source
    existing = logging.root.manager.loggerDict.keys()
    ompy_loggers = [k.split('.')[1] for k in existing
                    if 'ompy' in k and len(k) > 5]
    return ompy_loggers

