import logging
import sys


def getLogger(name: str, level=logging.INFO):
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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
