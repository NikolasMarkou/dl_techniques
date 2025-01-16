import logging

# ---------------------------------------------------------------------
# setup logger
# ---------------------------------------------------------------------

LOGGER_FORMAT = \
    "%(asctime)s %(levelname)-4s %(filename)s:%(funcName)s:%(lineno)s] " \
    "%(message)s"

logging.basicConfig(level=logging.INFO,
                    format=LOGGER_FORMAT)
logging.getLogger("dl").setLevel(logging.INFO)
logger = logging.getLogger("dl")