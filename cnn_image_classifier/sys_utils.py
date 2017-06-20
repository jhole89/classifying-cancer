import sys
import logging


def graceful_exit():
    logging.error("Invalid input, now exiting program.")
    sys.exit()
