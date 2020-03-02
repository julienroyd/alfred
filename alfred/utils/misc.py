import logging
import sys
from math import floor, log10
import re


def create_logger(name, loglevel, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - {} - %(message)s'.format(name),
                                  datefmt='%d/%m/%Y %H:%M:%S', )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode='a'))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def create_new_filehandler(logger_name, logfile):

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - {} - %(message)s'.format(logger_name),
                                  datefmt='%d/%m/%Y %H:%M:%S', )

    file_handler = logging.FileHandler(logfile, mode='a')
    file_handler.setFormatter(formatter)

    return file_handler


def round_to_two(x):
    return round(x, -int(floor(log10(abs(x))) - 1))


def keep_two_signif_digits(x):
    return round(x, -int(floor(log10(abs(x))) - 1))


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)
