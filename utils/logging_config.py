import argparse
import logging
import socket

def setup_logging(loglevel: str = "WARNING") -> None:
    """
    Configures application-wide logging.

    Args:
        loglevel: Log level string.
            Default is "WARNING".
            Valid options: DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Returns:
        None         
    """

    format_str = (
        f'[%(asctime)s {socket.gethostname()}] '
        '%(filename)s:%(funcName)s:%(lineno)s - %(levelname)s: %(message)s'
    )

    logging.basicConfig(
        level=getattr(logging, loglevel.upper(), logging.WARNING),
        format=format_str
    )

def add_logging_argument(parser: argparse.ArgumentParser) -> None:
    """
    Adds a --loglevel argument to an existing ArgumentParser.
    """
    parser.add_argument(
        "-l",
        "--loglevel",
        required=False,
        default="WARNING",
        choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],
        help="Set log level to DEBUG, INFO, WARNING, ERROR, or CRITICAL"
    )