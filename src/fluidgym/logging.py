"""Logging setup for FluidGym."""

import logging

LOGGER_NAMESPACE = "fluidgym"


def get_logger(name: str) -> logging.Logger:
    """Get a logger below the FluidGym namespace.

    Parameters
    ----------
    name: str
        Logger name. Prefixed with ``fluidgym.`` unless it already is.

    Returns
    -------
    logging.Logger
        The namespaced logger.
    """
    if name == LOGGER_NAMESPACE or name.startswith(f"{LOGGER_NAMESPACE}."):
        return logging.getLogger(name)

    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")


def set_verbosity(level: int | str = logging.INFO) -> None:
    """Set the log level of every FluidGym logger.

    FluidGym is quiet by default (``WARNING``).

    Parameters
    ----------
    level: int | str
        A :mod:`logging` level, either numeric or by name, e.g. ``"INFO"``.
    """
    logging.getLogger(LOGGER_NAMESPACE).setLevel(level)


logging.getLogger(LOGGER_NAMESPACE).addHandler(logging.NullHandler())
set_verbosity(logging.WARNING)
