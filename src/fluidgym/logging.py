"""Logging setup for FluidGym.

Every logger in the package lives below the ``fluidgym`` namespace, so that a
single call to :func:`set_verbosity` controls the log level of the whole
library. As a library, FluidGym is quiet by default: the namespace is set to
``WARNING`` on import and a ``NullHandler`` is attached.
"""

import logging

LOGGER_NAMESPACE = "fluidgym"
#: Namespace of the solver loggers (the phipict package)
SOLVER_LOGGER_NAMESPACE = "phipict"


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
    """Set the log level of every FluidGym and phipict logger.

    FluidGym is quiet by default (``WARNING``): as a library it does not emit
    progress output unless asked to. Call this once, e.g. at the top of a
    runscript, to turn on the solver, environment and AMG INFO logs. The solver
    logs live in the ``phipict`` namespace and are set to the same level.

    Parameters
    ----------
    level: int | str
        A :mod:`logging` level, either numeric or by name, e.g. ``"INFO"``.
    """
    logging.getLogger(LOGGER_NAMESPACE).setLevel(level)
    logging.getLogger(SOLVER_LOGGER_NAMESPACE).setLevel(level)


# Library default: quiet, and never complain about a missing handler
logging.getLogger(LOGGER_NAMESPACE).addHandler(logging.NullHandler())
set_verbosity(logging.WARNING)
