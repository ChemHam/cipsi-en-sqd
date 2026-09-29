"""sqdlib — the shared pipeline behind the CIPSI-EN-SQD benchmarks."""

from . import config
from .config import Config, bind, load

__all__ = ["config", "Config", "bind", "load"]
