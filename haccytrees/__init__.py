"""Top-level package for MPIPartition."""

__author__ = """Michael Buehlmann"""
__email__ = "buehlmann.michi@gmail.com"
__version__ = "0.10.0"

from .simulations import Simulation
from . import mergertrees
from . import coretrees

__all__ = ["Simulation"]
