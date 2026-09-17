"""
Principal package for practice_sdoml.

It provides modelating, processing and ML utilities pipelines.
"""

from practice_sdoml import config
from practice_sdoml import dataset
from practice_sdoml import features
from practice_sdoml import plots
from practice_sdoml import modeling

from practice_sdoml.modeling.model import SimpleNet
from practice_sdoml.modeling.train import train, load_data

__all__ = [
    "config",
    "dataset",
    "features",
    "plots",
    "modeling",
    "SimpleNet",
    "train",
    "load_data",
]