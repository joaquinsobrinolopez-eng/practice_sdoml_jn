"""
Paquete principal de practice_sdoml.
"""
from practice_sdoml import config
from practice_sdoml import dataset
from practice_sdoml import features
from practice_sdoml import plots
from practice_sdoml import modeling

from practice_sdoml.dataset import DiabetesDataset, get_dataloader
from practice_sdoml.modeling.model import SimpleNet
from practice_sdoml.modeling.train import train

__all__ = [
    "config",
    "dataset",
    "features",
    "plots",
    "modeling",
    "DiabetesDataset",
    "get_dataloader",
    "SimpleNet",
    "train",
]
