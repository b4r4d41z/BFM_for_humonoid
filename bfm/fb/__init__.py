"""Vanilla Forward-Backward representation learning and latent policies."""
from .model import FBModel, ModelConfig
from .agent import FBAgent, TrainConfig

__all__ = ["FBModel", "ModelConfig", "FBAgent", "TrainConfig"]
