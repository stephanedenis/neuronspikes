"""
NeuronSpikes - Réseau de neurones à impulsions déterministe.

Un système SNN ultra-léger, sans hasard, accéléré par GPU.
"""

__all__ = [
    "SpikingModel",
    "RetinaLayer",
    "RetinaConfig", 
    "create_retina",
    "BIT_REVERSAL_LUT",
    "INTENSITY_TO_SPIKES",
]

from .model import SpikingModel
from .retina import RetinaLayer, RetinaConfig, create_retina
from .lut import BIT_REVERSAL_LUT, INTENSITY_TO_SPIKES
