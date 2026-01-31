"""
NeuronSpikes - Réseau de neurones à impulsions déterministe.

Un système SNN ultra-léger, sans hasard, accéléré par GPU.
"""

__all__ = [
    # Modèle de base
    "SpikingModel",
    # Rétine
    "RetinaLayer",
    "RetinaConfig", 
    "create_retina",
    # LUT
    "BIT_REVERSAL_LUT",
    "INTENSITY_TO_SPIKES",
    # Groupes
    "GroupDetector",
    "GroupDetectorConfig",
    "ActivationGroup",
    "visualize_groups",
    # Corrélations temporelles
    "TemporalPattern",
    "TemporalCorrelator",
    "CorrelationConfig",
    "visualize_patterns",
    # Genèse des neurones
    "Neuron",
    "NeuronConfig",
    "NeuronState",
    "NeuronLayer",
    "GenesisConfig",
    "visualize_neurons",
    # Synapses et apprentissage Hebbien
    "Synapse",
    "SynapseType",
    "SynapticConfig",
    "SynapticNetwork",
    "HebbianLayer",
    # Fabric et Cortex
    "Fabric",
    "FabricConfig",
    "LearningCapability",
    "Cortex",
    # Fovéa polaire
    "Fovea",
    "FoveaConfig",
    "GazePoint",
    "PolarCell",
    "StereoFovea",
    "visualize_fovea",
    # OpenCL Backend
    "OpenCLBackend",
    "get_opencl_backend",
    "is_opencl_available",
    "list_opencl_devices",
]

from .model import SpikingModel
from .retina import RetinaLayer, RetinaConfig, create_retina
from .lut import BIT_REVERSAL_LUT, INTENSITY_TO_SPIKES
from .groups import GroupDetector, GroupDetectorConfig, ActivationGroup, visualize_groups
from .temporal import TemporalPattern, TemporalCorrelator, CorrelationConfig, visualize_patterns
from .genesis import Neuron, NeuronConfig, NeuronState, NeuronLayer, GenesisConfig, visualize_neurons
from .synapses import Synapse, SynapseType, SynapticConfig, SynapticNetwork, HebbianLayer
from .fabric import Fabric, FabricConfig, LearningCapability, Cortex
from .fovea import Fovea, FoveaConfig, GazePoint, PolarCell, StereoFovea, visualize_fovea
from .opencl_backend import OpenCLBackend, get_opencl_backend, is_opencl_available, list_opencl_devices
