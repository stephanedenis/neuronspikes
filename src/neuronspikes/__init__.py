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
    # Pile de couches de neurones
    "NeuronStack",
    "visualize_stack",
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
    # Attention et mémoire
    "ZoomConfig",
    "ZoomLevel",
    "AttentionConfig",
    "VirtualZoom",
    "InhibitionMap",
    "AttentionMemory",
    "AttentionController",
    "GazeMemory",
    "PointOfInterest",
    # Fovéa couleur et mouvement
    "ColorChannel",
    "ColorFoveaConfig",
    "MotionVector",
    "TrackedObject",
    "ColorFovea",
    "ObjectTracker",
    "visualize_color_fovea",
    # Voies rétiniennes bio-inspirées
    "PathwayConfig",
    "MagnocellularPathway",
    "ParvocellularPathway",
    "LateralInhibition",
    "GaborFilterBank",
    "RetinalProcessor",
    # OpenCL Backend
    "OpenCLBackend",
    "get_opencl_backend",
    "is_opencl_available",
    "list_opencl_devices",
    # Visualisation 3D
    "NeuronVisualizer3D",
    "VisualizerConfig",
    "Camera3D",
]

from .model import SpikingModel
from .retina import RetinaLayer, RetinaConfig, create_retina
from .lut import BIT_REVERSAL_LUT, INTENSITY_TO_SPIKES
from .groups import GroupDetector, GroupDetectorConfig, ActivationGroup, visualize_groups
from .temporal import TemporalPattern, TemporalCorrelator, CorrelationConfig, visualize_patterns
from .genesis import Neuron, NeuronConfig, NeuronState, NeuronLayer, GenesisConfig, visualize_neurons, NeuronStack, visualize_stack
from .synapses import Synapse, SynapseType, SynapticConfig, SynapticNetwork, HebbianLayer
from .fabric import Fabric, FabricConfig, LearningCapability, Cortex
from .fovea import Fovea, FoveaConfig, GazePoint, PolarCell, StereoFovea, visualize_fovea
from .attention import (
    ZoomConfig, ZoomLevel, AttentionConfig, VirtualZoom,
    InhibitionMap, AttentionMemory, AttentionController,
    GazeMemory, PointOfInterest
)
from .color_fovea import (
    ColorChannel, ColorFoveaConfig, MotionVector,
    TrackedObject, ColorFovea, ObjectTracker, visualize_color_fovea
)
from .retinal_pathways import (
    PathwayConfig, MagnocellularPathway, ParvocellularPathway,
    LateralInhibition, GaborFilterBank, RetinalProcessor
)
from .opencl_backend import OpenCLBackend, get_opencl_backend, is_opencl_available, list_opencl_devices

# Import optionnel du visualiseur 3D (nécessite PyOpenGL)
try:
    from .visualizer_3d import NeuronVisualizer3D, VisualizerConfig, Camera3D
except ImportError:
    # PyOpenGL non installé
    NeuronVisualizer3D = None
    VisualizerConfig = None
    Camera3D = None
