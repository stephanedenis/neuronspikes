"""
Couche Rétine - Première couche sensorielle du réseau NeuronSpikes.

Cette couche convertit une entrée visuelle (image monochrome) en trains
d'impulsions neuronales à haute fréquence (jusqu'à 15360 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from .lut import INTENSITY_TO_SPIKES, BIT_REVERSAL_LUT


@dataclass
class RetinaConfig:
    """Configuration de la couche rétine.
    
    Attributes:
        width: Largeur en pixels
        height: Hauteur en pixels
        threshold: Seuil d'activation (très bas pour haute sensibilité)
        decay: Facteur de décroissance de la charge (non utilisé si decay=0)
        fps: Frames par seconde de la source
    """
    width: int = 64
    height: int = 64
    threshold: float = 1.0  # Seuil minimal pour activation rapide
    decay: float = 0.0  # Pas de décroissance par défaut (tout est intégré)
    fps: int = 60


@dataclass
class RetinaState:
    """État interne de la couche rétine.
    
    Attributes:
        charges: Charge accumulée par neurone (H, W)
        frame_index: Index de la frame courante
        slot_index: Index du slot temporel dans la frame (0-255)
        total_spikes: Compteur total d'impulsions émises
    """
    charges: NDArray[np.float32] = field(default=None)
    frame_index: int = 0
    slot_index: int = 0
    total_spikes: int = 0
    
    def __post_init__(self):
        if self.charges is None:
            self.charges = np.zeros((64, 64), dtype=np.float32)


class RetinaLayer:
    """Couche rétine - conversion intensité → impulsions.
    
    Cette couche implémente le premier niveau du réseau SNN:
    - Reçoit des images monochromes 8-bit
    - Convertit chaque pixel en train d'impulsions via LUT bit-reversal
    - Émet des frames d'activation à haute fréquence (jusqu'à 15360 Hz)
    
    Le système est entièrement déterministe - même entrée = même sortie.
    """
    
    def __init__(self, config: Optional[RetinaConfig] = None):
        """Initialise la couche rétine.
        
        Args:
            config: Configuration de la couche (défaut: 64x64 @ 60fps)
        """
        self.config = config or RetinaConfig()
        self.state = RetinaState(
            charges=np.zeros((self.config.height, self.config.width), 
                           dtype=np.float32)
        )
        
        # Cache pour les trains d'impulsions de la frame courante
        self._current_spike_trains: Optional[NDArray[np.bool_]] = None
        
        # Statistiques
        self.stats = {
            'frames_processed': 0,
            'total_spikes': 0,
            'activation_groups': 0,
        }
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Forme de la couche (height, width)."""
        return (self.config.height, self.config.width)
    
    @property
    def max_frequency(self) -> int:
        """Fréquence maximale d'impulsions en Hz."""
        return self.config.fps * 256  # 60 * 256 = 15360 Hz
    
    @property
    def slot_duration_us(self) -> float:
        """Durée d'un slot temporel en microsecondes."""
        return (1_000_000 / self.config.fps) / 256  # ~65.1 µs @ 60fps
    
    def process_frame(self, frame: NDArray[np.uint8]) -> None:
        """Traite une nouvelle frame d'image.
        
        Args:
            frame: Image monochrome (H, W) en uint8
            
        Raises:
            ValueError: Si la forme de l'image ne correspond pas
        """
        if frame.shape != self.shape:
            raise ValueError(
                f"Frame shape {frame.shape} doesn't match layer shape {self.shape}"
            )
        
        # Convertir l'image en trains d'impulsions via la LUT
        # Résultat: (H, W, 256) booléens
        self._current_spike_trains = INTENSITY_TO_SPIKES[frame]
        
        # Reset de l'index de slot pour la nouvelle frame
        self.state.slot_index = 0
        self.state.frame_index += 1
        self.stats['frames_processed'] += 1
    
    def get_activations(self, slot: Optional[int] = None) -> NDArray[np.bool_]:
        """Obtient les activations pour un slot temporel donné.
        
        Args:
            slot: Index du slot (0-255). Si None, utilise le slot courant.
            
        Returns:
            NDArray[np.bool_]: Masque d'activation (H, W)
        """
        if self._current_spike_trains is None:
            return np.zeros(self.shape, dtype=np.bool_)
        
        if slot is None:
            slot = self.state.slot_index
        
        return self._current_spike_trains[:, :, slot]
    
    def step(self) -> NDArray[np.bool_]:
        """Avance d'un slot temporel et retourne les activations.
        
        Cette méthode fait progresser le temps d'un slot (~65µs @ 60fps)
        et retourne les neurones qui s'activent à ce moment.
        
        Returns:
            NDArray[np.bool_]: Masque d'activation (H, W)
        """
        activations = self.get_activations()
        
        # Statistiques
        spike_count = int(np.sum(activations))
        self.state.total_spikes += spike_count
        self.stats['total_spikes'] += spike_count
        
        if spike_count > 0:
            self.stats['activation_groups'] += 1
        
        # Avancer au slot suivant
        self.state.slot_index = (self.state.slot_index + 1) % 256
        
        return activations
    
    def run_frame(self) -> list[NDArray[np.bool_]]:
        """Exécute tous les slots d'une frame et retourne toutes les activations.
        
        Utile pour le debug et la visualisation.
        
        Returns:
            Liste de 256 masques d'activation
        """
        activations = []
        for _ in range(256):
            activations.append(self.step())
        return activations
    
    def get_activation_pattern(self) -> NDArray[np.uint8]:
        """Retourne le nombre d'impulsions par neurone pour la frame courante.
        
        Returns:
            NDArray[np.uint8]: Compte d'impulsions (H, W) - équivaut à l'intensité
        """
        if self._current_spike_trains is None:
            return np.zeros(self.shape, dtype=np.uint8)
        
        return np.sum(self._current_spike_trains, axis=2, dtype=np.uint8)
    
    def reset(self) -> None:
        """Réinitialise l'état de la couche."""
        self.state = RetinaState(
            charges=np.zeros((self.config.height, self.config.width),
                           dtype=np.float32)
        )
        self._current_spike_trains = None
        self.stats = {
            'frames_processed': 0,
            'total_spikes': 0,
            'activation_groups': 0,
        }


def create_retina(width: int = 64, height: int = 64, fps: int = 60) -> RetinaLayer:
    """Factory pour créer une couche rétine avec les paramètres courants.
    
    Args:
        width: Largeur en pixels
        height: Hauteur en pixels
        fps: Frames par seconde
        
    Returns:
        RetinaLayer configurée
    """
    config = RetinaConfig(width=width, height=height, fps=fps)
    return RetinaLayer(config)
