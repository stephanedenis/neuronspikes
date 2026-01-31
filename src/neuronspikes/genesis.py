"""
Genèse des neurones - Création de neurones à partir des patterns récurrents.

Ce module implémente la création dynamique de neurones dans les couches
supérieures du réseau. Chaque neurone est créé à partir d'un pattern
temporel stable détecté par le corrélateur.

Principe:
- Un pattern stable = pixels qui s'activent ensemble de façon répétée
- Ce pattern devient le "champ récepteur" d'un nouveau neurone
- Le neurone s'active quand son pattern est reconnu dans l'entrée
- Aucun hasard: la création est déterministe basée sur les statistiques

Hiérarchie:
- Couche 0: Rétine (pixels)
- Couche 1: Neurones de premier niveau (détecteurs de patterns simples)
- Couche N: Neurones combinant les patterns des couches inférieures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum, auto
import numpy as np

from .temporal import TemporalPattern


class NeuronState(Enum):
    """État d'un neurone."""
    DORMANT = auto()      # Pas d'activité récente
    CHARGING = auto()     # Accumule de l'activation
    FIRING = auto()       # Émet un spike
    REFRACTORY = auto()   # Période réfractaire après spike


@dataclass
class NeuronConfig:
    """Configuration pour un neurone.
    
    Attributes:
        threshold: Seuil d'activation pour émettre un spike
        decay_rate: Taux de décroissance du potentiel par step
        refractory_period: Nombre de steps après un spike avant de pouvoir refirer
        learning_rate: Vitesse d'adaptation du champ récepteur
    """
    threshold: float = 0.8
    decay_rate: float = 0.1
    refractory_period: int = 3
    learning_rate: float = 0.01


@dataclass
class Neuron:
    """Un neurone artificiel avec champ récepteur.
    
    Le neurone surveille un pattern spécifique et s'active quand
    ce pattern est détecté dans l'entrée.
    
    Attributes:
        neuron_id: Identifiant unique
        layer: Couche à laquelle appartient le neurone
        receptive_field: Masque binaire du champ récepteur (H, W)
        config: Configuration du neurone
        birth_frame: Frame de création du neurone
        parent_pattern_id: ID du pattern qui a engendré ce neurone
    """
    neuron_id: int
    layer: int
    receptive_field: np.ndarray  # Masque booléen (H, W)
    config: NeuronConfig = field(default_factory=NeuronConfig)
    birth_frame: int = 0
    parent_pattern_id: int = -1
    
    # État interne
    potential: float = field(default=0.0, init=False)
    state: NeuronState = field(default=NeuronState.DORMANT, init=False)
    refractory_counter: int = field(default=0, init=False)
    
    # Statistiques
    total_spikes: int = field(default=0, init=False)
    last_spike_frame: int = field(default=-1, init=False)
    
    def __post_init__(self):
        """Calcule les propriétés dérivées."""
        if self.receptive_field is not None:
            self._rf_size = int(np.sum(self.receptive_field))
            self._rf_indices = np.argwhere(self.receptive_field)
        else:
            self._rf_size = 0
            self._rf_indices = np.array([])
    
    @property
    def centroid(self) -> tuple[float, float]:
        """Centre de masse du champ récepteur."""
        if self._rf_size == 0:
            return (0.0, 0.0)
        ys, xs = np.where(self.receptive_field)
        return (float(np.mean(xs)), float(np.mean(ys)))
    
    def compute_activation(self, input_pattern: np.ndarray) -> float:
        """Calcule l'activation du neurone pour une entrée donnée.
        
        Args:
            input_pattern: Pattern d'activation de l'entrée (même shape que RF)
            
        Returns:
            Score d'activation [0-1]
        """
        if self._rf_size == 0:
            return 0.0
        
        # Intersection: combien de pixels du RF sont actifs dans l'entrée
        overlap = np.sum(self.receptive_field & input_pattern)
        
        # Normaliser par la taille du RF
        return float(overlap / self._rf_size)
    
    def step(self, activation: float, current_frame: int) -> bool:
        """Effectue un pas de temps pour le neurone.
        
        Args:
            activation: Score d'activation [0-1]
            current_frame: Numéro du frame courant
            
        Returns:
            True si le neurone a émis un spike
        """
        # Gestion de la période réfractaire
        if self.state == NeuronState.REFRACTORY:
            self.refractory_counter -= 1
            if self.refractory_counter <= 0:
                self.state = NeuronState.DORMANT
            return False
        
        # Mise à jour du potentiel
        # Décroissance naturelle + nouvelle activation
        self.potential = (
            self.potential * (1 - self.config.decay_rate) +
            activation
        )
        
        # Vérifier le seuil
        if self.potential >= self.config.threshold:
            # SPIKE!
            self.state = NeuronState.FIRING
            self.total_spikes += 1
            self.last_spike_frame = current_frame
            
            # Reset et période réfractaire
            self.potential = 0.0
            self.refractory_counter = self.config.refractory_period
            self.state = NeuronState.REFRACTORY
            
            return True
        
        # Mise à jour de l'état
        if self.potential > 0.1:
            self.state = NeuronState.CHARGING
        else:
            self.state = NeuronState.DORMANT
        
        return False
    
    def reset(self):
        """Réinitialise l'état du neurone."""
        self.potential = 0.0
        self.state = NeuronState.DORMANT
        self.refractory_counter = 0


@dataclass
class GenesisConfig:
    """Configuration pour la genèse des neurones.
    
    Attributes:
        min_pattern_confidence: Confiance minimale d'un pattern pour créer un neurone
        min_pattern_occurrences: Occurrences minimales pour créer un neurone
        max_neurons_per_layer: Nombre maximum de neurones par couche
        neuron_merge_threshold: Chevauchement pour fusionner deux neurones
        prune_inactive_after: Frames d'inactivité avant suppression
    """
    min_pattern_confidence: float = 0.6
    min_pattern_occurrences: int = 10
    max_neurons_per_layer: int = 1000
    neuron_merge_threshold: float = 0.8
    prune_inactive_after: int = 600  # 10 secondes à 60 fps


class NeuronLayer:
    """Couche de neurones créés dynamiquement.
    
    Gère un ensemble de neurones avec leur création, mise à jour
    et suppression basées sur les patterns détectés.
    """
    
    def __init__(
        self,
        layer_id: int,
        shape: tuple[int, int],
        config: Optional[GenesisConfig] = None,
        neuron_config: Optional[NeuronConfig] = None
    ):
        """Initialise la couche.
        
        Args:
            layer_id: Identifiant de la couche
            shape: Dimensions (H, W) des entrées
            config: Configuration de genèse
            neuron_config: Configuration par défaut des neurones
        """
        self.layer_id = layer_id
        self.shape = shape
        self.config = config or GenesisConfig()
        self.neuron_config = neuron_config or NeuronConfig()
        
        # Neurones de cette couche
        self._neurons: dict[int, Neuron] = {}
        self._next_neuron_id = 0
        
        # Mapping pattern_id -> neuron_id pour éviter les doublons
        self._pattern_to_neuron: dict[int, int] = {}
        
        # Compteur de frames
        self._frame_count = 0
        
        # Statistiques
        self._total_neurons_created = 0
        self._total_neurons_pruned = 0
        self._total_spikes = 0
    
    @property
    def neuron_count(self) -> int:
        """Nombre de neurones actifs."""
        return len(self._neurons)
    
    @property
    def neurons(self) -> list[Neuron]:
        """Liste des neurones."""
        return list(self._neurons.values())
    
    def create_neuron_from_pattern(self, pattern: TemporalPattern) -> Optional[Neuron]:
        """Crée un neurone à partir d'un pattern temporel.
        
        Args:
            pattern: Pattern temporel stable
            
        Returns:
            Nouveau neurone ou None si création refusée
        """
        # Vérifier les critères de création
        if pattern.confidence < self.config.min_pattern_confidence:
            return None
        if pattern.occurrences < self.config.min_pattern_occurrences:
            return None
        
        # Vérifier si ce pattern a déjà un neurone
        if pattern.pattern_id in self._pattern_to_neuron:
            return None
        
        # Vérifier la limite de neurones
        if self.neuron_count >= self.config.max_neurons_per_layer:
            # Élaguer les plus inactifs
            self._prune_inactive_neurons()
            if self.neuron_count >= self.config.max_neurons_per_layer:
                return None
        
        # Vérifier le chevauchement avec les neurones existants
        for existing in self._neurons.values():
            overlap = self._compute_overlap(pattern.signature, existing.receptive_field)
            if overlap > self.config.neuron_merge_threshold:
                # Pattern trop similaire à un neurone existant
                # Enregistrer le mapping et ne pas créer
                self._pattern_to_neuron[pattern.pattern_id] = existing.neuron_id
                return None
        
        # Créer le neurone
        neuron = Neuron(
            neuron_id=self._next_neuron_id,
            layer=self.layer_id,
            receptive_field=pattern.signature.copy(),
            config=self.neuron_config,
            birth_frame=self._frame_count,
            parent_pattern_id=pattern.pattern_id,
        )
        
        self._neurons[neuron.neuron_id] = neuron
        self._pattern_to_neuron[pattern.pattern_id] = neuron.neuron_id
        self._next_neuron_id += 1
        self._total_neurons_created += 1
        
        return neuron
    
    def _compute_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calcule le chevauchement (IoU) entre deux masques."""
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def process(self, input_pattern: np.ndarray) -> np.ndarray:
        """Traite une entrée et retourne les activations des neurones.
        
        Args:
            input_pattern: Masque d'activation de l'entrée (H, W)
            
        Returns:
            Masque des neurones qui ont spiké (H, W)
        """
        self._frame_count += 1
        
        # Masque de sortie
        output = np.zeros(self.shape, dtype=bool)
        
        for neuron in self._neurons.values():
            # Calculer l'activation
            activation = neuron.compute_activation(input_pattern)
            
            # Faire avancer le neurone d'un pas
            spiked = neuron.step(activation, self._frame_count)
            
            if spiked:
                self._total_spikes += 1
                # Marquer le champ récepteur comme actif dans la sortie
                output |= neuron.receptive_field
        
        return output
    
    def _prune_inactive_neurons(self):
        """Supprime les neurones inactifs depuis trop longtemps."""
        to_prune = []
        
        for neuron_id, neuron in self._neurons.items():
            frames_since_spike = self._frame_count - neuron.last_spike_frame
            
            # Si jamais spiké ou inactif depuis trop longtemps
            if neuron.last_spike_frame < 0 or frames_since_spike > self.config.prune_inactive_after:
                to_prune.append(neuron_id)
        
        for neuron_id in to_prune:
            del self._neurons[neuron_id]
            self._total_neurons_pruned += 1
            
            # Nettoyer le mapping
            patterns_to_remove = [
                pid for pid, nid in self._pattern_to_neuron.items()
                if nid == neuron_id
            ]
            for pid in patterns_to_remove:
                del self._pattern_to_neuron[pid]
    
    def get_activation_map(self) -> np.ndarray:
        """Retourne une carte des potentiels de tous les neurones.
        
        Returns:
            Image (H, W) float32 avec les potentiels cumulés
        """
        activation_map = np.zeros(self.shape, dtype=np.float32)
        
        for neuron in self._neurons.values():
            # Ajouter le potentiel du neurone sur son RF
            activation_map += neuron.receptive_field.astype(np.float32) * neuron.potential
        
        return activation_map
    
    def get_neuron_map(self) -> np.ndarray:
        """Retourne une carte des IDs de neurones.
        
        Returns:
            Image (H, W) int32 avec neuron_id + 1 (0 = pas de neurone)
        """
        neuron_map = np.zeros(self.shape, dtype=np.int32)
        
        for neuron in self._neurons.values():
            neuron_map[neuron.receptive_field] = neuron.neuron_id + 1
        
        return neuron_map
    
    def reset(self):
        """Réinitialise l'état de tous les neurones (sans les supprimer)."""
        for neuron in self._neurons.values():
            neuron.reset()
    
    def clear(self):
        """Supprime tous les neurones."""
        self._neurons.clear()
        self._pattern_to_neuron.clear()
        self._frame_count = 0
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de la couche."""
        return {
            'layer_id': self.layer_id,
            'neuron_count': self.neuron_count,
            'total_created': self._total_neurons_created,
            'total_pruned': self._total_neurons_pruned,
            'total_spikes': self._total_spikes,
            'frame_count': self._frame_count,
        }


def visualize_neurons(
    neuron_layer: NeuronLayer,
    show_potentials: bool = True
) -> np.ndarray:
    """Visualise les neurones d'une couche.
    
    Args:
        neuron_layer: Couche de neurones à visualiser
        show_potentials: Si True, montre les potentiels, sinon les RFs
        
    Returns:
        Image RGB (H, W, 3) uint8
    """
    h, w = neuron_layer.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Couleurs pour les neurones
    colors = np.array([
        [255, 100, 100],
        [100, 255, 100],
        [100, 100, 255],
        [255, 255, 100],
        [255, 100, 255],
        [100, 255, 255],
        [255, 180, 100],
        [180, 100, 255],
    ], dtype=np.uint8)
    
    for neuron in neuron_layer.neurons:
        color = colors[neuron.neuron_id % len(colors)]
        
        if show_potentials:
            # Moduler par le potentiel
            brightness = 0.3 + 0.7 * min(1.0, neuron.potential)
            output[neuron.receptive_field] = (color * brightness).astype(np.uint8)
        else:
            output[neuron.receptive_field] = color
    
    return output
