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
                          Peut être booléen ou flottant [0-1]
            
        Returns:
            Score d'activation [0-1]
        """
        if self._rf_size == 0:
            return 0.0
        
        # Convertir l'entrée en booléen si c'est un float
        if input_pattern.dtype in (np.float32, np.float64, float):
            input_bool = input_pattern > 0.5
        else:
            input_bool = input_pattern.astype(bool)
        
        # Intersection: combien de pixels du RF sont actifs dans l'entrée
        overlap = np.sum(self.receptive_field & input_bool)
        
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
        # Transition FIRING -> REFRACTORY (après 1 frame visible)
        if self.state == NeuronState.FIRING:
            self.state = NeuronState.REFRACTORY
            # Continue pour décrémenter le compteur ci-dessous
        
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
            self.total_spikes += 1
            self.last_spike_frame = current_frame
            
            # Reset et préparer la période réfractaire
            self.potential = 0.0
            self.refractory_counter = self.config.refractory_period + 1  # +1 pour le frame FIRING
            
            # L'état FIRING sera visible pendant ce frame
            # Le prochain step() passera en REFRACTORY
            self.state = NeuronState.FIRING
            
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


class NeuronStack:
    """Pile de couches de neurones empilables.
    
    Permet de créer une hiérarchie de N couches où chaque couche
    détecte des patterns de la couche inférieure:
    
    - Couche 0: Détecte les patterns de pixels (rétine)
    - Couche 1: Détecte les patterns de neurones de couche 0
    - Couche N: Détecte les compositions de neurones N-1
    
    Chaque couche peut avoir sa propre résolution et configuration.
    """
    
    def __init__(
        self,
        base_shape: tuple[int, int],
        num_layers: int = 3,
        config: Optional[GenesisConfig] = None,
        neuron_config: Optional[NeuronConfig] = None,
        reduction_factor: float = 0.5
    ):
        """Initialise la pile de couches.
        
        Args:
            base_shape: Dimensions (H, W) de la couche d'entrée (rétine)
            num_layers: Nombre de couches de neurones à créer
            config: Configuration de genèse (partagée ou personnalisée)
            neuron_config: Configuration des neurones
            reduction_factor: Facteur de réduction de résolution par couche
                             (1.0 = même taille, 0.5 = moitié)
        """
        self.base_shape = base_shape
        self.num_layers = num_layers
        self.config = config or GenesisConfig()
        self.neuron_config = neuron_config or NeuronConfig()
        self.reduction_factor = reduction_factor
        
        # Import ici pour éviter import circulaire
        from .temporal import TemporalCorrelator, CorrelationConfig
        from .groups import GroupDetector, GroupDetectorConfig
        
        # Créer les couches avec leurs corrélateurs
        self._layers: list[NeuronLayer] = []
        self._layer_shapes: list[tuple[int, int]] = []
        self._correlators: list[TemporalCorrelator] = []
        self._group_detectors: list[GroupDetector] = []
        
        current_shape = base_shape
        for layer_id in range(num_layers):
            self._layer_shapes.append(current_shape)
            self._layers.append(NeuronLayer(
                layer_id=layer_id,
                shape=current_shape,
                config=self.config,
                neuron_config=self.neuron_config,
            ))
            
            # Corrélateur temporel pour cette couche
            # Paramètres ajustés pour une meilleure détection de patterns visuels
            self._correlators.append(TemporalCorrelator(
                shape=current_shape,
                config=CorrelationConfig(
                    history_size=30,
                    min_overlap=0.4,           # Plus permissif (était 0.6)
                    min_occurrences=3,         # Plus rapide (était 5)
                    confidence_threshold=0.5,  # Plus sensible (était 0.6)
                )
            ))
            
            # Détecteur de groupes pour cette couche
            self._group_detectors.append(GroupDetector(
                config=GroupDetectorConfig(
                    min_group_size=2,          # Groupes plus petits (était 3)
                    connectivity=8,
                    track_history=30,
                )
            ))
            
            # Réduire la résolution pour la couche suivante
            if layer_id < num_layers - 1:
                h, w = current_shape
                new_h = max(8, int(h * reduction_factor))
                new_w = max(8, int(w * reduction_factor))
                current_shape = (new_h, new_w)
        
        # Statistiques
        self._frame_count = 0
        self._propagation_enabled = True
        self._auto_genesis = True  # Création automatique de neurones
    
    @property
    def layers(self) -> list[NeuronLayer]:
        """Accès aux couches."""
        return self._layers
    
    def get_layer(self, layer_id: int) -> NeuronLayer:
        """Récupère une couche par son ID."""
        if 0 <= layer_id < len(self._layers):
            return self._layers[layer_id]
        raise IndexError(f"Layer {layer_id} not found (0-{len(self._layers)-1})")
    
    @property
    def total_neurons(self) -> int:
        """Nombre total de neurones dans toutes les couches."""
        return sum(layer.neuron_count for layer in self._layers)
    
    def process(
        self,
        input_pattern: np.ndarray,
        propagate: bool = True,
        learn: bool = True
    ) -> list[np.ndarray]:
        """Traite une entrée à travers toutes les couches.
        
        Chaque couche:
        1. Détecte les groupes d'activation
        2. Cherche des patterns temporels récurrents
        3. Crée des neurones pour les patterns stables
        4. Propage les activations neuronales vers la couche supérieure
        
        Args:
            input_pattern: Pattern d'activation de l'entrée (H, W) booléen
            propagate: Si True, propage vers les couches supérieures
            learn: Si True, crée des neurones pour les patterns stables
            
        Returns:
            Liste des sorties de chaque couche
        """
        self._frame_count += 1
        outputs = []
        neurons_created_this_frame = []
        
        current_input = input_pattern
        
        for layer_idx, layer in enumerate(self._layers):
            # Adapter la taille si nécessaire
            if current_input.shape != layer.shape:
                current_input = self._resize_pattern(
                    current_input, layer.shape
                )
            
            # 1. Détecter les groupes d'activation dans l'entrée
            # Seuil adaptatif: moyenne + 0.5 * écart-type
            # Cela capture les pixels "plus actifs que la normale"
            mean_val = current_input.mean()
            std_val = current_input.std()
            threshold = max(0.1, mean_val + 0.5 * std_val)
            activation_mask = current_input > threshold
            
            groups = self._group_detectors[layer_idx].detect_groups(
                activation_mask, 
                slot=0, 
                frame=self._frame_count
            )
            
            # 2. Corrélation temporelle: chercher les patterns récurrents
            if learn and self._auto_genesis:
                self._correlators[layer_idx].process_groups(groups)
                
                # 3. Créer des neurones pour les patterns stables
                stable_patterns = self._correlators[layer_idx].stable_patterns
                for pattern in stable_patterns:
                    neuron = layer.create_neuron_from_pattern(pattern)
                    if neuron is not None:
                        neurons_created_this_frame.append((layer_idx, neuron))
            
            # 4. Traiter avec les neurones existants
            output = layer.process(current_input)
            outputs.append(output)
            
            # Préparer l'entrée pour la couche suivante
            # La sortie = neurones qui ont spiké (leurs champs récepteurs)
            if propagate and layer_idx < len(self._layers) - 1:
                current_input = output
        
        return outputs
    
    @property
    def auto_genesis(self) -> bool:
        """Activation de la création automatique de neurones."""
        return self._auto_genesis
    
    @auto_genesis.setter
    def auto_genesis(self, value: bool):
        """Active/désactive la création automatique de neurones."""
        self._auto_genesis = value
        
        return outputs
    
    def _resize_pattern(
        self,
        pattern: np.ndarray,
        target_shape: tuple[int, int]
    ) -> np.ndarray:
        """Redimensionne un pattern booléen vers une nouvelle taille.
        
        Utilise un pooling ou upsampling selon le rapport de tailles.
        """
        src_h, src_w = pattern.shape
        dst_h, dst_w = target_shape
        
        if (src_h, src_w) == (dst_h, dst_w):
            return pattern
        
        # Convertir en float pour le redimensionnement
        pattern_float = pattern.astype(np.float32)
        
        # Calculer les ratios
        h_ratio = src_h / dst_h
        w_ratio = src_w / dst_w
        
        # Créer la sortie
        output = np.zeros(target_shape, dtype=bool)
        
        for y in range(dst_h):
            for x in range(dst_w):
                # Région source correspondante
                src_y1 = int(y * h_ratio)
                src_y2 = min(src_h, int((y + 1) * h_ratio) + 1)
                src_x1 = int(x * w_ratio)
                src_x2 = min(src_w, int((x + 1) * w_ratio) + 1)
                
                # Max pooling (un pixel actif = région active)
                region = pattern[src_y1:src_y2, src_x1:src_x2]
                output[y, x] = np.any(region)
        
        return output
    
    def create_neurons_from_patterns(
        self,
        layer_id: int,
        patterns: list[TemporalPattern]
    ) -> list[Neuron]:
        """Crée des neurones dans une couche à partir de patterns.
        
        Args:
            layer_id: Couche cible
            patterns: Liste de patterns stables
            
        Returns:
            Liste des neurones créés
        """
        layer = self.get_layer(layer_id)
        created = []
        
        for pattern in patterns:
            neuron = layer.create_neuron_from_pattern(pattern)
            if neuron is not None:
                created.append(neuron)
        
        return created
    
    def get_all_active_neurons(self) -> dict[int, list[Neuron]]:
        """Retourne tous les neurones qui ont spiké récemment.
        
        Returns:
            {layer_id: [neurons qui ont spiké]}
        """
        active = {}
        for layer in self._layers:
            active_in_layer = [
                n for n in layer.neurons
                if n.last_spike_frame == layer._frame_count
            ]
            if active_in_layer:
                active[layer.layer_id] = active_in_layer
        return active
    
    def get_stats(self) -> dict:
        """Statistiques globales de la pile."""
        layer_stats = [layer.get_stats() for layer in self._layers]
        correlator_stats = [
            {
                'pattern_count': c.pattern_count,
                'stable_patterns': len(c.stable_patterns),
            }
            for c in self._correlators
        ]
        return {
            'num_layers': self.num_layers,
            'total_neurons': self.total_neurons,
            'frame_count': self._frame_count,
            'auto_genesis': self._auto_genesis,
            'layers': layer_stats,
            'correlators': correlator_stats,
            'neurons_per_layer': [s['neuron_count'] for s in layer_stats],
            'spikes_per_layer': [s['total_spikes'] for s in layer_stats],
            'patterns_per_layer': [c['pattern_count'] for c in correlator_stats],
            'stable_patterns_per_layer': [c['stable_patterns'] for c in correlator_stats],
        }
    
    def reset(self):
        """Réinitialise l'état de tous les neurones."""
        for layer in self._layers:
            layer.reset()
    
    def clear(self):
        """Supprime tous les neurones de toutes les couches."""
        for layer in self._layers:
            layer.clear()
        self._frame_count = 0


def visualize_stack(
    stack: NeuronStack,
    show_potentials: bool = True,
    max_width: int = 800
) -> np.ndarray:
    """Visualise toutes les couches d'une pile.
    
    Affiche les couches empilées verticalement avec la couche 0 en bas.
    
    Args:
        stack: Pile de neurones
        show_potentials: Afficher les potentiels
        max_width: Largeur maximale de l'image
        
    Returns:
        Image RGB (H, W, 3) uint8
    """
    import cv2
    
    visualizations = []
    
    # Visualiser chaque couche (de haut en bas = N à 0)
    for layer in reversed(stack.layers):
        viz = visualize_neurons(layer, show_potentials)
        
        # Redimensionner pour la largeur max
        h, w = viz.shape[:2]
        if w < max_width:
            scale = max_width / w
            new_h = int(h * scale)
            viz = cv2.resize(viz, (max_width, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Ajouter un label
        label_h = 20
        labeled = np.zeros((viz.shape[0] + label_h, viz.shape[1], 3), dtype=np.uint8)
        labeled[label_h:, :, :] = viz
        
        # Texte du label
        cv2.putText(
            labeled, 
            f"Layer {layer.layer_id}: {layer.neuron_count} neurons, {layer.shape}",
            (5, 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.4, 
            (200, 200, 200), 
            1
        )
        
        visualizations.append(labeled)
    
    # Empiler verticalement
    return np.vstack(visualizations)

