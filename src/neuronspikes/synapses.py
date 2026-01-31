"""
Synapses et connexions entre neurones.

Inspiré de spikingcortex (Stéphane Denis, 2010s), ce module implémente:
- Connexions synaptiques explicites avec poids
- Règles d'apprentissage Hebbiennes
- Inhibition latérale (concurrence)
- Feedback (rétro-inhibition)

Principe Hebbien: "Neurons that fire together, wire together"
- Excitation: renforce les connexions entre neurones co-actifs
- Inhibition latérale: supprime les concurrents dans le même groupe
- Feedback: les neurones activés inhibent leurs sources
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Tuple
from enum import Enum, auto
import numpy as np

from .genesis import Neuron, NeuronState


class SynapseType(Enum):
    """Type de synapse."""
    EXCITATORY = auto()   # Excitation (poids positif)
    INHIBITORY = auto()   # Inhibition (poids négatif)
    FEEDBACK = auto()     # Rétro-inhibition


@dataclass
class Synapse:
    """Une connexion synaptique entre deux neurones.
    
    Attributes:
        source_id: ID du neurone source (pré-synaptique)
        target_id: ID du neurone cible (post-synaptique)
        weight: Poids de la connexion
        synapse_type: Type de synapse
        creation_frame: Frame de création
        last_active: Dernière frame où la synapse a été active
    """
    source_id: int
    target_id: int
    weight: float
    synapse_type: SynapseType = SynapseType.EXCITATORY
    creation_frame: int = 0
    last_active: int = 0
    
    @property
    def is_excitatory(self) -> bool:
        return self.weight > 0
    
    @property
    def is_inhibitory(self) -> bool:
        return self.weight < 0


@dataclass
class SynapticConfig:
    """Configuration des règles synaptiques.
    
    Attributes:
        learning_rate: Taux d'apprentissage Hebbien
        excitation_strength: Force initiale des connexions excitatrices
        inhibition_strength: Force initiale des connexions inhibitrices (négatif)
        feedback_ratio: Ratio feedback par rapport à l'excitation reçue
        weight_decay: Décroissance des poids non utilisés
        max_weight: Poids maximum absolu
        prune_threshold: Seuil sous lequel les synapses sont supprimées
    """
    learning_rate: float = 0.1
    excitation_strength: float = 0.3
    inhibition_strength: float = -0.5
    feedback_ratio: float = -0.8
    weight_decay: float = 0.999
    max_weight: float = 2.0
    prune_threshold: float = 0.01


class SynapticNetwork:
    """Réseau de connexions synaptiques.
    
    Gère les synapses entre neurones avec apprentissage Hebbien.
    """
    
    def __init__(self, config: Optional[SynapticConfig] = None):
        """Initialise le réseau synaptique.
        
        Args:
            config: Configuration des règles synaptiques
        """
        self.config = config or SynapticConfig()
        
        # Synapses indexées par (source_id, target_id)
        self._synapses: Dict[Tuple[int, int], Synapse] = {}
        
        # Index pour accès rapide
        self._efferents: Dict[int, Set[int]] = {}  # source -> {targets}
        self._afferents: Dict[int, Set[int]] = {}  # target -> {sources}
        
        # Compteur de frames
        self._frame_count = 0
        
        # Statistiques
        self._total_created = 0
        self._total_pruned = 0
    
    @property
    def synapse_count(self) -> int:
        """Nombre total de synapses."""
        return len(self._synapses)
    
    def get_synapse(self, source_id: int, target_id: int) -> Optional[Synapse]:
        """Récupère une synapse par ses IDs."""
        return self._synapses.get((source_id, target_id))
    
    def has_synapse(self, source_id: int, target_id: int) -> bool:
        """Vérifie si une synapse existe."""
        return (source_id, target_id) in self._synapses
    
    def create_synapse(
        self,
        source_id: int,
        target_id: int,
        weight: float,
        synapse_type: SynapseType = SynapseType.EXCITATORY
    ) -> Synapse:
        """Crée une nouvelle synapse.
        
        Args:
            source_id: ID du neurone source
            target_id: ID du neurone cible
            weight: Poids initial
            synapse_type: Type de synapse
            
        Returns:
            Synapse créée
        """
        key = (source_id, target_id)
        
        if key in self._synapses:
            # Mettre à jour le poids existant
            self._synapses[key].weight = weight
            return self._synapses[key]
        
        synapse = Synapse(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            synapse_type=synapse_type,
            creation_frame=self._frame_count,
            last_active=self._frame_count
        )
        
        self._synapses[key] = synapse
        
        # Mettre à jour les index
        if source_id not in self._efferents:
            self._efferents[source_id] = set()
        self._efferents[source_id].add(target_id)
        
        if target_id not in self._afferents:
            self._afferents[target_id] = set()
        self._afferents[target_id].add(source_id)
        
        self._total_created += 1
        
        return synapse
    
    def remove_synapse(self, source_id: int, target_id: int):
        """Supprime une synapse."""
        key = (source_id, target_id)
        
        if key not in self._synapses:
            return
        
        del self._synapses[key]
        
        # Nettoyer les index
        if source_id in self._efferents:
            self._efferents[source_id].discard(target_id)
        if target_id in self._afferents:
            self._afferents[target_id].discard(source_id)
        
        self._total_pruned += 1
    
    def get_efferents(self, source_id: int) -> List[Synapse]:
        """Récupère toutes les synapses sortantes d'un neurone."""
        if source_id not in self._efferents:
            return []
        
        return [
            self._synapses[(source_id, tid)]
            for tid in self._efferents[source_id]
            if (source_id, tid) in self._synapses
        ]
    
    def get_afferents(self, target_id: int) -> List[Synapse]:
        """Récupère toutes les synapses entrantes vers un neurone."""
        if target_id not in self._afferents:
            return []
        
        return [
            self._synapses[(sid, target_id)]
            for sid in self._afferents[target_id]
            if (sid, target_id) in self._synapses
        ]
    
    def compute_input(self, target_id: int, active_neurons: Set[int]) -> float:
        """Calcule l'entrée synaptique totale pour un neurone.
        
        Args:
            target_id: ID du neurone cible
            active_neurons: Ensemble des neurones actifs (qui ont spiké)
            
        Returns:
            Somme pondérée des entrées synaptiques
        """
        total = 0.0
        
        for synapse in self.get_afferents(target_id):
            if synapse.source_id in active_neurons:
                total += synapse.weight
                synapse.last_active = self._frame_count
        
        return total
    
    def learn_excitation(
        self,
        sources: List[int],
        target: int,
        strength: Optional[float] = None
    ):
        """Apprentissage Hebbien: renforce les connexions excitatrices.
        
        "Neurons that fire together, wire together"
        
        Args:
            sources: IDs des neurones sources (co-actifs)
            target: ID du neurone cible
            strength: Force totale souhaitée (divisée entre sources)
        """
        if not sources:
            return
        
        strength = strength or self.config.excitation_strength
        weight_per_source = strength / len(sources)
        
        for source_id in sources:
            if source_id == target:
                continue  # Pas d'auto-connexion
            
            synapse = self.get_synapse(source_id, target)
            
            if synapse is None:
                # Créer nouvelle synapse
                self.create_synapse(
                    source_id, target, weight_per_source,
                    SynapseType.EXCITATORY
                )
            else:
                # Renforcer synapse existante (LTP)
                if synapse.weight > 0:  # Seulement pour excitation
                    delta = (weight_per_source - synapse.weight) * self.config.learning_rate
                    synapse.weight = min(
                        self.config.max_weight,
                        synapse.weight + max(0, delta)  # Pas d'affaiblissement
                    )
    
    def learn_inhibition(
        self,
        source: int,
        targets: List[int],
        strength: Optional[float] = None
    ):
        """Développe l'inhibition latérale (concurrence).
        
        Le neurone source inhibe ses concurrents dans le même groupe.
        
        Args:
            source: ID du neurone source
            targets: IDs des neurones concurrents à inhiber
            strength: Force d'inhibition (négatif)
        """
        strength = strength or self.config.inhibition_strength
        
        for target_id in targets:
            if target_id == source:
                continue
            
            synapse = self.get_synapse(source, target_id)
            
            if synapse is None:
                self.create_synapse(
                    source, target_id, strength,
                    SynapseType.INHIBITORY
                )
            elif synapse.weight < 0:  # Seulement pour inhibition
                # Renforcer inhibition
                synapse.weight = min(synapse.weight, strength)
    
    def learn_feedback(self, target: int, threshold: float = 0.0):
        """Développe le feedback (rétro-inhibition).
        
        Le neurone cible inhibe ses sources excitatrices après avoir spiké.
        Évite les boucles de rétroaction positive.
        
        Args:
            target: ID du neurone qui a spiké
            threshold: Seuil d'activation de la source pour appliquer feedback
        """
        for synapse in self.get_afferents(target):
            if synapse.weight > threshold:  # Source excitatrice
                # Créer feedback inverse
                feedback_weight = synapse.weight * self.config.feedback_ratio
                
                feedback = self.get_synapse(target, synapse.source_id)
                if feedback is None:
                    self.create_synapse(
                        target, synapse.source_id, feedback_weight,
                        SynapseType.FEEDBACK
                    )
                elif feedback.weight < 0:  # Renforcer feedback existant
                    feedback.weight = min(feedback.weight, feedback_weight)
    
    def decay_weights(self):
        """Applique la décroissance des poids non utilisés."""
        to_prune = []
        
        for key, synapse in self._synapses.items():
            # Décroissance si non actif récemment
            if synapse.last_active < self._frame_count:
                synapse.weight *= self.config.weight_decay
                
                # Marquer pour suppression si trop faible
                if abs(synapse.weight) < self.config.prune_threshold:
                    to_prune.append(key)
        
        for source_id, target_id in to_prune:
            self.remove_synapse(source_id, target_id)
    
    def step(self):
        """Avance d'un pas de temps."""
        self._frame_count += 1
    
    def reset(self):
        """Réinitialise le réseau."""
        self._synapses.clear()
        self._efferents.clear()
        self._afferents.clear()
        self._frame_count = 0
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du réseau."""
        excitatory = sum(1 for s in self._synapses.values() if s.weight > 0)
        inhibitory = sum(1 for s in self._synapses.values() if s.weight < 0)
        
        return {
            'synapse_count': self.synapse_count,
            'excitatory': excitatory,
            'inhibitory': inhibitory,
            'total_created': self._total_created,
            'total_pruned': self._total_pruned,
            'frame_count': self._frame_count,
        }


class HebbianLayer:
    """Couche de neurones avec apprentissage Hebbien.
    
    Combine NeuronLayer et SynapticNetwork pour un apprentissage
    basé sur les co-activations.
    """
    
    def __init__(
        self,
        neurons: Dict[int, Neuron],
        synaptic_config: Optional[SynapticConfig] = None,
        enable_feedback: bool = True,
        enable_lateral_inhibition: bool = True
    ):
        """Initialise la couche Hebbienne.
        
        Args:
            neurons: Dictionnaire des neurones {id: Neuron}
            synaptic_config: Configuration synaptique
            enable_feedback: Activer le feedback
            enable_lateral_inhibition: Activer l'inhibition latérale
        """
        self.neurons = neurons
        self.network = SynapticNetwork(synaptic_config)
        self.enable_feedback = enable_feedback
        self.enable_lateral_inhibition = enable_lateral_inhibition
        
        # Historique des activations pour apprentissage
        self._active_history: List[Set[int]] = []
        self._history_size = 5
        
        # Frame courante
        self._frame_count = 0
    
    def process(
        self,
        input_activations: Dict[int, float],
        learn: bool = True
    ) -> Set[int]:
        """Traite les entrées et retourne les neurones qui ont spiké.
        
        Args:
            input_activations: Activations des neurones {id: activation}
            learn: Activer l'apprentissage
            
        Returns:
            Ensemble des IDs des neurones qui ont spiké
        """
        self._frame_count += 1
        self.network.step()
        
        spiked = set()
        
        # Neurones actifs au step précédent (pour calcul synaptique)
        prev_active = self._active_history[-1] if self._active_history else set()
        
        for neuron_id, neuron in self.neurons.items():
            # Activation directe
            direct_activation = input_activations.get(neuron_id, 0.0)
            
            # Entrée synaptique
            synaptic_input = self.network.compute_input(neuron_id, prev_active)
            
            # Activation totale
            total_activation = direct_activation + synaptic_input
            
            # Faire avancer le neurone
            if neuron.step(total_activation, self._frame_count):
                spiked.add(neuron_id)
        
        # Apprentissage
        if learn and spiked:
            self._learn_from_activity(spiked, prev_active)
        
        # Mettre à jour l'historique
        self._active_history.append(spiked)
        if len(self._active_history) > self._history_size:
            self._active_history.pop(0)
        
        # Décroissance des poids
        if self._frame_count % 100 == 0:
            self.network.decay_weights()
        
        return spiked
    
    def _learn_from_activity(self, spiked: Set[int], prev_active: Set[int]):
        """Applique les règles d'apprentissage Hebbien.
        
        Args:
            spiked: Neurones qui ont spiké maintenant
            prev_active: Neurones actifs au step précédent
        """
        for neuron_id in spiked:
            # 1. Renforcer excitation depuis sources actives
            sources = list(prev_active - {neuron_id})
            if sources:
                self.network.learn_excitation(sources, neuron_id)
            
            # 2. Feedback vers sources
            if self.enable_feedback:
                self.network.learn_feedback(neuron_id)
            
            # 3. Inhibition latérale
            if self.enable_lateral_inhibition:
                # Inhiber les autres neurones qui ont spiké en même temps
                concurrents = list(spiked - {neuron_id})
                if concurrents:
                    self.network.learn_inhibition(neuron_id, concurrents)
    
    def get_stats(self) -> dict:
        """Retourne les statistiques."""
        return {
            'neuron_count': len(self.neurons),
            'frame_count': self._frame_count,
            **self.network.get_stats()
        }
