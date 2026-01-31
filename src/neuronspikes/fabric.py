"""
Fabric - Groupes de neurones avec propriétés partagées.

Inspiré de spikingcortex (Stéphane Denis, 2010s).

Un Fabric est la plus fine granularité de l'architecture corticale.
Il définit les affinités, taux et seuils d'un groupe de neurones.

Caractéristiques:
- Groupe de neurones avec configuration commune
- Capacités d'apprentissage configurables par groupe
- Working set pour le traitement actif
- Gestion du cycle de vie des neurones
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional
from enum import Enum, auto

from .genesis import Neuron, NeuronConfig, NeuronState
from .synapses import SynapticNetwork, SynapticConfig, HebbianLayer


class LearningCapability(Enum):
    """Capacités d'apprentissage d'un Fabric."""
    NONE = 0
    LEARN_CONCEPTS = 1      # Peut créer de nouveaux neurones
    DEVELOP_CONCURRENCY = 2  # Peut développer inhibition latérale
    DEVELOP_FEEDBACK = 4     # Peut développer rétro-inhibition
    ALL = 7                  # Toutes les capacités


@dataclass
class FabricConfig:
    """Configuration d'un Fabric.
    
    Attributes:
        name: Nom unique du fabric
        learning_capabilities: Capacités d'apprentissage (bitmask)
        max_neurons: Nombre maximum de neurones
        threshold: Seuil par défaut pour les neurones
        leak: Taux de fuite (decay) des potentiels
        spike_effect: Effet du spike (0=reset complet)
        moderation_period: Frames entre apprentissages
    """
    name: str = "default"
    learning_capabilities: int = LearningCapability.ALL.value
    max_neurons: int = 1000
    threshold: float = 1.0
    leak: float = 0.1  # 0.9 dans C# = 1-0.9 = 0.1 decay
    spike_effect: float = 0.0
    moderation_period: int = 1
    
    @property
    def can_learn_concepts(self) -> bool:
        return bool(self.learning_capabilities & LearningCapability.LEARN_CONCEPTS.value)
    
    @property
    def can_develop_concurrency(self) -> bool:
        return bool(self.learning_capabilities & LearningCapability.DEVELOP_CONCURRENCY.value)
    
    @property
    def can_develop_feedback(self) -> bool:
        return bool(self.learning_capabilities & LearningCapability.DEVELOP_FEEDBACK.value)


@dataclass
class Fabric:
    """Groupe de neurones avec propriétés partagées.
    
    Un Fabric gère un ensemble de neurones qui partagent:
    - Configuration de seuil et decay
    - Capacités d'apprentissage
    - Réseau synaptique commun
    
    Attributes:
        config: Configuration du fabric
        neurons: Dictionnaire des neurones {id: Neuron}
        working_set: Ensemble des neurones actifs
    """
    config: FabricConfig = field(default_factory=FabricConfig)
    
    # État interne
    neurons: Dict[int, Neuron] = field(default_factory=dict)
    working_set: Set[int] = field(default_factory=set)
    
    # Réseau synaptique partagé
    network: SynapticNetwork = field(init=False)
    
    # Historique des spikes
    _previously_spiked: Set[int] = field(default_factory=set, init=False)
    _newly_spiked: Set[int] = field(default_factory=set, init=False)
    
    # Compteurs
    _frame_count: int = field(default=0, init=False)
    _next_neuron_id: int = field(default=0, init=False)
    _not_learning_count: int = field(default=0, init=False)
    _ready: bool = field(default=True, init=False)
    
    def __post_init__(self):
        """Initialise le réseau synaptique."""
        self.network = SynapticNetwork(SynapticConfig())
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def neuron_count(self) -> int:
        return len(self.neurons)
    
    @property
    def can_add_neuron(self) -> bool:
        return len(self.neurons) < self.config.max_neurons
    
    def add_neuron(self, neuron: Neuron) -> bool:
        """Ajoute un neurone au fabric.
        
        Args:
            neuron: Neurone à ajouter
            
        Returns:
            True si ajouté, False si limite atteinte
        """
        if not self.can_add_neuron:
            return False
        
        self.neurons[neuron.neuron_id] = neuron
        self.working_set.add(neuron.neuron_id)
        return True
    
    def create_neuron(
        self,
        receptive_field,
        birth_frame: int = 0
    ) -> Optional[Neuron]:
        """Crée un nouveau neurone dans le fabric.
        
        Args:
            receptive_field: Champ récepteur du neurone
            birth_frame: Frame de création
            
        Returns:
            Neurone créé ou None si limite atteinte
        """
        if not self.can_add_neuron:
            return None
        
        neuron_id = self._next_neuron_id
        self._next_neuron_id += 1
        
        config = NeuronConfig(
            threshold=self.config.threshold,
            decay_rate=self.config.leak,
            refractory_period=2
        )
        
        neuron = Neuron(
            neuron_id=neuron_id,
            layer=0,  # Fabric = layer 0
            receptive_field=receptive_field,
            config=config,
            birth_frame=birth_frame
        )
        
        self.neurons[neuron_id] = neuron
        self.working_set.add(neuron_id)
        
        return neuron
    
    def remove_neuron(self, neuron_id: int) -> bool:
        """Retire un neurone du fabric.
        
        Args:
            neuron_id: ID du neurone à retirer
            
        Returns:
            True si retiré, False si non trouvé
        """
        if neuron_id not in self.neurons:
            return False
        
        del self.neurons[neuron_id]
        self.working_set.discard(neuron_id)
        
        # Nettoyer les synapses liées
        for synapse in list(self.network.get_efferents(neuron_id)):
            self.network.remove_synapse(neuron_id, synapse.target_id)
        for synapse in list(self.network.get_afferents(neuron_id)):
            self.network.remove_synapse(synapse.source_id, neuron_id)
        
        return True
    
    def process(self, activations: Dict[int, float]) -> Set[int]:
        """Traite un pas de temps et retourne les neurones qui ont spiké.
        
        Équivalent à processAndSee() dans spikingcortex.
        
        Args:
            activations: Activations des neurones {id: activation}
            
        Returns:
            Ensemble des IDs des neurones qui ont spiké
        """
        self._frame_count += 1
        self.network.step()
        
        # Décaler l'historique
        self._previously_spiked = self._newly_spiked
        self._newly_spiked = set()
        
        # Calculer entrées synaptiques pour chaque neurone
        for neuron_id in self.working_set:
            if neuron_id not in self.neurons:
                continue
            
            neuron = self.neurons[neuron_id]
            
            # Activation directe
            direct = activations.get(neuron_id, 0.0)
            
            # Entrée synaptique depuis neurones précédemment actifs
            synaptic = self.network.compute_input(neuron_id, self._previously_spiked)
            
            # Activation totale
            total = direct + synaptic
            
            # Faire avancer le neurone
            if neuron.step(total, self._frame_count):
                self._newly_spiked.add(neuron_id)
        
        # Apprentissage
        if self._newly_spiked or self._previously_spiked:
            if self._ready:
                self._learn_through()
            else:
                self._not_learning_count += 1
        
        return self._newly_spiked
    
    def _learn_through(self):
        """Applique les règles d'apprentissage du fabric.
        
        Équivalent à learnTrough() dans spikingcortex.
        """
        self._ready = False
        
        # 1. Créer un nouveau neurone si plusieurs ont spiké précédemment
        #    mais aucun maintenant (pattern non reconnu)
        if (self.config.can_learn_concepts and 
            len(self._newly_spiked) == 0 and 
            len(self._previously_spiked) > 1):
            
            # Créer un neurone qui répond à ce pattern
            # Son RF sera la combinaison des RFs des neurones précédents
            import numpy as np
            
            # Union des champs récepteurs
            combined_rf = None
            for nid in self._previously_spiked:
                if nid in self.neurons:
                    neuron = self.neurons[nid]
                    if combined_rf is None:
                        combined_rf = neuron.receptive_field.copy()
                    else:
                        combined_rf = combined_rf | neuron.receptive_field
            
            if combined_rf is not None:
                new_neuron = self.create_neuron(combined_rf, self._frame_count)
                
                if new_neuron is not None:
                    # Connecter les sources au nouveau neurone
                    sources = list(self._previously_spiked)
                    self.network.learn_excitation(sources, new_neuron.neuron_id)
        
        # 2. Développer excitation, concurrence et feedback
        for neuron_id in self._newly_spiked:
            # Excitation: renforcer les connexions depuis les sources précédentes
            if self._previously_spiked:
                sources = list(self._previously_spiked - {neuron_id})
                if sources:
                    self.network.learn_excitation(sources, neuron_id)
            
            # Inhibition latérale (concurrence)
            if self.config.can_develop_concurrency:
                concurrents = list(self._newly_spiked - {neuron_id})
                if concurrents:
                    self.network.learn_inhibition(neuron_id, concurrents)
            
            # Feedback vers les sources précédentes
            if self.config.can_develop_feedback:
                self.network.learn_feedback(neuron_id)
        
        self._ready = True
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du fabric."""
        return {
            'name': self.name,
            'neuron_count': len(self.neurons),
            'working_set_size': len(self.working_set),
            'synapse_count': self.network.synapse_count,
            'frame_count': self._frame_count,
            'previously_spiked': len(self._previously_spiked),
            'newly_spiked': len(self._newly_spiked),
            'not_learning_count': self._not_learning_count,
            **self.network.get_stats()
        }
    
    def reset(self):
        """Réinitialise le fabric."""
        self.neurons.clear()
        self.working_set.clear()
        self.network.reset()
        self._previously_spiked.clear()
        self._newly_spiked.clear()
        self._frame_count = 0
        self._next_neuron_id = 0


class Cortex:
    """Collection de Fabrics formant un cortex.
    
    Permet d'organiser plusieurs Fabrics avec des spécialisations
    différentes (visuel, auditif, moteur, etc.).
    """
    
    def __init__(self, name: str = "cortex"):
        """Initialise le cortex.
        
        Args:
            name: Nom du cortex
        """
        self.name = name
        self.fabrics: Dict[str, Fabric] = {}
        self._frame_count = 0
    
    def add_fabric(self, fabric: Fabric) -> bool:
        """Ajoute un fabric au cortex.
        
        Args:
            fabric: Fabric à ajouter
            
        Returns:
            True si ajouté, False si nom déjà utilisé
        """
        if fabric.name in self.fabrics:
            return False
        
        self.fabrics[fabric.name] = fabric
        return True
    
    def create_fabric(self, config: FabricConfig) -> Fabric:
        """Crée et ajoute un nouveau fabric.
        
        Args:
            config: Configuration du fabric
            
        Returns:
            Fabric créé
        """
        fabric = Fabric(config)
        self.fabrics[config.name] = fabric
        return fabric
    
    def get_fabric(self, name: str) -> Optional[Fabric]:
        """Récupère un fabric par son nom."""
        return self.fabrics.get(name)
    
    def process(self, fabric_activations: Dict[str, Dict[int, float]]) -> Dict[str, Set[int]]:
        """Traite tous les fabrics.
        
        Args:
            fabric_activations: Activations par fabric {fabric_name: {neuron_id: activation}}
            
        Returns:
            Neurones qui ont spiké par fabric
        """
        self._frame_count += 1
        results = {}
        
        for name, fabric in self.fabrics.items():
            activations = fabric_activations.get(name, {})
            results[name] = fabric.process(activations)
        
        return results
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du cortex."""
        total_neurons = sum(f.neuron_count for f in self.fabrics.values())
        total_synapses = sum(f.network.synapse_count for f in self.fabrics.values())
        
        return {
            'name': self.name,
            'fabric_count': len(self.fabrics),
            'total_neurons': total_neurons,
            'total_synapses': total_synapses,
            'frame_count': self._frame_count,
            'fabrics': {name: f.get_stats() for name, f in self.fabrics.items()}
        }
