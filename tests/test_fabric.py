"""Tests pour le module Fabric."""

import numpy as np
import pytest
from neuronspikes.fabric import (
    Fabric,
    FabricConfig,
    LearningCapability,
    Cortex,
)
from neuronspikes.genesis import Neuron, NeuronConfig


class TestFabricConfig:
    """Tests pour FabricConfig."""
    
    def test_default_values(self):
        """Test des valeurs par défaut."""
        config = FabricConfig()
        
        assert config.name == "default"
        assert config.max_neurons == 1000
        assert config.can_learn_concepts
        assert config.can_develop_concurrency
        assert config.can_develop_feedback
    
    def test_learning_capabilities_none(self):
        """Test sans capacités d'apprentissage."""
        config = FabricConfig(learning_capabilities=LearningCapability.NONE.value)
        
        assert not config.can_learn_concepts
        assert not config.can_develop_concurrency
        assert not config.can_develop_feedback
    
    def test_learning_capabilities_partial(self):
        """Test avec capacités partielles."""
        config = FabricConfig(
            learning_capabilities=LearningCapability.LEARN_CONCEPTS.value
        )
        
        assert config.can_learn_concepts
        assert not config.can_develop_concurrency
        assert not config.can_develop_feedback
    
    def test_custom_name(self):
        """Test avec nom personnalisé."""
        config = FabricConfig(name="visual_cortex")
        assert config.name == "visual_cortex"


class TestFabric:
    """Tests pour Fabric."""
    
    def make_rf(self, size: int = 5) -> np.ndarray:
        """Crée un champ récepteur de test."""
        return np.ones((size, size), dtype=bool)
    
    def test_creation(self):
        """Test de création basique."""
        fabric = Fabric()
        
        assert fabric.neuron_count == 0
        assert fabric.name == "default"
        assert fabric.can_add_neuron
    
    def test_creation_with_config(self):
        """Test de création avec config."""
        config = FabricConfig(name="test", max_neurons=10)
        fabric = Fabric(config)
        
        assert fabric.name == "test"
        assert fabric.can_add_neuron
    
    def test_create_neuron(self):
        """Test de création de neurone."""
        fabric = Fabric()
        rf = self.make_rf()
        
        neuron = fabric.create_neuron(rf, birth_frame=0)
        
        assert neuron is not None
        assert fabric.neuron_count == 1
        assert neuron.neuron_id == 0
        assert 0 in fabric.working_set
    
    def test_create_neuron_limit(self):
        """Test de limite de neurones."""
        config = FabricConfig(max_neurons=3)
        fabric = Fabric(config)
        rf = self.make_rf()
        
        for i in range(3):
            assert fabric.create_neuron(rf) is not None
        
        assert fabric.neuron_count == 3
        assert fabric.create_neuron(rf) is None  # Limite atteinte
    
    def test_add_neuron(self):
        """Test d'ajout de neurone externe."""
        fabric = Fabric()
        rf = self.make_rf()
        
        neuron = Neuron(
            neuron_id=42,
            layer=1,
            receptive_field=rf,
            config=NeuronConfig()
        )
        
        assert fabric.add_neuron(neuron)
        assert fabric.neuron_count == 1
        assert 42 in fabric.neurons
        assert 42 in fabric.working_set
    
    def test_remove_neuron(self):
        """Test de suppression de neurone."""
        fabric = Fabric()
        rf = self.make_rf()
        
        neuron = fabric.create_neuron(rf)
        nid = neuron.neuron_id
        
        assert fabric.neuron_count == 1
        assert fabric.remove_neuron(nid)
        assert fabric.neuron_count == 0
        assert nid not in fabric.working_set
    
    def test_process_no_spike(self):
        """Test de traitement sans spike."""
        fabric = Fabric()
        rf = self.make_rf()
        fabric.create_neuron(rf)
        
        # Activation faible
        spiked = fabric.process({0: 0.1})
        
        assert len(spiked) == 0
    
    def test_process_with_spike(self):
        """Test de traitement avec spike."""
        config = FabricConfig(threshold=0.5)
        fabric = Fabric(config)
        rf = self.make_rf()
        fabric.create_neuron(rf)
        
        # Activation forte
        spiked = fabric.process({0: 2.0})
        
        assert 0 in spiked
    
    def test_process_learns_excitation(self):
        """Test que le traitement apprend les excitations."""
        config = FabricConfig(
            threshold=0.5,
            learning_capabilities=LearningCapability.ALL.value
        )
        fabric = Fabric(config)
        rf = self.make_rf()
        
        # Créer deux neurones
        n0 = fabric.create_neuron(rf)
        n1 = fabric.create_neuron(rf)
        
        # Frame 1: neurone 0 spike
        fabric.process({0: 2.0, 1: 0.0})
        
        # Frame 2: neurone 1 spike (après 0)
        fabric.process({0: 0.0, 1: 2.0})
        
        # Devrait avoir créé synapse 0 → 1
        assert fabric.network.has_synapse(0, 1)
    
    def test_process_develops_concurrency(self):
        """Test du développement de la concurrence."""
        config = FabricConfig(
            threshold=0.5,
            learning_capabilities=LearningCapability.DEVELOP_CONCURRENCY.value
        )
        fabric = Fabric(config)
        rf = self.make_rf()
        
        fabric.create_neuron(rf)
        fabric.create_neuron(rf)
        
        # Les deux spikent en même temps
        fabric.process({0: 2.0, 1: 2.0})
        
        # Devrait avoir créé inhibition mutuelle
        has_inhibition = (
            fabric.network.has_synapse(0, 1) or 
            fabric.network.has_synapse(1, 0)
        )
        assert has_inhibition
    
    def test_learn_new_concepts(self):
        """Test de la création de nouveaux concepts."""
        config = FabricConfig(
            threshold=0.5,
            learning_capabilities=LearningCapability.LEARN_CONCEPTS.value
        )
        fabric = Fabric(config)
        rf = self.make_rf()
        
        # Créer deux neurones de base
        fabric.create_neuron(rf)
        fabric.create_neuron(rf)
        
        # Frame 1: les deux spikent
        fabric.process({0: 2.0, 1: 2.0})
        
        # Frame 2: aucun ne spike (pattern non reconnu)
        fabric.process({0: 0.0, 1: 0.0})
        
        # Un nouveau neurone devrait avoir été créé
        assert fabric.neuron_count == 3
    
    def test_get_stats(self):
        """Test des statistiques."""
        fabric = Fabric()
        rf = self.make_rf()
        fabric.create_neuron(rf)
        fabric.process({0: 0.5})
        
        stats = fabric.get_stats()
        
        assert stats['name'] == "default"
        assert stats['neuron_count'] == 1
        assert 'synapse_count' in stats
    
    def test_reset(self):
        """Test de réinitialisation."""
        fabric = Fabric()
        rf = self.make_rf()
        fabric.create_neuron(rf)
        fabric.process({0: 2.0})
        
        fabric.reset()
        
        assert fabric.neuron_count == 0
        assert fabric.network.synapse_count == 0


class TestCortex:
    """Tests pour Cortex."""
    
    def test_creation(self):
        """Test de création."""
        cortex = Cortex("test_cortex")
        
        assert cortex.name == "test_cortex"
        assert len(cortex.fabrics) == 0
    
    def test_create_fabric(self):
        """Test de création de fabric."""
        cortex = Cortex()
        config = FabricConfig(name="visual")
        
        fabric = cortex.create_fabric(config)
        
        assert fabric.name == "visual"
        assert "visual" in cortex.fabrics
    
    def test_add_fabric(self):
        """Test d'ajout de fabric."""
        cortex = Cortex()
        fabric = Fabric(FabricConfig(name="motor"))
        
        assert cortex.add_fabric(fabric)
        assert "motor" in cortex.fabrics
    
    def test_add_fabric_duplicate(self):
        """Test d'ajout de fabric en double."""
        cortex = Cortex()
        fabric1 = Fabric(FabricConfig(name="same"))
        fabric2 = Fabric(FabricConfig(name="same"))
        
        assert cortex.add_fabric(fabric1)
        assert not cortex.add_fabric(fabric2)
    
    def test_get_fabric(self):
        """Test de récupération de fabric."""
        cortex = Cortex()
        config = FabricConfig(name="auditory")
        cortex.create_fabric(config)
        
        fabric = cortex.get_fabric("auditory")
        assert fabric is not None
        assert fabric.name == "auditory"
        
        assert cortex.get_fabric("inexistant") is None
    
    def test_process(self):
        """Test du traitement multi-fabric."""
        cortex = Cortex()
        
        # Créer deux fabrics
        v_config = FabricConfig(name="visual", threshold=0.5)
        m_config = FabricConfig(name="motor", threshold=0.5)
        
        v_fabric = cortex.create_fabric(v_config)
        m_fabric = cortex.create_fabric(m_config)
        
        # Ajouter des neurones
        rf = np.ones((5, 5), dtype=bool)
        v_fabric.create_neuron(rf)
        m_fabric.create_neuron(rf)
        
        # Traiter
        activations = {
            "visual": {0: 2.0},
            "motor": {0: 0.1}
        }
        
        results = cortex.process(activations)
        
        assert 0 in results["visual"]
        assert 0 not in results["motor"]
    
    def test_get_stats(self):
        """Test des statistiques du cortex."""
        cortex = Cortex("main")
        cortex.create_fabric(FabricConfig(name="f1"))
        cortex.create_fabric(FabricConfig(name="f2"))
        
        stats = cortex.get_stats()
        
        assert stats['name'] == "main"
        assert stats['fabric_count'] == 2
        assert 'f1' in stats['fabrics']
        assert 'f2' in stats['fabrics']


class TestIntegration:
    """Tests d'intégration."""
    
    def test_full_learning_cycle(self):
        """Test d'un cycle d'apprentissage complet."""
        config = FabricConfig(
            name="learning_test",
            threshold=0.5,
            learning_capabilities=LearningCapability.ALL.value
        )
        fabric = Fabric(config)
        
        # Créer des neurones de base
        rf = np.ones((5, 5), dtype=bool)
        for _ in range(3):
            fabric.create_neuron(rf)
        
        # Simuler une séquence répétée: 0 → 1 → 2
        for _ in range(10):
            fabric.process({0: 2.0})
            fabric.process({1: 2.0})
            fabric.process({2: 2.0})
            fabric.process({})  # Pause
        
        stats = fabric.get_stats()
        
        # Devrait avoir appris des connexions
        assert stats['synapse_count'] > 0
        
        # Les connexions excitatrices devraient exister
        assert (
            fabric.network.has_synapse(0, 1) or
            fabric.network.has_synapse(1, 2) or
            fabric.network.has_synapse(0, 2)
        )
