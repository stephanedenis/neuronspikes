"""Tests pour le module de synapses et apprentissage Hebbien."""

import numpy as np
import pytest
from neuronspikes.synapses import (
    Synapse,
    SynapseType,
    SynapticConfig,
    SynapticNetwork,
    HebbianLayer,
)
from neuronspikes.genesis import Neuron, NeuronConfig


class TestSynapse:
    """Tests pour Synapse."""
    
    def test_creation(self):
        """Test de création basique."""
        synapse = Synapse(
            source_id=0,
            target_id=1,
            weight=0.5,
            synapse_type=SynapseType.EXCITATORY
        )
        
        assert synapse.source_id == 0
        assert synapse.target_id == 1
        assert synapse.weight == 0.5
        assert synapse.is_excitatory
        assert not synapse.is_inhibitory
    
    def test_inhibitory(self):
        """Test synapse inhibitrice."""
        synapse = Synapse(
            source_id=0,
            target_id=1,
            weight=-0.3,
            synapse_type=SynapseType.INHIBITORY
        )
        
        assert synapse.is_inhibitory
        assert not synapse.is_excitatory


class TestSynapticConfig:
    """Tests pour SynapticConfig."""
    
    def test_default_values(self):
        """Test des valeurs par défaut."""
        config = SynapticConfig()
        
        assert config.learning_rate == 0.1
        assert config.excitation_strength == 0.3
        assert config.inhibition_strength < 0
    
    def test_custom_values(self):
        """Test des valeurs personnalisées."""
        config = SynapticConfig(learning_rate=0.2, max_weight=5.0)
        
        assert config.learning_rate == 0.2
        assert config.max_weight == 5.0


class TestSynapticNetwork:
    """Tests pour SynapticNetwork."""
    
    def test_creation(self):
        """Test de création."""
        network = SynapticNetwork()
        
        assert network.synapse_count == 0
    
    def test_create_synapse(self):
        """Test de création de synapse."""
        network = SynapticNetwork()
        
        synapse = network.create_synapse(0, 1, 0.5)
        
        assert network.synapse_count == 1
        assert synapse.source_id == 0
        assert synapse.target_id == 1
        assert synapse.weight == 0.5
    
    def test_get_synapse(self):
        """Test de récupération de synapse."""
        network = SynapticNetwork()
        network.create_synapse(0, 1, 0.5)
        
        synapse = network.get_synapse(0, 1)
        
        assert synapse is not None
        assert synapse.weight == 0.5
        
        # Synapse inexistante
        assert network.get_synapse(1, 0) is None
    
    def test_has_synapse(self):
        """Test de vérification d'existence."""
        network = SynapticNetwork()
        network.create_synapse(0, 1, 0.5)
        
        assert network.has_synapse(0, 1)
        assert not network.has_synapse(1, 0)
    
    def test_remove_synapse(self):
        """Test de suppression de synapse."""
        network = SynapticNetwork()
        network.create_synapse(0, 1, 0.5)
        
        assert network.synapse_count == 1
        
        network.remove_synapse(0, 1)
        
        assert network.synapse_count == 0
        assert not network.has_synapse(0, 1)
    
    def test_get_efferents(self):
        """Test des synapses sortantes."""
        network = SynapticNetwork()
        network.create_synapse(0, 1, 0.5)
        network.create_synapse(0, 2, 0.3)
        network.create_synapse(1, 2, 0.4)
        
        efferents = network.get_efferents(0)
        
        assert len(efferents) == 2
        targets = {s.target_id for s in efferents}
        assert targets == {1, 2}
    
    def test_get_afferents(self):
        """Test des synapses entrantes."""
        network = SynapticNetwork()
        network.create_synapse(0, 2, 0.5)
        network.create_synapse(1, 2, 0.3)
        network.create_synapse(0, 1, 0.4)
        
        afferents = network.get_afferents(2)
        
        assert len(afferents) == 2
        sources = {s.source_id for s in afferents}
        assert sources == {0, 1}
    
    def test_compute_input(self):
        """Test du calcul d'entrée synaptique."""
        network = SynapticNetwork()
        network.create_synapse(0, 2, 0.5)
        network.create_synapse(1, 2, 0.3)
        
        # Seul neurone 0 actif
        input_value = network.compute_input(2, {0})
        assert input_value == pytest.approx(0.5, abs=0.01)
        
        # Les deux actifs
        input_value = network.compute_input(2, {0, 1})
        assert input_value == pytest.approx(0.8, abs=0.01)
        
        # Aucun actif
        input_value = network.compute_input(2, set())
        assert input_value == 0.0
    
    def test_learn_excitation(self):
        """Test de l'apprentissage excitateur."""
        config = SynapticConfig(excitation_strength=1.0)
        network = SynapticNetwork(config)
        
        # Apprentissage: sources 0,1 → target 2
        network.learn_excitation([0, 1], 2)
        
        # Deux synapses créées
        assert network.has_synapse(0, 2)
        assert network.has_synapse(1, 2)
        
        # Poids divisé entre sources
        s0 = network.get_synapse(0, 2)
        s1 = network.get_synapse(1, 2)
        assert s0.weight == pytest.approx(0.5, abs=0.01)
        assert s1.weight == pytest.approx(0.5, abs=0.01)
    
    def test_learn_inhibition(self):
        """Test de l'apprentissage inhibiteur."""
        config = SynapticConfig(inhibition_strength=-0.5)
        network = SynapticNetwork(config)
        
        # Neurone 0 inhibe 1 et 2
        network.learn_inhibition(0, [1, 2])
        
        assert network.has_synapse(0, 1)
        assert network.has_synapse(0, 2)
        
        s1 = network.get_synapse(0, 1)
        assert s1.weight < 0
    
    def test_learn_feedback(self):
        """Test de l'apprentissage feedback."""
        config = SynapticConfig(feedback_ratio=-0.5)
        network = SynapticNetwork(config)
        
        # Créer excitation 0 → 1
        network.create_synapse(0, 1, 0.8, SynapseType.EXCITATORY)
        
        # Neurone 1 spike → feedback vers 0
        network.learn_feedback(1)
        
        # Synapse inverse créée
        assert network.has_synapse(1, 0)
        feedback = network.get_synapse(1, 0)
        assert feedback.weight < 0  # Inhibition
    
    def test_decay_weights(self):
        """Test de la décroissance des poids."""
        config = SynapticConfig(weight_decay=0.5, prune_threshold=0.1)
        network = SynapticNetwork(config)
        
        network.create_synapse(0, 1, 0.5)
        network.step()  # Avancer le temps
        
        network.decay_weights()
        
        synapse = network.get_synapse(0, 1)
        assert synapse.weight == pytest.approx(0.25, abs=0.01)
        
        # Plusieurs décroissances → suppression
        for _ in range(10):
            network.step()
            network.decay_weights()
        
        assert not network.has_synapse(0, 1)
    
    def test_get_stats(self):
        """Test des statistiques."""
        network = SynapticNetwork()
        network.create_synapse(0, 1, 0.5)
        network.create_synapse(1, 2, -0.3)
        
        stats = network.get_stats()
        
        assert stats['synapse_count'] == 2
        assert stats['excitatory'] == 1
        assert stats['inhibitory'] == 1
    
    def test_reset(self):
        """Test de la réinitialisation."""
        network = SynapticNetwork()
        network.create_synapse(0, 1, 0.5)
        network.create_synapse(1, 2, 0.3)
        
        network.reset()
        
        assert network.synapse_count == 0


class TestHebbianLayer:
    """Tests pour HebbianLayer."""
    
    def make_neurons(self, count: int) -> dict:
        """Crée un dict de neurones pour les tests."""
        neurons = {}
        rf = np.ones((5, 5), dtype=bool)
        config = NeuronConfig(threshold=0.5, decay_rate=0.0)
        
        for i in range(count):
            neurons[i] = Neuron(
                neuron_id=i,
                layer=1,
                receptive_field=rf,
                config=config
            )
        return neurons
    
    def test_creation(self):
        """Test de création."""
        neurons = self.make_neurons(5)
        layer = HebbianLayer(neurons)
        
        assert len(layer.neurons) == 5
        assert layer.network.synapse_count == 0
    
    def test_process_spike(self):
        """Test du traitement avec spike."""
        neurons = self.make_neurons(3)
        layer = HebbianLayer(neurons)
        
        # Activation forte sur neurone 0
        activations = {0: 0.8, 1: 0.1, 2: 0.1}
        spiked = layer.process(activations, learn=False)
        
        assert 0 in spiked
        assert 1 not in spiked
    
    def test_process_learns_excitation(self):
        """Test que le processing apprend les excitations."""
        neurons = self.make_neurons(3)
        layer = HebbianLayer(neurons, enable_feedback=False, enable_lateral_inhibition=False)
        
        # Frame 1: neurone 0 spike
        layer.process({0: 0.8, 1: 0.0, 2: 0.0}, learn=True)
        
        # Frame 2: neurone 1 spike (après 0)
        layer.process({0: 0.0, 1: 0.8, 2: 0.0}, learn=True)
        
        # Devrait avoir créé synapse 0 → 1
        assert layer.network.has_synapse(0, 1)
    
    def test_process_learns_lateral_inhibition(self):
        """Test de l'inhibition latérale."""
        neurons = self.make_neurons(3)
        layer = HebbianLayer(neurons, enable_feedback=False, enable_lateral_inhibition=True)
        
        # Deux neurones spikent en même temps
        layer.process({0: 0.8, 1: 0.8, 2: 0.0}, learn=True)
        
        # Devrait avoir créé inhibition mutuelle
        assert layer.network.has_synapse(0, 1) or layer.network.has_synapse(1, 0)
    
    def test_get_stats(self):
        """Test des statistiques."""
        neurons = self.make_neurons(3)
        layer = HebbianLayer(neurons)
        
        layer.process({0: 0.8}, learn=True)
        
        stats = layer.get_stats()
        
        assert 'neuron_count' in stats
        assert 'synapse_count' in stats
        assert stats['neuron_count'] == 3


class TestIntegration:
    """Tests d'intégration."""
    
    def test_hebbian_learning_chain(self):
        """Test d'une chaîne d'apprentissage Hebbien."""
        # Créer des neurones
        neurons = {}
        rf = np.ones((5, 5), dtype=bool)
        config = NeuronConfig(threshold=0.4, decay_rate=0.1)
        
        for i in range(5):
            neurons[i] = Neuron(
                neuron_id=i,
                layer=1,
                receptive_field=rf,
                config=config
            )
        
        layer = HebbianLayer(
            neurons,
            SynapticConfig(excitation_strength=0.5),
            enable_feedback=True,
            enable_lateral_inhibition=True
        )
        
        # Simuler une séquence répétée: 0 → 1 → 2
        for _ in range(10):
            layer.process({0: 0.8}, learn=True)
            layer.process({1: 0.8}, learn=True)
            layer.process({2: 0.8}, learn=True)
            layer.process({}, learn=True)  # Pause
        
        # Devrait avoir des connexions 0→1 et 1→2
        stats = layer.get_stats()
        assert stats['synapse_count'] > 0
        
        # La chaîne devrait être apprise
        assert layer.network.has_synapse(0, 1) or layer.network.has_synapse(1, 2)
