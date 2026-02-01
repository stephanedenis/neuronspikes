"""Tests pour le module de genèse des neurones."""

import numpy as np
import pytest
from neuronspikes.genesis import (
    Neuron,
    NeuronConfig,
    NeuronState,
    NeuronLayer,
    GenesisConfig,
    visualize_neurons,
)
from neuronspikes.temporal import TemporalPattern


class TestNeuronConfig:
    """Tests pour NeuronConfig."""
    
    def test_default_values(self):
        """Test des valeurs par défaut."""
        config = NeuronConfig()
        
        assert config.threshold == 0.8
        assert config.decay_rate == 0.1
        assert config.refractory_period == 3
        assert config.learning_rate == 0.01
    
    def test_custom_values(self):
        """Test des valeurs personnalisées."""
        config = NeuronConfig(threshold=0.5, decay_rate=0.2)
        
        assert config.threshold == 0.5
        assert config.decay_rate == 0.2


class TestNeuron:
    """Tests pour Neuron."""
    
    def test_creation(self):
        """Test de création basique."""
        rf = np.zeros((10, 10), dtype=bool)
        rf[3:6, 3:6] = True  # 3x3 au centre
        
        neuron = Neuron(
            neuron_id=0,
            layer=1,
            receptive_field=rf,
        )
        
        assert neuron.neuron_id == 0
        assert neuron.layer == 1
        assert neuron._rf_size == 9
        assert neuron.state == NeuronState.DORMANT
        assert neuron.potential == 0.0
    
    def test_centroid(self):
        """Test du calcul du centroïde."""
        rf = np.zeros((10, 10), dtype=bool)
        rf[4:6, 4:6] = True  # 2x2 au centre
        
        neuron = Neuron(neuron_id=0, layer=1, receptive_field=rf)
        
        # Centroïde au milieu du bloc 2x2
        assert neuron.centroid[0] == pytest.approx(4.5, abs=0.1)
        assert neuron.centroid[1] == pytest.approx(4.5, abs=0.1)
    
    def test_compute_activation_full_match(self):
        """Test de l'activation avec match complet."""
        rf = np.zeros((10, 10), dtype=bool)
        rf[3:6, 3:6] = True
        
        neuron = Neuron(neuron_id=0, layer=1, receptive_field=rf)
        
        # Entrée identique au RF
        input_pattern = rf.copy()
        activation = neuron.compute_activation(input_pattern)
        
        assert activation == 1.0
    
    def test_compute_activation_no_match(self):
        """Test de l'activation sans match."""
        rf = np.zeros((10, 10), dtype=bool)
        rf[0:3, 0:3] = True  # Haut gauche
        
        neuron = Neuron(neuron_id=0, layer=1, receptive_field=rf)
        
        # Entrée dans l'autre coin
        input_pattern = np.zeros((10, 10), dtype=bool)
        input_pattern[7:10, 7:10] = True  # Bas droite
        
        activation = neuron.compute_activation(input_pattern)
        
        assert activation == 0.0
    
    def test_compute_activation_partial_match(self):
        """Test de l'activation avec match partiel."""
        rf = np.zeros((10, 10), dtype=bool)
        rf[2:5, 2:5] = True  # 3x3 = 9 pixels
        
        neuron = Neuron(neuron_id=0, layer=1, receptive_field=rf)
        
        # Entrée qui chevauche partiellement
        input_pattern = np.zeros((10, 10), dtype=bool)
        input_pattern[3:6, 3:6] = True  # Décalé d'1 pixel
        
        activation = neuron.compute_activation(input_pattern)
        
        # Intersection: 2x2 = 4 pixels sur 9
        assert activation == pytest.approx(4/9, abs=0.01)
    
    def test_step_accumulation(self):
        """Test de l'accumulation du potentiel."""
        rf = np.ones((5, 5), dtype=bool)
        config = NeuronConfig(threshold=2.0, decay_rate=0.0)
        
        neuron = Neuron(neuron_id=0, layer=1, receptive_field=rf, config=config)
        
        # Accumuler sans décroissance
        neuron.step(0.5, current_frame=1)
        assert neuron.potential == pytest.approx(0.5, abs=0.01)
        assert neuron.state == NeuronState.CHARGING
        
        neuron.step(0.5, current_frame=2)
        assert neuron.potential == pytest.approx(1.0, abs=0.01)
    
    def test_step_spike(self):
        """Test du déclenchement d'un spike."""
        rf = np.ones((5, 5), dtype=bool)
        config = NeuronConfig(threshold=0.5, decay_rate=0.0, refractory_period=2)
        
        neuron = Neuron(neuron_id=0, layer=1, receptive_field=rf, config=config)
        
        # Activation suffisante pour spike
        spiked = neuron.step(0.6, current_frame=1)
        
        assert spiked is True
        # L'état FIRING est visible pendant 1 frame
        assert neuron.state == NeuronState.FIRING
        assert neuron.total_spikes == 1
        assert neuron.potential == 0.0
        
        # Au step suivant, il passe en REFRACTORY
        spiked = neuron.step(0.0, current_frame=2)
        assert spiked is False
        assert neuron.state == NeuronState.REFRACTORY
    
    def test_refractory_period(self):
        """Test de la période réfractaire."""
        rf = np.ones((5, 5), dtype=bool)
        config = NeuronConfig(threshold=0.5, decay_rate=0.0, refractory_period=3)
        
        neuron = Neuron(neuron_id=0, layer=1, receptive_field=rf, config=config)
        
        # Provoquer un spike - état FIRING visible
        neuron.step(0.6, current_frame=1)
        assert neuron.state == NeuronState.FIRING
        
        # Frame 2: transition FIRING -> REFRACTORY (compteur=3)
        spiked = neuron.step(1.0, current_frame=2)
        assert spiked is False
        assert neuron.state == NeuronState.REFRACTORY
        
        # Frame 3: REFRACTORY (compteur=2)
        spiked = neuron.step(1.0, current_frame=3)
        assert spiked is False
        assert neuron.state == NeuronState.REFRACTORY
        
        # Frame 4: REFRACTORY (compteur=1)
        spiked = neuron.step(1.0, current_frame=4)
        assert spiked is False
        assert neuron.state == NeuronState.REFRACTORY
        
        # Frame 5: REFRACTORY -> DORMANT (compteur=0)
        spiked = neuron.step(1.0, current_frame=5)
        assert spiked is False
        assert neuron.state == NeuronState.DORMANT
        
        # Frame 6: période finie, devrait pouvoir spiker
        spiked = neuron.step(0.6, current_frame=6)
        assert spiked is True
    
    def test_reset(self):
        """Test de la réinitialisation."""
        rf = np.ones((5, 5), dtype=bool)
        neuron = Neuron(neuron_id=0, layer=1, receptive_field=rf)
        
        neuron.potential = 0.5
        neuron.state = NeuronState.CHARGING
        
        neuron.reset()
        
        assert neuron.potential == 0.0
        assert neuron.state == NeuronState.DORMANT


class TestGenesisConfig:
    """Tests pour GenesisConfig."""
    
    def test_default_values(self):
        """Test des valeurs par défaut."""
        config = GenesisConfig()
        
        assert config.min_pattern_confidence == 0.6
        assert config.min_pattern_occurrences == 10
        assert config.max_neurons_per_layer == 1000
    
    def test_custom_values(self):
        """Test des valeurs personnalisées."""
        config = GenesisConfig(min_pattern_confidence=0.8, max_neurons_per_layer=100)
        
        assert config.min_pattern_confidence == 0.8
        assert config.max_neurons_per_layer == 100


class TestNeuronLayer:
    """Tests pour NeuronLayer."""
    
    def test_creation(self):
        """Test de création."""
        layer = NeuronLayer(layer_id=1, shape=(64, 64))
        
        assert layer.layer_id == 1
        assert layer.shape == (64, 64)
        assert layer.neuron_count == 0
    
    def test_create_neuron_from_pattern(self):
        """Test de création de neurone à partir d'un pattern."""
        config = GenesisConfig(
            min_pattern_confidence=0.5,
            min_pattern_occurrences=5
        )
        layer = NeuronLayer(layer_id=1, shape=(10, 10), config=config)
        
        # Créer un pattern valide
        signature = np.zeros((10, 10), dtype=bool)
        signature[3:6, 3:6] = True
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=10,
            occurrences=10,
            confidence=0.7
        )
        
        neuron = layer.create_neuron_from_pattern(pattern)
        
        assert neuron is not None
        assert layer.neuron_count == 1
        assert np.array_equal(neuron.receptive_field, signature)
    
    def test_create_neuron_rejected_low_confidence(self):
        """Test de rejet pour confiance trop basse."""
        config = GenesisConfig(min_pattern_confidence=0.8)
        layer = NeuronLayer(layer_id=1, shape=(10, 10), config=config)
        
        signature = np.ones((10, 10), dtype=bool)
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=10,
            occurrences=20,
            confidence=0.5  # Trop bas
        )
        
        neuron = layer.create_neuron_from_pattern(pattern)
        
        assert neuron is None
        assert layer.neuron_count == 0
    
    def test_create_neuron_rejected_duplicate_pattern(self):
        """Test de rejet pour pattern déjà associé."""
        config = GenesisConfig(
            min_pattern_confidence=0.5,
            min_pattern_occurrences=5
        )
        layer = NeuronLayer(layer_id=1, shape=(10, 10), config=config)
        
        signature = np.ones((10, 10), dtype=bool)
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=10,
            occurrences=10,
            confidence=0.7
        )
        
        # Premier neurone créé
        layer.create_neuron_from_pattern(pattern)
        
        # Même pattern -> refusé
        neuron = layer.create_neuron_from_pattern(pattern)
        
        assert neuron is None
        assert layer.neuron_count == 1
    
    def test_process_no_spike(self):
        """Test du traitement sans spike."""
        config = GenesisConfig(
            min_pattern_confidence=0.5,
            min_pattern_occurrences=5
        )
        neuron_config = NeuronConfig(threshold=0.9)
        layer = NeuronLayer(
            layer_id=1, 
            shape=(10, 10), 
            config=config,
            neuron_config=neuron_config
        )
        
        # Créer un neurone
        signature = np.zeros((10, 10), dtype=bool)
        signature[3:6, 3:6] = True
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=10,
            occurrences=10,
            confidence=0.7
        )
        layer.create_neuron_from_pattern(pattern)
        
        # Input qui ne matche pas
        input_pattern = np.zeros((10, 10), dtype=bool)
        input_pattern[0:2, 0:2] = True
        
        output = layer.process(input_pattern)
        
        assert not np.any(output)  # Pas de spike
    
    def test_process_with_spike(self):
        """Test du traitement avec spike."""
        config = GenesisConfig(
            min_pattern_confidence=0.5,
            min_pattern_occurrences=5
        )
        neuron_config = NeuronConfig(threshold=0.5, decay_rate=0.0)
        layer = NeuronLayer(
            layer_id=1, 
            shape=(10, 10), 
            config=config,
            neuron_config=neuron_config
        )
        
        # Créer un neurone
        signature = np.zeros((10, 10), dtype=bool)
        signature[3:6, 3:6] = True
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=10,
            occurrences=10,
            confidence=0.7
        )
        layer.create_neuron_from_pattern(pattern)
        
        # Input qui matche le RF
        output = layer.process(signature)
        
        assert np.any(output)  # Spike!
        assert layer._total_spikes == 1
    
    def test_get_neuron_map(self):
        """Test de la génération de la carte des neurones."""
        config = GenesisConfig(
            min_pattern_confidence=0.5,
            min_pattern_occurrences=5
        )
        layer = NeuronLayer(layer_id=1, shape=(10, 10), config=config)
        
        signature = np.zeros((10, 10), dtype=bool)
        signature[3:6, 3:6] = True
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=10,
            occurrences=10,
            confidence=0.7
        )
        layer.create_neuron_from_pattern(pattern)
        
        neuron_map = layer.get_neuron_map()
        
        assert neuron_map.shape == (10, 10)
        assert neuron_map[4, 4] > 0  # Neurone présent
        assert neuron_map[0, 0] == 0  # Pas de neurone
    
    def test_get_stats(self):
        """Test des statistiques."""
        layer = NeuronLayer(layer_id=1, shape=(10, 10))
        
        stats = layer.get_stats()
        
        assert 'layer_id' in stats
        assert 'neuron_count' in stats
        assert 'total_created' in stats
        assert stats['layer_id'] == 1
    
    def test_clear(self):
        """Test de la suppression de tous les neurones."""
        config = GenesisConfig(
            min_pattern_confidence=0.5,
            min_pattern_occurrences=5
        )
        layer = NeuronLayer(layer_id=1, shape=(10, 10), config=config)
        
        # Créer un neurone
        signature = np.ones((10, 10), dtype=bool)
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=10,
            occurrences=10,
            confidence=0.7
        )
        layer.create_neuron_from_pattern(pattern)
        
        assert layer.neuron_count == 1
        
        layer.clear()
        
        assert layer.neuron_count == 0


class TestVisualizeNeurons:
    """Tests pour la visualisation des neurones."""
    
    def test_empty_layer(self):
        """Test avec une couche vide."""
        layer = NeuronLayer(layer_id=1, shape=(10, 10))
        
        output = visualize_neurons(layer)
        
        assert output.shape == (10, 10, 3)
        assert np.all(output == 0)
    
    def test_with_neurons(self):
        """Test avec des neurones."""
        config = GenesisConfig(
            min_pattern_confidence=0.5,
            min_pattern_occurrences=5
        )
        layer = NeuronLayer(layer_id=1, shape=(10, 10), config=config)
        
        signature = np.zeros((10, 10), dtype=bool)
        signature[3:6, 3:6] = True
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=10,
            occurrences=10,
            confidence=0.7
        )
        layer.create_neuron_from_pattern(pattern)
        
        output = visualize_neurons(layer)
        
        assert output.shape == (10, 10, 3)
        # Zone du neurone devrait avoir une couleur
        assert np.any(output[4, 4] > 0)
        # Zone hors neurone devrait être noire
        assert np.all(output[0, 0] == 0)


class TestIntegration:
    """Tests d'intégration."""
    
    def test_full_pipeline(self):
        """Test du pipeline complet pattern -> neurone -> spike."""
        from neuronspikes.groups import ActivationGroup, GroupDetector
        from neuronspikes.temporal import TemporalCorrelator, CorrelationConfig
        
        # Configuration pour des tests rapides
        corr_config = CorrelationConfig(
            min_overlap=0.5,
            min_occurrences=3,
            confidence_threshold=0.3
        )
        genesis_config = GenesisConfig(
            min_pattern_confidence=0.3,
            min_pattern_occurrences=3
        )
        neuron_config = NeuronConfig(threshold=0.5, decay_rate=0.0)
        
        # Créer les composants
        detector = GroupDetector()
        correlator = TemporalCorrelator((20, 20), corr_config)
        layer = NeuronLayer(1, (20, 20), genesis_config, neuron_config)
        
        # Simuler une activation répétée
        activation = np.zeros((20, 20), dtype=bool)
        activation[5:10, 5:10] = True
        
        # Plusieurs frames pour créer un pattern stable
        for i in range(10):
            groups = detector.detect_groups(activation, slot=0, frame=i)
            correlator.process_groups(groups)
        
        # Créer des neurones à partir des patterns stables
        for pattern in correlator.stable_patterns:
            layer.create_neuron_from_pattern(pattern)
        
        # Devrait avoir au moins un neurone
        assert layer.neuron_count >= 1
        
        # Traiter avec l'entrée qui correspond au pattern
        output = layer.process(activation)
        
        # Le neurone devrait spiker
        assert np.any(output)
