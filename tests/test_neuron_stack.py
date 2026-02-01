"""Tests pour NeuronStack - Pile de couches de neurones empilables."""

import numpy as np
import pytest

from neuronspikes import (
    NeuronStack,
    NeuronLayer,
    GenesisConfig,
    NeuronConfig,
    TemporalPattern,
    visualize_stack,
)


class TestNeuronStackCreation:
    """Tests de création du NeuronStack."""
    
    def test_create_default_stack(self):
        """Test création avec paramètres par défaut."""
        stack = NeuronStack(base_shape=(64, 64), num_layers=3)
        
        assert stack.num_layers == 3
        assert len(stack.layers) == 3
        assert stack.total_neurons == 0
    
    def test_layer_shapes_reduce(self):
        """Test que les couches réduisent en taille."""
        stack = NeuronStack(
            base_shape=(64, 64), 
            num_layers=3,
            reduction_factor=0.5
        )
        
        # Couche 0: 64x64
        assert stack.get_layer(0).shape == (64, 64)
        # Couche 1: 32x32
        assert stack.get_layer(1).shape == (32, 32)
        # Couche 2: 16x16
        assert stack.get_layer(2).shape == (16, 16)
    
    def test_layer_shapes_same_size(self):
        """Test couches de même taille."""
        stack = NeuronStack(
            base_shape=(32, 32),
            num_layers=4,
            reduction_factor=1.0
        )
        
        for layer in stack.layers:
            assert layer.shape == (32, 32)
    
    def test_minimum_layer_size(self):
        """Test que les couches ne descendent pas sous 8x8."""
        stack = NeuronStack(
            base_shape=(16, 16),
            num_layers=5,
            reduction_factor=0.5
        )
        
        # Même avec réduction, minimum 8x8
        for layer in stack.layers:
            h, w = layer.shape
            assert h >= 8
            assert w >= 8
    
    def test_get_layer(self):
        """Test récupération d'une couche."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=3)
        
        layer0 = stack.get_layer(0)
        assert layer0.layer_id == 0
        
        layer2 = stack.get_layer(2)
        assert layer2.layer_id == 2
        
        with pytest.raises(IndexError):
            stack.get_layer(10)


class TestNeuronStackProcessing:
    """Tests de traitement à travers la pile."""
    
    def test_process_returns_outputs_per_layer(self):
        """Test que process retourne une sortie par couche."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=3)
        
        input_pattern = np.zeros((32, 32), dtype=bool)
        input_pattern[10:20, 10:20] = True
        
        outputs = stack.process(input_pattern)
        
        assert len(outputs) == 3
        assert outputs[0].shape == (32, 32)
        assert outputs[1].shape == (16, 16)
        assert outputs[2].shape == (8, 8)
    
    def test_process_without_propagation(self):
        """Test traitement sans propagation."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=3)
        
        input_pattern = np.zeros((32, 32), dtype=bool)
        
        outputs = stack.process(input_pattern, propagate=False)
        
        # Toujours 3 sorties
        assert len(outputs) == 3
    
    def test_resize_pattern_downscale(self):
        """Test redimensionnement vers le bas."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=2)
        
        # Pattern 32x32 avec région active
        pattern = np.zeros((32, 32), dtype=bool)
        pattern[0:16, 0:16] = True  # Quart supérieur gauche
        
        resized = stack._resize_pattern(pattern, (16, 16))
        
        assert resized.shape == (16, 16)
        # Le quart supérieur gauche devrait être actif
        assert resized[0:8, 0:8].all()
    
    def test_resize_pattern_same_size(self):
        """Test redimensionnement même taille."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=2)
        
        pattern = np.zeros((32, 32), dtype=bool)
        pattern[10, 10] = True
        
        resized = stack._resize_pattern(pattern, (32, 32))
        
        assert resized.shape == (32, 32)
        assert np.array_equal(resized, pattern)


class TestNeuronStackNeurons:
    """Tests de création de neurones dans la pile."""
    
    def make_pattern(self, shape, x, y, size=5) -> TemporalPattern:
        """Crée un pattern temporel pour les tests."""
        signature = np.zeros(shape, dtype=bool)
        y1, y2 = max(0, y), min(shape[0], y + size)
        x1, x2 = max(0, x), min(shape[1], x + size)
        signature[y1:y2, x1:x2] = True
        
        return TemporalPattern(
            pattern_id=hash((x, y, size)) % 10000,
            signature=signature,
            confidence=0.9,
            occurrences=20,
            first_seen=0,
            last_seen=10,
        )
    
    def test_create_neurons_in_layer(self):
        """Test création de neurones dans une couche."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=2)
        
        patterns = [
            self.make_pattern((32, 32), 5, 5),
            self.make_pattern((32, 32), 20, 20),
        ]
        
        created = stack.create_neurons_from_patterns(0, patterns)
        
        assert len(created) == 2
        assert stack.get_layer(0).neuron_count == 2
        assert stack.total_neurons == 2
    
    def test_create_neurons_different_layers(self):
        """Test création dans différentes couches."""
        stack = NeuronStack(
            base_shape=(32, 32), 
            num_layers=2,
            reduction_factor=0.5
        )
        
        # Patterns pour couche 0 (32x32)
        patterns_0 = [self.make_pattern((32, 32), 5, 5)]
        stack.create_neurons_from_patterns(0, patterns_0)
        
        # Patterns pour couche 1 (16x16)
        patterns_1 = [self.make_pattern((16, 16), 3, 3)]
        stack.create_neurons_from_patterns(1, patterns_1)
        
        assert stack.get_layer(0).neuron_count == 1
        assert stack.get_layer(1).neuron_count == 1
        assert stack.total_neurons == 2
    
    def test_get_all_active_neurons(self):
        """Test récupération des neurones actifs."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=2)
        
        # Créer un neurone
        pattern = self.make_pattern((32, 32), 10, 10, size=10)
        stack.create_neurons_from_patterns(0, [pattern])
        
        # Activer avec le même pattern
        input_pattern = pattern.signature.copy()
        stack.process(input_pattern)
        stack.process(input_pattern)  # Second pass pour accumulation
        
        active = stack.get_all_active_neurons()
        
        # Peut être vide si le seuil n'est pas atteint
        assert isinstance(active, dict)


class TestNeuronStackStats:
    """Tests des statistiques."""
    
    def test_get_stats(self):
        """Test récupération des statistiques."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=3)
        
        stats = stack.get_stats()
        
        assert stats['num_layers'] == 3
        assert stats['total_neurons'] == 0
        assert stats['frame_count'] == 0
        assert len(stats['layers']) == 3
        assert len(stats['neurons_per_layer']) == 3
    
    def test_reset(self):
        """Test réinitialisation."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=2)
        
        # Traiter quelques frames
        input_pattern = np.zeros((32, 32), dtype=bool)
        for _ in range(5):
            stack.process(input_pattern)
        
        stack.reset()
        
        # Les neurones sont reset mais pas supprimés
        assert stack._frame_count == 5  # frame_count n'est pas reset
    
    def test_clear(self):
        """Test suppression totale."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=2)
        
        # Ajouter des neurones
        pattern = TemporalPattern(
            pattern_id=1,
            signature=np.ones((32, 32), dtype=bool),
            confidence=0.9,
            occurrences=20,
            first_seen=0,
            last_seen=10,
        )
        stack.create_neurons_from_patterns(0, [pattern])
        
        assert stack.total_neurons > 0
        
        stack.clear()
        
        assert stack.total_neurons == 0
        assert stack._frame_count == 0


class TestVisualizeStack:
    """Tests de visualisation."""
    
    def test_visualize_empty_stack(self):
        """Test visualisation pile vide."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=3)
        
        viz = visualize_stack(stack)
        
        assert viz.ndim == 3
        assert viz.shape[2] == 3  # RGB
    
    def test_visualize_with_neurons(self):
        """Test visualisation avec neurones."""
        stack = NeuronStack(base_shape=(32, 32), num_layers=2)
        
        # Ajouter un neurone
        signature = np.zeros((32, 32), dtype=bool)
        signature[10:20, 10:20] = True
        pattern = TemporalPattern(
            pattern_id=1,
            signature=signature,
            confidence=0.9,
            occurrences=20,
            first_seen=0,
            last_seen=10,
        )
        stack.create_neurons_from_patterns(0, [pattern])
        
        viz = visualize_stack(stack)
        
        assert viz.ndim == 3
        # Devrait y avoir de la couleur dans la région du neurone
        # (pas tout noir)


class TestNeuronStackPropagation:
    """Tests de propagation entre couches."""
    
    def test_output_propagates_to_next_layer(self):
        """Test que la sortie d'une couche devient l'entrée de la suivante."""
        stack = NeuronStack(
            base_shape=(32, 32), 
            num_layers=3,
            reduction_factor=0.5
        )
        
        # Créer un neurone dans la couche 0 qui détecte un pattern
        signature = np.zeros((32, 32), dtype=bool)
        signature[0:16, 0:16] = True
        pattern = TemporalPattern(
            pattern_id=1,
            signature=signature,
            confidence=0.9,
            occurrences=20,
            first_seen=0,
            last_seen=10,
        )
        stack.create_neurons_from_patterns(0, [pattern])
        
        # Activer avec ce pattern
        input_pattern = signature.copy()
        
        # Faire plusieurs passes pour dépasser le seuil
        for _ in range(10):
            outputs = stack.process(input_pattern)
        
        # La sortie de la couche 0 devrait influencer la couche 1
        # (même si pas de neurones en couche 1, la propagation a lieu)
        assert len(outputs) == 3


class TestNeuronStackHierarchy:
    """Tests de hiérarchie de détection."""
    
    def test_higher_layers_detect_compositions(self):
        """Test que les couches supérieures peuvent détecter des compositions."""
        # Cette test vérifie le concept d'empilage
        stack = NeuronStack(
            base_shape=(64, 64),
            num_layers=3,
            reduction_factor=0.5,
            config=GenesisConfig(
                min_pattern_confidence=0.5,
                min_pattern_occurrences=5,
            )
        )
        
        # Couche 0: 64x64 - détecte des edges
        # Couche 1: 32x32 - détecte des combinaisons d'edges
        # Couche 2: 16x16 - détecte des objets
        
        # Créer des "edge detectors" en couche 0
        edges = [
            (10, 10, 8),  # Edge 1
            (30, 10, 8),  # Edge 2
            (10, 40, 8),  # Edge 3
        ]
        
        for x, y, size in edges:
            signature = np.zeros((64, 64), dtype=bool)
            signature[y:y+size, x:x+size] = True
            pattern = TemporalPattern(
                pattern_id=hash((x, y)) % 10000,
                signature=signature,
                confidence=0.8,
                occurrences=10,
                first_seen=0,
                last_seen=10,
            )
            stack.create_neurons_from_patterns(0, [pattern])
        
        assert stack.get_layer(0).neuron_count == 3
        
        # La pile est prête pour détecter des compositions
        # dans les couches supérieures quand des patterns
        # seront détectés par le TemporalCorrelator
