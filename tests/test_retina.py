"""Tests pour la couche rétine."""

import numpy as np
import pytest

from neuronspikes.retina import (
    RetinaConfig,
    RetinaLayer,
    RetinaState,
    create_retina,
)


class TestRetinaConfig:
    """Tests pour la configuration de la rétine."""
    
    def test_default_config(self):
        """Configuration par défaut."""
        config = RetinaConfig()
        assert config.width == 64
        assert config.height == 64
        assert config.threshold == 1.0
        assert config.decay == 0.0
        assert config.fps == 60
    
    def test_custom_config(self):
        """Configuration personnalisée."""
        config = RetinaConfig(width=128, height=96, fps=30)
        assert config.width == 128
        assert config.height == 96
        assert config.fps == 30


class TestRetinaLayer:
    """Tests pour la couche rétine."""
    
    def test_creation(self):
        """Création d'une couche rétine."""
        layer = create_retina(32, 32, 60)
        assert layer.shape == (32, 32)
        assert layer.max_frequency == 15360  # 60 * 256
    
    def test_slot_duration(self):
        """Vérification de la durée d'un slot."""
        layer = create_retina(64, 64, 60)
        # 1/60 seconde = 16666.67 µs / 256 slots ≈ 65.1 µs
        assert abs(layer.slot_duration_us - 65.104) < 0.1
    
    def test_process_frame(self):
        """Traitement d'une frame."""
        layer = create_retina(4, 4, 60)
        frame = np.zeros((4, 4), dtype=np.uint8)
        frame[0, 0] = 100
        frame[1, 1] = 200
        
        layer.process_frame(frame)
        
        assert layer.state.frame_index == 1
        assert layer.stats['frames_processed'] == 1
    
    def test_process_frame_wrong_shape(self):
        """Erreur si la frame a une mauvaise forme."""
        layer = create_retina(4, 4, 60)
        frame = np.zeros((8, 8), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            layer.process_frame(frame)
    
    def test_get_activations(self):
        """Obtention des activations pour un slot."""
        layer = create_retina(4, 4, 60)
        
        # Frame avec un seul pixel allumé à intensité 1
        # Intensité 1 → slot 128 (bit reversal de 00000001 = 10000000)
        frame = np.zeros((4, 4), dtype=np.uint8)
        frame[2, 3] = 1
        
        layer.process_frame(frame)
        
        # Vérifier que le slot 128 a l'activation
        activations_128 = layer.get_activations(128)
        assert activations_128[2, 3] == True
        assert activations_128.sum() == 1
        
        # Le slot 0 ne devrait pas avoir d'activation
        activations_0 = layer.get_activations(0)
        assert activations_0.sum() == 0
    
    def test_step(self):
        """Avancement d'un slot temporel."""
        layer = create_retina(4, 4, 60)
        frame = np.full((4, 4), 255, dtype=np.uint8)  # Tous les pixels à max
        
        layer.process_frame(frame)
        
        initial_slot = layer.state.slot_index
        activations = layer.step()
        
        assert layer.state.slot_index == initial_slot + 1
        assert activations.shape == (4, 4)
    
    def test_run_frame_256_steps(self):
        """Exécution des 256 slots d'une frame."""
        layer = create_retina(4, 4, 60)
        frame = np.full((4, 4), 128, dtype=np.uint8)
        
        layer.process_frame(frame)
        all_activations = layer.run_frame()
        
        assert len(all_activations) == 256
        
        # Chaque pixel devrait avoir 128 activations au total
        total_per_pixel = np.zeros((4, 4), dtype=np.int32)
        for activations in all_activations:
            total_per_pixel += activations.astype(np.int32)
        
        assert np.all(total_per_pixel == 128)
    
    def test_activation_pattern(self):
        """Le pattern d'activation doit correspondre à l'intensité."""
        layer = create_retina(4, 4, 60)
        frame = np.array([
            [0, 50, 100, 150],
            [200, 255, 1, 2],
            [10, 20, 30, 40],
            [128, 64, 32, 16]
        ], dtype=np.uint8)
        
        layer.process_frame(frame)
        pattern = layer.get_activation_pattern()
        
        # Le pattern devrait être identique à la frame
        assert np.array_equal(pattern, frame)
    
    def test_reset(self):
        """Réinitialisation de la couche."""
        layer = create_retina(4, 4, 60)
        frame = np.full((4, 4), 100, dtype=np.uint8)
        
        layer.process_frame(frame)
        layer.step()
        
        layer.reset()
        
        assert layer.state.frame_index == 0
        assert layer.state.slot_index == 0
        assert layer.stats['frames_processed'] == 0
        assert layer.stats['total_spikes'] == 0
    
    def test_deterministic(self):
        """Le traitement doit être parfaitement déterministe."""
        frame = np.arange(256, dtype=np.uint8).reshape(16, 16)
        
        layer1 = create_retina(16, 16, 60)
        layer2 = create_retina(16, 16, 60)
        
        layer1.process_frame(frame)
        layer2.process_frame(frame)
        
        activations1 = layer1.run_frame()
        activations2 = layer2.run_frame()
        
        for a1, a2 in zip(activations1, activations2):
            assert np.array_equal(a1, a2)


class TestRetinaIntegration:
    """Tests d'intégration pour la rétine."""
    
    def test_black_frame_no_spikes(self):
        """Une frame noire ne devrait produire aucune impulsion."""
        layer = create_retina(32, 32, 60)
        frame = np.zeros((32, 32), dtype=np.uint8)
        
        layer.process_frame(frame)
        layer.run_frame()
        
        assert layer.stats['total_spikes'] == 0
    
    def test_white_frame_max_spikes(self):
        """Une frame blanche devrait produire le max d'impulsions."""
        layer = create_retina(8, 8, 60)  # 64 pixels
        frame = np.full((8, 8), 255, dtype=np.uint8)
        
        layer.process_frame(frame)
        layer.run_frame()
        
        # 64 pixels × 255 impulsions chacun = 16320 impulsions
        assert layer.stats['total_spikes'] == 64 * 255
    
    def test_gradient_frame(self):
        """Frame avec gradient - vérifier la cohérence."""
        layer = create_retina(16, 16, 60)
        
        # Créer un gradient 0-255
        gradient = np.arange(256, dtype=np.uint8).reshape(16, 16)
        
        layer.process_frame(gradient)
        layer.run_frame()
        
        # Somme des intensités = somme des impulsions
        expected_spikes = int(gradient.sum())
        assert layer.stats['total_spikes'] == expected_spikes
    
    def test_multiple_frames(self):
        """Traitement de plusieurs frames consécutives."""
        layer = create_retina(8, 8, 60)
        
        for i in range(10):
            frame = np.full((8, 8), i * 25, dtype=np.uint8)
            layer.process_frame(frame)
            layer.run_frame()
        
        assert layer.stats['frames_processed'] == 10
        assert layer.state.frame_index == 10
