"""Tests pour le module de corrélations temporelles."""

import numpy as np
import pytest
from neuronspikes.temporal import (
    TemporalPattern,
    TemporalCorrelator,
    CorrelationConfig,
    visualize_patterns,
)
from neuronspikes.groups import ActivationGroup, GroupDetector, GroupDetectorConfig


def make_group(pixels: set, slot: int = 0, frame: int = 0) -> ActivationGroup:
    """Helper pour créer des groupes d'activation facilement."""
    return ActivationGroup(pixels=pixels, slot=slot, frame=frame)


class TestTemporalPattern:
    """Tests pour TemporalPattern."""
    
    def test_creation(self):
        """Test de création basique."""
        signature = np.zeros((10, 10), dtype=bool)
        signature[2:5, 3:7] = True
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=0,
        )
        
        assert pattern.pattern_id == 0
        assert pattern.pixel_count == 12  # 3x4 pixels
        assert pattern.occurrences == 1
        assert pattern.age == 0
    
    def test_centroid_calculation(self):
        """Test du calcul automatique du centroïde."""
        signature = np.zeros((10, 10), dtype=bool)
        signature[4:6, 4:6] = True  # 2x2 au centre
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=0,
        )
        
        # Centroïde devrait être au milieu du bloc 2x2
        assert pattern.centroid[0] == pytest.approx(4.5, abs=0.1)
        assert pattern.centroid[1] == pytest.approx(4.5, abs=0.1)
    
    def test_age(self):
        """Test du calcul de l'âge."""
        signature = np.ones((5, 5), dtype=bool)
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=10,
            last_seen=25,
        )
        
        assert pattern.age == 15
    
    def test_frequency(self):
        """Test du calcul de fréquence."""
        signature = np.ones((5, 5), dtype=bool)
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=9,
            occurrences=5,
        )
        
        # 5 occurrences sur 10 frames (0 à 9 inclus)
        assert pattern.frequency == pytest.approx(0.5, abs=0.01)
    
    def test_frequency_single_frame(self):
        """Test de fréquence sur un seul frame."""
        signature = np.ones((5, 5), dtype=bool)
        
        pattern = TemporalPattern(
            pattern_id=0,
            signature=signature,
            first_seen=0,
            last_seen=0,
            occurrences=1,
        )
        
        assert pattern.frequency == 1.0


class TestCorrelationConfig:
    """Tests pour CorrelationConfig."""
    
    def test_default_values(self):
        """Test des valeurs par défaut."""
        config = CorrelationConfig()
        
        assert config.history_size == 30
        assert config.min_overlap == 0.7
        assert config.min_occurrences == 3
        assert config.confidence_threshold == 0.5
        assert config.decay_rate == 0.95
    
    def test_custom_values(self):
        """Test des valeurs personnalisées."""
        config = CorrelationConfig(
            history_size=60,
            min_overlap=0.5,
            min_occurrences=5,
        )
        
        assert config.history_size == 60
        assert config.min_overlap == 0.5
        assert config.min_occurrences == 5


class TestTemporalCorrelator:
    """Tests pour TemporalCorrelator."""
    
    def test_creation(self):
        """Test de création."""
        correlator = TemporalCorrelator((64, 64))
        
        assert correlator.shape == (64, 64)
        assert correlator.pattern_count == 0
        assert len(correlator.stable_patterns) == 0
    
    def test_creation_with_config(self):
        """Test avec configuration personnalisée."""
        config = CorrelationConfig(history_size=10)
        correlator = TemporalCorrelator((32, 32), config)
        
        assert correlator.config.history_size == 10
    
    def test_process_single_group(self):
        """Test du traitement d'un groupe unique."""
        correlator = TemporalCorrelator((10, 10))
        
        # Groupe 3x3 aux positions (2,2) à (4,4)
        pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)}
        group = make_group(pixels)
        
        patterns = correlator.process_groups([group])
        
        assert len(patterns) == 1
        assert correlator.pattern_count == 1
        assert patterns[0].pixel_count == 9
    
    def test_pattern_matching(self):
        """Test du matching de patterns similaires."""
        config = CorrelationConfig(min_overlap=0.5)
        correlator = TemporalCorrelator((10, 10), config)
        
        # Groupe 3x3
        pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)}
        group1 = make_group(pixels, slot=0, frame=0)
        
        correlator.process_groups([group1])
        initial_patterns = correlator.pattern_count
        
        # Deuxième groupe identique
        group2 = make_group(pixels, slot=0, frame=1)
        
        correlator.process_groups([group2])
        
        # Ne devrait pas créer de nouveau pattern
        assert correlator.pattern_count == initial_patterns
        
        # Le pattern devrait avoir 2 occurrences
        pattern = list(correlator._patterns.values())[0]
        assert pattern.occurrences == 2
    
    def test_different_groups_create_different_patterns(self):
        """Test que des groupes différents créent des patterns différents."""
        config = CorrelationConfig(min_overlap=0.7)
        correlator = TemporalCorrelator((20, 20), config)
        
        # Premier groupe en haut à gauche (5x5)
        pixels1 = {(y, x) for y in range(0, 5) for x in range(0, 5)}
        group1 = make_group(pixels1)
        
        # Deuxième groupe en bas à droite (5x5)
        pixels2 = {(y, x) for y in range(15, 20) for x in range(15, 20)}
        group2 = make_group(pixels2)
        
        correlator.process_groups([group1, group2])
        
        # Devrait créer 2 patterns distincts
        assert correlator.pattern_count == 2
    
    def test_confidence_increases_with_occurrences(self):
        """Test que la confiance augmente avec les occurrences."""
        config = CorrelationConfig(min_overlap=0.5)
        correlator = TemporalCorrelator((10, 10), config)
        
        pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)}
        group = make_group(pixels)
        
        # Premier passage
        correlator.process_groups([group])
        initial_confidence = list(correlator._patterns.values())[0].confidence
        
        # Plusieurs passages supplémentaires
        for _ in range(5):
            correlator.process_groups([group])
        
        final_confidence = list(correlator._patterns.values())[0].confidence
        
        assert final_confidence > initial_confidence
    
    def test_confidence_decays_when_not_seen(self):
        """Test que la confiance décroît quand le pattern n'est pas vu."""
        config = CorrelationConfig(min_overlap=0.5, decay_rate=0.9)
        correlator = TemporalCorrelator((10, 10), config)
        
        pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)}
        group = make_group(pixels)
        
        # Créer le pattern
        correlator.process_groups([group])
        
        # Augmenter la confiance
        for _ in range(5):
            correlator.process_groups([group])
        
        confidence_before = list(correlator._patterns.values())[0].confidence
        
        # Frames vides (pattern non vu)
        for _ in range(5):
            correlator.process_groups([])
        
        confidence_after = list(correlator._patterns.values())[0].confidence
        
        assert confidence_after < confidence_before
    
    def test_stable_patterns(self):
        """Test de la détection des patterns stables."""
        config = CorrelationConfig(
            min_overlap=0.5,
            min_occurrences=3,
            confidence_threshold=0.3,
        )
        correlator = TemporalCorrelator((10, 10), config)
        
        pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)}
        group = make_group(pixels)
        
        # Pas encore stable
        correlator.process_groups([group])
        correlator.process_groups([group])
        assert len(correlator.stable_patterns) == 0
        
        # Maintenant stable (3+ occurrences)
        correlator.process_groups([group])
        correlator.process_groups([group])
        assert len(correlator.stable_patterns) >= 1
    
    def test_pattern_map(self):
        """Test de la génération de la carte des patterns."""
        config = CorrelationConfig(
            min_overlap=0.5,
            min_occurrences=2,
            confidence_threshold=0.2,
        )
        correlator = TemporalCorrelator((10, 10), config)
        
        pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)}
        group = make_group(pixels)
        
        # Rendre le pattern stable
        for _ in range(3):
            correlator.process_groups([group])
        
        pattern_map = correlator.get_pattern_map()
        
        assert pattern_map.shape == (10, 10)
        assert pattern_map[3, 3] > 0  # Pixel au centre du groupe
        assert pattern_map[0, 0] == 0  # Pixel hors du groupe
    
    def test_confidence_map(self):
        """Test de la génération de la carte de confiance."""
        config = CorrelationConfig(min_overlap=0.5)
        correlator = TemporalCorrelator((10, 10), config)
        
        pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)}
        group = make_group(pixels)
        
        for _ in range(5):
            correlator.process_groups([group])
        
        conf_map = correlator.get_confidence_map()
        
        assert conf_map.shape == (10, 10)
        assert conf_map[3, 3] > 0  # Confiance au centre du groupe
        assert conf_map[0, 0] == 0  # Pas de confiance hors groupe
    
    def test_reset(self):
        """Test de la réinitialisation."""
        correlator = TemporalCorrelator((10, 10))
        
        pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)}
        group = make_group(pixels)
        
        correlator.process_groups([group])
        correlator.process_groups([group])
        
        assert correlator.pattern_count > 0
        
        correlator.reset()
        
        assert correlator.pattern_count == 0
        assert correlator._frame_count == 0
    
    def test_get_stats(self):
        """Test des statistiques."""
        correlator = TemporalCorrelator((10, 10))
        
        pixels = {(y, x) for y in range(2, 5) for x in range(2, 5)}
        group = make_group(pixels)
        
        correlator.process_groups([group])
        correlator.process_groups([group])
        
        stats = correlator.get_stats()
        
        assert 'frame_count' in stats
        assert 'active_patterns' in stats
        assert 'stable_patterns' in stats
        assert stats['frame_count'] == 2
        assert stats['active_patterns'] == 1


class TestVisualizePatterns:
    """Tests pour la visualisation des patterns."""
    
    def test_empty_map(self):
        """Test avec une carte vide."""
        pattern_map = np.zeros((10, 10), dtype=np.int32)
        
        output = visualize_patterns(pattern_map)
        
        assert output.shape == (10, 10, 3)
        assert output.dtype == np.uint8
        assert np.all(output == 0)
    
    def test_single_pattern(self):
        """Test avec un seul pattern."""
        pattern_map = np.zeros((10, 10), dtype=np.int32)
        pattern_map[2:5, 2:5] = 1
        
        output = visualize_patterns(pattern_map)
        
        assert output.shape == (10, 10, 3)
        # La zone du pattern devrait avoir une couleur
        assert np.any(output[3, 3] > 0)
        # L'extérieur devrait être noir
        assert np.all(output[0, 0] == 0)
    
    def test_multiple_patterns(self):
        """Test avec plusieurs patterns."""
        pattern_map = np.zeros((20, 20), dtype=np.int32)
        pattern_map[0:5, 0:5] = 1
        pattern_map[15:20, 15:20] = 2
        
        output = visualize_patterns(pattern_map)
        
        # Les deux zones devraient avoir des couleurs différentes
        color1 = output[2, 2]
        color2 = output[17, 17]
        
        assert not np.array_equal(color1, color2)
    
    def test_with_confidence_map(self):
        """Test avec carte de confiance."""
        pattern_map = np.zeros((10, 10), dtype=np.int32)
        pattern_map[2:8, 2:8] = 1
        
        confidence_map = np.zeros((10, 10), dtype=np.float32)
        confidence_map[2:5, 2:5] = 1.0  # Haute confiance
        confidence_map[5:8, 5:8] = 0.3  # Basse confiance
        
        output = visualize_patterns(pattern_map, confidence_map)
        
        # La zone haute confiance devrait être plus lumineuse
        # (difficile à tester précisément, on vérifie juste que ça ne plante pas)
        assert output.shape == (10, 10, 3)


class TestIntegration:
    """Tests d'intégration avec GroupDetector."""
    
    def test_full_pipeline(self):
        """Test du pipeline complet groupes -> corrélations."""
        detector = GroupDetector()  # Utilise config par défaut
        correlator = TemporalCorrelator((20, 20))
        
        # Simuler une activation qui se répète (masque booléen)
        activation = np.zeros((20, 20), dtype=bool)
        activation[5:10, 5:10] = True  # Bloc 5x5
        
        # Plusieurs frames
        for frame_idx in range(10):
            groups = detector.detect_groups(activation, slot=0, frame=frame_idx)
            correlator.process_groups(groups)
        
        # Devrait avoir détecté un pattern récurrent
        assert correlator.pattern_count >= 1
        
        stats = correlator.get_stats()
        assert stats['frame_count'] == 10
