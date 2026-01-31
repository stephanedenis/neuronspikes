"""Tests pour le détecteur de groupes d'activation."""

import numpy as np
import pytest

from neuronspikes.groups import (
    ActivationGroup,
    GroupDetector,
    GroupDetectorConfig,
    visualize_groups,
)


class TestActivationGroup:
    """Tests pour la classe ActivationGroup."""
    
    def test_group_creation(self):
        """Création d'un groupe simple."""
        pixels = {(0, 0), (0, 1), (1, 0), (1, 1)}
        group = ActivationGroup(pixels=pixels, slot=0, frame=0)
        
        assert group.size == 4
        assert group.centroid == (0.5, 0.5)
        assert group.bounding_box == (0, 0, 1, 1)
    
    def test_empty_group(self):
        """Groupe vide."""
        group = ActivationGroup(pixels=set(), slot=0, frame=0)
        assert group.size == 0
        assert group.centroid == (0.0, 0.0)
    
    def test_overlap_identical(self):
        """Groupes identiques ont overlap = 1."""
        pixels = {(0, 0), (0, 1), (1, 0)}
        g1 = ActivationGroup(pixels=pixels, slot=0, frame=0)
        g2 = ActivationGroup(pixels=pixels.copy(), slot=1, frame=0)
        
        assert g1.overlaps(g2, threshold=0.5)
        assert g1.overlaps(g2, threshold=1.0)
    
    def test_overlap_partial(self):
        """Groupes partiellement chevauchants."""
        g1 = ActivationGroup(pixels={(0, 0), (0, 1)}, slot=0, frame=0)
        g2 = ActivationGroup(pixels={(0, 1), (1, 1)}, slot=0, frame=0)
        
        # Jaccard = 1/3 ≈ 0.33
        assert g1.overlaps(g2, threshold=0.3)
        assert not g1.overlaps(g2, threshold=0.5)
    
    def test_overlap_disjoint(self):
        """Groupes disjoints."""
        g1 = ActivationGroup(pixels={(0, 0)}, slot=0, frame=0)
        g2 = ActivationGroup(pixels={(5, 5)}, slot=0, frame=0)
        
        assert not g1.overlaps(g2, threshold=0.1)
    
    def test_distance(self):
        """Distance entre centroïdes."""
        g1 = ActivationGroup(pixels={(0, 0)}, slot=0, frame=0)
        g2 = ActivationGroup(pixels={(3, 4)}, slot=0, frame=0)
        
        assert g1.distance_to(g2) == 5.0  # 3-4-5 triangle


class TestGroupDetector:
    """Tests pour le détecteur de groupes."""
    
    def test_no_activation(self):
        """Pas d'activation = pas de groupes."""
        detector = GroupDetector()
        activations = np.zeros((8, 8), dtype=np.bool_)
        
        groups = detector.detect_groups(activations)
        
        assert len(groups) == 0
    
    def test_single_pixel_filtered(self):
        """Un seul pixel est filtré (min_group_size=2)."""
        detector = GroupDetector(GroupDetectorConfig(min_group_size=2))
        activations = np.zeros((8, 8), dtype=np.bool_)
        activations[3, 3] = True
        
        groups = detector.detect_groups(activations)
        
        assert len(groups) == 0
    
    def test_single_pixel_included(self):
        """Un seul pixel est inclus si min_group_size=1."""
        detector = GroupDetector(GroupDetectorConfig(min_group_size=1))
        activations = np.zeros((8, 8), dtype=np.bool_)
        activations[3, 3] = True
        
        groups = detector.detect_groups(activations)
        
        assert len(groups) == 1
        assert groups[0].size == 1
    
    def test_connected_component_4(self):
        """Composante connexe en 4-connexité."""
        detector = GroupDetector(GroupDetectorConfig(
            min_group_size=1,
            connectivity=4
        ))
        
        # Carré 2x2
        activations = np.zeros((8, 8), dtype=np.bool_)
        activations[2, 2] = True
        activations[2, 3] = True
        activations[3, 2] = True
        activations[3, 3] = True
        
        groups = detector.detect_groups(activations)
        
        assert len(groups) == 1
        assert groups[0].size == 4
    
    def test_connected_component_8(self):
        """Composante connexe en 8-connexité (diagonales)."""
        detector = GroupDetector(GroupDetectorConfig(
            min_group_size=1,
            connectivity=8
        ))
        
        # Diagonale
        activations = np.zeros((8, 8), dtype=np.bool_)
        activations[0, 0] = True
        activations[1, 1] = True
        activations[2, 2] = True
        
        groups = detector.detect_groups(activations)
        
        # En 8-connexité, c'est un seul groupe
        assert len(groups) == 1
        assert groups[0].size == 3
    
    def test_diagonal_4_connectivity(self):
        """Diagonale en 4-connexité = groupes séparés."""
        detector = GroupDetector(GroupDetectorConfig(
            min_group_size=1,
            connectivity=4
        ))
        
        # Diagonale
        activations = np.zeros((8, 8), dtype=np.bool_)
        activations[0, 0] = True
        activations[1, 1] = True
        activations[2, 2] = True
        
        groups = detector.detect_groups(activations)
        
        # En 4-connexité, ce sont 3 groupes séparés
        assert len(groups) == 3
    
    def test_two_separate_groups(self):
        """Deux groupes distincts."""
        detector = GroupDetector(GroupDetectorConfig(min_group_size=2))
        
        activations = np.zeros((8, 8), dtype=np.bool_)
        # Groupe 1 en haut gauche
        activations[0, 0] = True
        activations[0, 1] = True
        # Groupe 2 en bas droite
        activations[6, 6] = True
        activations[6, 7] = True
        
        groups = detector.detect_groups(activations)
        
        assert len(groups) == 2
        assert all(g.size == 2 for g in groups)
    
    def test_statistics(self):
        """Vérification des statistiques."""
        detector = GroupDetector(GroupDetectorConfig(min_group_size=1))
        
        activations = np.zeros((8, 8), dtype=np.bool_)
        activations[0:3, 0:3] = True  # 9 pixels
        
        groups = detector.detect_groups(activations)
        
        assert detector.stats['total_groups'] == 1
        assert detector.stats['total_pixels_in_groups'] == 9
        assert detector.stats['max_group_size'] == 9
    
    def test_reset(self):
        """Reset des statistiques."""
        detector = GroupDetector(GroupDetectorConfig(min_group_size=1))
        
        activations = np.ones((4, 4), dtype=np.bool_)
        detector.detect_groups(activations)
        
        detector.reset()
        
        assert detector.stats['total_groups'] == 0
        assert len(detector.history) == 0


class TestVisualization:
    """Tests pour la visualisation."""
    
    def test_visualize_empty(self):
        """Visualisation sans groupes."""
        activations = np.zeros((4, 4), dtype=np.bool_)
        vis = visualize_groups(activations, [])
        
        assert vis.shape == (4, 4, 3)
        assert vis.sum() == 0
    
    def test_visualize_with_groups(self):
        """Visualisation avec groupes colorés."""
        activations = np.zeros((8, 8), dtype=np.bool_)
        activations[0, 0] = True
        activations[0, 1] = True
        
        group = ActivationGroup(pixels={(0, 0), (0, 1)}, slot=0, frame=0)
        vis = visualize_groups(activations, [group])
        
        # Les pixels du groupe doivent être colorés
        assert vis[0, 0].sum() > 0
        assert vis[0, 1].sum() > 0
        # Les autres doivent être noirs
        assert vis[4, 4].sum() == 0


class TestRecurringPatterns:
    """Tests pour la détection de patterns récurrents."""
    
    def test_recurring_pattern_detection(self):
        """Détection de patterns qui se répètent."""
        detector = GroupDetector(GroupDetectorConfig(
            min_group_size=2,
            track_history=5
        ))
        
        # Simuler le même pattern sur plusieurs frames
        activations = np.zeros((8, 8), dtype=np.bool_)
        activations[2:4, 2:4] = True  # Carré 2x2 fixe
        
        for frame in range(5):
            detector.detect_groups(activations, slot=0, frame=frame)
            detector.history.append([detector.detect_groups(activations, slot=0, frame=frame)])
        
        patterns = detector.get_recurring_patterns(min_occurrences=3)
        
        # Le carré 2x2 devrait être détecté comme pattern récurrent
        assert len(patterns) > 0
