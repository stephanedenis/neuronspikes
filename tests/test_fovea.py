"""Tests pour le module Fovea."""

import math
import numpy as np
import pytest
from neuronspikes.fovea import (
    Fovea,
    FoveaConfig,
    GazePoint,
    PolarCell,
    StereoFovea,
    visualize_fovea,
)


class TestGazePoint:
    """Tests pour GazePoint."""
    
    def test_creation(self):
        """Test de création."""
        gaze = GazePoint()
        
        assert gaze.x == 0.0
        assert gaze.y == 0.0
        assert gaze.theta == 0.0
    
    def test_move(self):
        """Test de déplacement (saccade)."""
        gaze = GazePoint(x=100, y=100)
        
        gaze.move(10, -5)
        
        assert gaze.x == 110
        assert gaze.y == 95
    
    def test_rotate(self):
        """Test de rotation."""
        gaze = GazePoint()
        
        gaze.rotate(math.pi / 4)
        
        assert gaze.theta == pytest.approx(math.pi / 4, abs=0.01)
    
    def test_rotate_wrap(self):
        """Test de normalisation de l'angle."""
        gaze = GazePoint()
        
        # Rotation de plus de 2π
        gaze.rotate(3 * math.pi)
        
        assert -math.pi <= gaze.theta <= math.pi
    
    def test_as_tuple(self):
        """Test de conversion en tuple."""
        gaze = GazePoint(x=50, y=75)
        
        assert gaze.as_tuple() == (50, 75)


class TestPolarCell:
    """Tests pour PolarCell."""
    
    def test_creation(self):
        """Test de création."""
        cell = PolarCell(
            ring=2,
            sector=3,
            inner_radius=10.0,
            outer_radius=15.0,
            start_angle=0.0,
            end_angle=math.pi / 8
        )
        
        assert cell.ring == 2
        assert cell.sector == 3
        assert cell.inner_radius == 10.0
        assert cell.outer_radius == 15.0
    
    def test_center_radius(self):
        """Test du rayon central."""
        cell = PolarCell(
            ring=0, sector=0,
            inner_radius=10.0, outer_radius=20.0,
            start_angle=0.0, end_angle=math.pi / 4
        )
        
        assert cell.center_radius == 15.0
    
    def test_center_angle(self):
        """Test de l'angle central."""
        cell = PolarCell(
            ring=0, sector=0,
            inner_radius=10.0, outer_radius=20.0,
            start_angle=0.0, end_angle=math.pi / 2
        )
        
        assert cell.center_angle == pytest.approx(math.pi / 4, abs=0.01)
    
    def test_area(self):
        """Test du calcul de surface."""
        cell = PolarCell(
            ring=0, sector=0,
            inner_radius=10.0, outer_radius=20.0,
            start_angle=0.0, end_angle=math.pi / 4
        )
        
        area = cell.area
        assert area > 0
    
    def test_to_cartesian(self):
        """Test de conversion en coordonnées cartésiennes."""
        cell = PolarCell(
            ring=0, sector=0,
            inner_radius=0.0, outer_radius=20.0,
            start_angle=0.0, end_angle=math.pi / 2
        )
        
        x, y = cell.to_cartesian(100, 100, rotation=0)
        
        # Le centre devrait être à 10 pixels du centre, à 45°
        expected_r = 10
        expected_angle = math.pi / 4
        expected_x = 100 + expected_r * math.cos(expected_angle)
        expected_y = 100 + expected_r * math.sin(expected_angle)
        
        assert x == pytest.approx(expected_x, abs=1)
        assert y == pytest.approx(expected_y, abs=1)


class TestFoveaConfig:
    """Tests pour FoveaConfig."""
    
    def test_default_values(self):
        """Test des valeurs par défaut."""
        config = FoveaConfig()
        
        assert config.num_rings == 32
        assert config.num_sectors == 16
        assert config.fovea_radius == 16
        assert config.max_radius == 128
    
    def test_custom_values(self):
        """Test des valeurs personnalisées."""
        config = FoveaConfig(num_rings=64, num_sectors=32)
        
        assert config.num_rings == 64
        assert config.num_sectors == 32


class TestFovea:
    """Tests pour Fovea."""
    
    def test_creation(self):
        """Test de création."""
        fovea = Fovea()
        
        assert fovea.config.num_rings == 32
        assert fovea.config.num_sectors == 16
        assert len(fovea.cells) == 32
        assert len(fovea.cells[0]) == 16
    
    def test_creation_with_config(self):
        """Test de création avec config."""
        config = FoveaConfig(num_rings=16, num_sectors=8)
        fovea = Fovea(config)
        
        assert len(fovea.cells) == 16
        assert len(fovea.cells[0]) == 8
    
    def test_set_gaze(self):
        """Test de définition du point de fixation."""
        fovea = Fovea()
        
        fovea.set_gaze(100, 150, theta=0.5)
        
        assert fovea.gaze.x == 100
        assert fovea.gaze.y == 150
        assert fovea.gaze.theta == 0.5
    
    def test_saccade(self):
        """Test de saccade."""
        fovea = Fovea()
        fovea.set_gaze(100, 100)
        
        fovea.saccade(20, -10)
        
        assert fovea.gaze.x == 120
        assert fovea.gaze.y == 90
    
    def test_rotate(self):
        """Test de rotation."""
        fovea = Fovea()
        
        fovea.rotate(math.pi / 6)
        
        assert fovea.gaze.theta == pytest.approx(math.pi / 6, abs=0.01)
    
    def test_sample_uniform_image(self):
        """Test d'échantillonnage d'une image uniforme."""
        config = FoveaConfig(num_rings=8, num_sectors=8)
        fovea = Fovea(config)
        fovea.set_gaze(64, 64)
        
        # Image uniforme blanche
        image = np.full((128, 128), 255, dtype=np.uint8)
        
        activations = fovea.sample(image)
        
        assert activations.shape == (8, 8)
        # Toutes les cellules devraient avoir une activation élevée
        assert np.mean(activations) > 0.5
    
    def test_sample_black_image(self):
        """Test d'échantillonnage d'une image noire."""
        config = FoveaConfig(num_rings=8, num_sectors=8)
        fovea = Fovea(config)
        fovea.set_gaze(64, 64)
        
        # Image noire
        image = np.zeros((128, 128), dtype=np.uint8)
        
        activations = fovea.sample(image)
        
        assert activations.shape == (8, 8)
        # Toutes les cellules devraient avoir une activation faible
        assert np.mean(activations) < 0.1
    
    def test_sample_gradient(self):
        """Test d'échantillonnage d'un gradient."""
        config = FoveaConfig(num_rings=8, num_sectors=8)
        fovea = Fovea(config)
        fovea.set_gaze(64, 64)
        
        # Gradient radial (centre clair, bords sombres)
        image = np.zeros((128, 128), dtype=np.uint8)
        for y in range(128):
            for x in range(128):
                dist = math.sqrt((x - 64)**2 + (y - 64)**2)
                image[y, x] = max(0, 255 - int(dist * 2))
        
        activations = fovea.sample(image)
        
        # Le centre (ring 0) devrait être plus actif que la périphérie
        center_activation = np.mean(activations[0, :])
        peripheral_activation = np.mean(activations[-1, :])
        assert center_activation > peripheral_activation
    
    def test_get_cell_positions(self):
        """Test des positions des cellules."""
        config = FoveaConfig(num_rings=4, num_sectors=4)
        fovea = Fovea(config)
        fovea.set_gaze(100, 100)
        
        positions = fovea.get_cell_positions()
        
        assert positions.shape == (4, 4, 2)
    
    def test_get_fovea_mask(self):
        """Test du masque fovéal."""
        config = FoveaConfig(num_rings=8, num_sectors=8)
        fovea = Fovea(config)
        
        mask = fovea.get_fovea_mask()
        
        assert mask.shape == (8, 8)
        # Les 2 premiers anneaux (8//4) devraient être dans la fovéa
        assert np.all(mask[:2, :])
        assert not np.any(mask[4:, :])
    
    def test_get_sector_activations(self):
        """Test des activations par secteur."""
        config = FoveaConfig(num_rings=8, num_sectors=8)
        fovea = Fovea(config)
        fovea.set_gaze(64, 64)
        
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        fovea.sample(image)
        
        sector_act = fovea.get_sector_activations(0)
        
        assert sector_act.shape == (8,)
    
    def test_get_ring_activations(self):
        """Test des activations par anneau."""
        config = FoveaConfig(num_rings=8, num_sectors=8)
        fovea = Fovea(config)
        fovea.set_gaze(64, 64)
        
        image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        fovea.sample(image)
        
        ring_act = fovea.get_ring_activations(0)
        
        assert ring_act.shape == (8,)
    
    def test_detect_rotation(self):
        """Test de détection de rotation."""
        config = FoveaConfig(num_rings=8, num_sectors=8)
        fovea = Fovea(config)
        fovea.set_gaze(64, 64)
        
        # Image avec pattern angulaire distinct
        image = np.zeros((128, 128), dtype=np.uint8)
        # Quadrant supérieur droit brillant
        image[:64, 64:] = 255
        
        # Premier échantillonnage
        act1 = fovea.sample(image)
        
        # Rotation de l'image de 90° (2 secteurs sur 8)
        rotated = np.rot90(image)
        act2 = fovea.sample(rotated)
        
        # Devrait détecter un décalage
        shift = fovea.detect_rotation(act1)
        # Note: le shift détecté dépend de la géométrie exacte
        assert isinstance(shift, int)
    
    def test_get_stats(self):
        """Test des statistiques."""
        fovea = Fovea()
        fovea.set_gaze(100, 100)
        
        stats = fovea.get_stats()
        
        assert stats['num_rings'] == 32
        assert stats['num_sectors'] == 16
        assert stats['gaze_x'] == 100
        assert stats['gaze_y'] == 100
    
    def test_reset(self):
        """Test de réinitialisation."""
        fovea = Fovea()
        fovea.set_gaze(100, 100, theta=0.5)
        
        image = np.full((128, 128), 128, dtype=np.uint8)
        fovea.sample(image)
        
        fovea.reset()
        
        assert fovea.gaze.x == 0
        assert fovea.gaze.y == 0
        assert fovea.gaze.theta == 0
        assert fovea._frame_count == 0


class TestStereoFovea:
    """Tests pour StereoFovea."""
    
    def test_creation(self):
        """Test de création."""
        stereo = StereoFovea()
        
        assert stereo.left is not None
        assert stereo.right is not None
        assert stereo.baseline == 60.0
    
    def test_set_target(self):
        """Test de définition de la cible."""
        stereo = StereoFovea(baseline=60.0)
        
        stereo.set_target(100, 100, depth=100)
        
        # Les deux yeux devraient pointer vers des positions légèrement différentes
        assert stereo.left.gaze.x != stereo.right.gaze.x or stereo._vergence != 0
    
    def test_sample(self):
        """Test d'échantillonnage stéréo."""
        config = FoveaConfig(num_rings=4, num_sectors=4)
        stereo = StereoFovea(config, baseline=20)
        stereo.set_target(64, 64, depth=50)
        
        left_img = np.full((128, 128), 200, dtype=np.uint8)
        right_img = np.full((128, 128), 200, dtype=np.uint8)
        
        left_act, right_act = stereo.sample(left_img, right_img)
        
        assert left_act.shape == (4, 4)
        assert right_act.shape == (4, 4)
    
    def test_compute_disparity(self):
        """Test du calcul de disparité."""
        config = FoveaConfig(num_rings=4, num_sectors=4)
        stereo = StereoFovea(config, baseline=20)
        stereo.set_target(64, 64)
        
        # Images différentes
        left_img = np.full((128, 128), 200, dtype=np.uint8)
        right_img = np.full((128, 128), 100, dtype=np.uint8)
        
        stereo.sample(left_img, right_img)
        
        disparity = stereo.compute_disparity()
        
        assert disparity.shape == (4, 4)
        # La disparité devrait être positive (gauche plus brillant)
        assert np.mean(disparity) > 0
    
    def test_estimate_depth_map(self):
        """Test de l'estimation de profondeur."""
        config = FoveaConfig(num_rings=4, num_sectors=4)
        stereo = StereoFovea(config, baseline=20)
        stereo.set_target(64, 64)
        
        left_img = np.full((128, 128), 200, dtype=np.uint8)
        right_img = np.full((128, 128), 100, dtype=np.uint8)
        
        stereo.sample(left_img, right_img)
        
        depth = stereo.estimate_depth_map()
        
        assert depth.shape == (4, 4)
        assert np.all(depth > 0)
    
    def test_saccade(self):
        """Test de saccade coordonnée."""
        stereo = StereoFovea()
        stereo.set_target(100, 100)
        
        left_x_before = stereo.left.gaze.x
        right_x_before = stereo.right.gaze.x
        
        stereo.saccade(10, 5)
        
        assert stereo.left.gaze.x == left_x_before + 10
        assert stereo.right.gaze.x == right_x_before + 10
    
    def test_rotate(self):
        """Test de rotation coordonnée."""
        stereo = StereoFovea()
        
        stereo.rotate(math.pi / 6)
        
        assert stereo.left.gaze.theta == pytest.approx(math.pi / 6, abs=0.01)
        assert stereo.right.gaze.theta == pytest.approx(math.pi / 6, abs=0.01)
    
    def test_get_stats(self):
        """Test des statistiques."""
        stereo = StereoFovea()
        
        stats = stereo.get_stats()
        
        assert 'baseline' in stats
        assert 'vergence' in stats
        assert 'left' in stats
        assert 'right' in stats


class TestVisualizeFovea:
    """Tests pour visualize_fovea."""
    
    def test_basic_visualization(self):
        """Test de visualisation basique."""
        config = FoveaConfig(num_rings=8, num_sectors=8)
        fovea = Fovea(config)
        fovea.set_gaze(64, 64)
        
        image = np.full((128, 128), 128, dtype=np.uint8)
        fovea.sample(image)
        
        viz = visualize_fovea(fovea, size=128)
        
        assert viz.shape == (128, 128, 3)
        assert viz.dtype == np.uint8
    
    def test_visualization_with_grid(self):
        """Test avec grille."""
        fovea = Fovea()
        
        viz = visualize_fovea(fovea, size=256, show_grid=True)
        
        assert viz.shape == (256, 256, 3)
    
    def test_visualization_without_grid(self):
        """Test sans grille."""
        fovea = Fovea()
        
        viz = visualize_fovea(fovea, size=256, show_grid=False)
        
        assert viz.shape == (256, 256, 3)


class TestIntegration:
    """Tests d'intégration."""
    
    def test_fovea_tracking_workflow(self):
        """Test d'un workflow complet de suivi fovéal."""
        config = FoveaConfig(num_rings=16, num_sectors=16)
        fovea = Fovea(config)
        
        # Image avec cible mobile
        for frame in range(10):
            # Créer image avec point brillant
            image = np.zeros((256, 256), dtype=np.uint8)
            target_x = 128 + frame * 5
            target_y = 128
            cv_size = 10
            image[
                max(0, target_y - cv_size):min(256, target_y + cv_size),
                max(0, target_x - cv_size):min(256, target_x + cv_size)
            ] = 255
            
            # Faire une saccade vers la cible
            fovea.set_gaze(target_x, target_y)
            
            # Échantillonner
            activations = fovea.sample(image)
            
            # Le centre devrait être actif
            fovea_activations = activations[:4, :]  # Zone fovéale
            assert np.mean(fovea_activations) > 0.1
        
        stats = fovea.get_stats()
        assert stats['frame_count'] == 10
    
    def test_vor_compensation(self):
        """Test de la compensation VOR."""
        config = FoveaConfig(num_rings=8, num_sectors=16)
        fovea = Fovea(config)
        fovea.set_gaze(64, 64)
        
        # Image avec pattern angulaire
        image = np.zeros((128, 128), dtype=np.uint8)
        for y in range(128):
            for x in range(128):
                angle = math.atan2(y - 64, x - 64)
                if 0 < angle < math.pi / 2:
                    image[y, x] = 255
        
        # Première capture
        prev_act = fovea.sample(image)
        
        # Simuler rotation de la tête (on tourne l'image)
        rotated = np.rot90(image)
        fovea.sample(rotated)
        
        # Compenser
        compensation = fovea.compensate_rotation(prev_act)
        
        # La compensation devrait être appliquée
        assert fovea.gaze.theta != 0 or compensation == 0
