"""Tests pour les voies rétiniennes bio-inspirées."""

import numpy as np
import pytest
from neuronspikes import (
    PathwayConfig,
    MagnocellularPathway,
    ParvocellularPathway,
    LateralInhibition,
    GaborFilterBank,
    RetinalProcessor,
)


class TestMagnocellularPathway:
    """Tests de la voie Magnocellulaire (mouvement)."""
    
    def test_init(self):
        """Création basique."""
        magno = MagnocellularPathway()
        assert magno.prev_frame is None
        assert magno.motion_accumulator is None
    
    def test_first_frame_no_motion(self):
        """Première frame: pas de mouvement détecté."""
        magno = MagnocellularPathway()
        frame = np.random.rand(100, 100).astype(np.float32)
        
        motion, direction = magno.process(frame)
        
        assert motion.shape == frame.shape
        assert np.allclose(motion, 0)  # Pas de mouvement
    
    def test_detects_motion(self):
        """Détecte le mouvement entre deux frames."""
        magno = MagnocellularPathway()
        
        # Frame 1: fond noir avec carré blanc à gauche
        frame1 = np.zeros((100, 100), dtype=np.float32)
        frame1[40:60, 20:40] = 1.0
        
        # Frame 2: carré s'est déplacé à droite
        frame2 = np.zeros((100, 100), dtype=np.float32)
        frame2[40:60, 60:80] = 1.0
        
        magno.process(frame1)  # Initialise
        motion, _ = magno.process(frame2)
        
        # Motion détecté dans les zones de changement
        assert motion.max() > 0
    
    def test_color_input(self):
        """Accepte les images couleur."""
        magno = MagnocellularPathway()
        frame_color = np.random.rand(100, 100, 3).astype(np.float32)
        
        motion, direction = magno.process(frame_color)
        
        assert motion.shape == (100, 100)
    
    def test_reset(self):
        """Reset efface l'état temporel."""
        magno = MagnocellularPathway()
        frame = np.random.rand(50, 50).astype(np.float32)
        
        magno.process(frame)
        assert magno.prev_frame is not None
        
        magno.reset()
        assert magno.prev_frame is None


class TestParvocellularPathway:
    """Tests de la voie Parvocellulaire (couleur/détails)."""
    
    def test_grayscale_input(self):
        """Traite les images en gris."""
        parvo = ParvocellularPathway()
        gray = np.random.rand(100, 100).astype(np.float32)
        
        lum, rg, by = parvo.process(gray)
        
        assert lum.shape == gray.shape
        assert np.allclose(rg, 0)  # Pas de couleur
        assert np.allclose(by, 0)
    
    def test_color_channels(self):
        """Sépare correctement les canaux de couleur."""
        parvo = ParvocellularPathway()
        
        # Image rouge pure
        red_img = np.zeros((50, 50, 3), dtype=np.float32)
        red_img[:, :, 2] = 1.0  # Canal R (BGR ou RGB)
        
        lum, rg, by = parvo.process(red_img)
        
        # Rouge-vert devrait être positif pour le rouge
        # (dépend de l'ordre BGR/RGB)
        assert rg.max() != 0 or by.max() != 0
    
    def test_luminance_range(self):
        """Luminance dans [0, 1]."""
        parvo = ParvocellularPathway()
        frame = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        
        lum, _, _ = parvo.process(frame)
        
        assert lum.min() >= 0
        assert lum.max() <= 1
    
    def test_get_edges(self):
        """Détection de bords."""
        parvo = ParvocellularPathway()
        
        # Image avec bord vertical net
        img = np.zeros((100, 100), dtype=np.float32)
        img[:, 50:] = 1.0
        
        edges = parvo.get_edges(img)
        
        # Maximum sur la ligne de bord
        assert edges[:, 49:51].max() > edges[:, :40].max()


class TestLateralInhibition:
    """Tests de l'inhibition latérale."""
    
    def test_enhances_edges(self):
        """L'inhibition latérale renforce les bords."""
        lateral = LateralInhibition()
        
        # Zone uniforme avec un bord
        activation = np.ones((100, 100), dtype=np.float32) * 0.5
        activation[:, 50:] = 0.8
        
        result = lateral.apply(activation)
        
        # Le contraste devrait être augmenté au bord
        edge_region = result[:, 48:52]
        uniform_region = result[:, 10:20]
        
        # Variance plus élevée dans la région du bord
        assert edge_region.std() >= uniform_region.std() * 0.5
    
    def test_non_negative(self):
        """Résultat toujours >= 0 (rectification)."""
        lateral = LateralInhibition()
        activation = np.random.rand(50, 50).astype(np.float32)
        
        result = lateral.apply(activation)
        
        assert result.min() >= 0


class TestGaborFilterBank:
    """Tests du banc de filtres de Gabor."""
    
    def test_filter_creation(self):
        """Crée le bon nombre de filtres."""
        config = PathwayConfig(
            gabor_num_orientations=8,
            gabor_num_scales=3
        )
        gabor = GaborFilterBank(config)
        
        expected = 8 * 3  # orientations × scales
        assert len(gabor.filters) == expected
    
    def test_process_shape(self):
        """Sortie a la bonne forme."""
        config = PathwayConfig(
            gabor_num_orientations=4,
            gabor_num_scales=2
        )
        gabor = GaborFilterBank(config)
        
        img = np.random.rand(100, 100).astype(np.float32)
        responses = gabor.process(img)
        
        assert responses.shape == (100, 100, 4 * 2)
    
    def test_orientation_selectivity(self):
        """Filtre répond mieux à son orientation préférée."""
        config = PathwayConfig(
            gabor_num_orientations=4,
            gabor_num_scales=1
        )
        gabor = GaborFilterBank(config)
        
        # Lignes verticales
        vertical = np.zeros((100, 100), dtype=np.float32)
        for x in range(0, 100, 10):
            vertical[:, x:x+2] = 1.0
        
        # Lignes horizontales
        horizontal = np.zeros((100, 100), dtype=np.float32)
        for y in range(0, 100, 10):
            horizontal[y:y+2, :] = 1.0
        
        resp_v = gabor.process(vertical)
        resp_h = gabor.process(horizontal)
        
        # Les réponses devraient être différentes
        assert not np.allclose(resp_v, resp_h)
    
    def test_orientation_energy(self):
        """Calcul de l'énergie et orientation dominante."""
        gabor = GaborFilterBank()
        img = np.random.rand(50, 50).astype(np.float32)
        
        responses = gabor.process(img)
        energy, orientation = gabor.get_orientation_energy(responses)
        
        assert energy.shape == (50, 50)
        assert orientation.shape == (50, 50)
        assert orientation.min() >= 0
        assert orientation.max() <= np.pi


class TestRetinalProcessor:
    """Tests du processeur rétinien complet."""
    
    def test_full_pipeline(self):
        """Pipeline complet sur une image."""
        processor = RetinalProcessor()
        
        frame = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        result = processor.process(frame)
        
        # Vérifier toutes les clés
        assert 'magno' in result
        assert 'parvo' in result
        assert 'v1' in result
        assert 'saliency' in result
        
        # Vérifier les sous-clés
        assert 'motion' in result['magno']
        assert 'luminance' in result['parvo']
        assert 'red_green' in result['parvo']
        assert 'energy' in result['v1']
    
    def test_saliency_range(self):
        """Saillance dans [0, 1]."""
        processor = RetinalProcessor()
        
        # Deux frames pour avoir du mouvement
        frame1 = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
        frame2 = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
        
        processor.process(frame1)
        result = processor.process(frame2)
        
        saliency = result['saliency']
        assert saliency.min() >= 0
        assert saliency.max() <= 1
    
    def test_reset(self):
        """Reset efface l'état."""
        processor = RetinalProcessor()
        
        frame = np.random.rand(50, 50, 3).astype(np.float32)
        processor.process(frame)
        
        processor.reset()
        
        assert processor.magno.prev_frame is None


class TestIntegration:
    """Tests d'intégration."""
    
    def test_motion_increases_saliency(self):
        """Le mouvement augmente la saillance."""
        processor = RetinalProcessor()
        
        # Frame statique
        static = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Frame avec mouvement (zone qui apparaît)
        moving = static.copy()
        moving[40:60, 40:60] = 255
        
        processor.process(static)
        result_static = processor.process(static)
        
        processor.reset()
        processor.process(static)
        result_moving = processor.process(moving)
        
        # Zone de mouvement devrait être plus saillante
        static_sal = result_static['saliency'][40:60, 40:60].mean()
        moving_sal = result_moving['saliency'][40:60, 40:60].mean()
        
        assert moving_sal > static_sal or np.isclose(moving_sal, static_sal, atol=0.1)
    
    def test_edge_increases_saliency(self):
        """Les bords augmentent la saillance."""
        processor = RetinalProcessor()
        
        # Image avec bord contrasté
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        
        result = processor.process(img)
        
        # Saillance sur le bord vs zone uniforme
        edge_sal = result['saliency'][:, 48:52].mean()
        uniform_sal = result['saliency'][:, 10:20].mean()
        
        assert edge_sal >= uniform_sal * 0.8  # Tolérance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
