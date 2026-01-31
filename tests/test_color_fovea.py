"""Tests pour le module color_fovea - Vision couleur et détection de mouvement."""

import math
import numpy as np
import pytest

from neuronspikes.color_fovea import (
    ColorChannel,
    ColorFoveaConfig,
    MotionVector,
    TrackedObject,
    ColorFovea,
    ObjectTracker,
)


# ============================================================================
# Tests ColorChannel
# ============================================================================

class TestColorChannel:
    """Tests pour l'enum ColorChannel."""

    def test_enum_values(self):
        """Vérifie les valeurs de l'enum."""
        assert ColorChannel.LUMA is not None
        assert ColorChannel.CHROMA_U is not None
        assert ColorChannel.CHROMA_V is not None
        assert ColorChannel.ALPHA is not None
        assert ColorChannel.MOTION is not None
        assert ColorChannel.MOTION_DIR is not None

    def test_enum_membership(self):
        """Vérifie l'appartenance à l'enum."""
        assert ColorChannel.LUMA in ColorChannel
        assert ColorChannel.CHROMA_U in ColorChannel
        assert ColorChannel.ALPHA in ColorChannel


# ============================================================================
# Tests ColorFoveaConfig
# ============================================================================

class TestColorFoveaConfig:
    """Tests pour ColorFoveaConfig."""

    def test_default_config(self):
        """Vérifie les valeurs par défaut."""
        config = ColorFoveaConfig()
        assert config.use_color is True
        assert config.motion_history == 5
        assert config.motion_threshold == 2.0
        assert config.alpha_padding == 0

    def test_custom_config(self):
        """Vérifie la configuration personnalisée."""
        config = ColorFoveaConfig(
            num_rings=16,
            num_sectors=48,
            fovea_radius=128,
            max_radius=512,
            use_color=False,
            motion_history=10,
            motion_threshold=3.5,
            alpha_padding=10,
        )
        assert config.num_rings == 16
        assert config.num_sectors == 48
        assert config.use_color is False
        assert config.motion_history == 10
        assert config.motion_threshold == 3.5
        assert config.alpha_padding == 10

    def test_inherits_fovea_config(self):
        """Vérifie l'héritage de FoveaConfig."""
        config = ColorFoveaConfig()
        # Attributs hérités
        assert hasattr(config, 'num_rings')
        assert hasattr(config, 'num_sectors')
        assert hasattr(config, 'fovea_radius')
        assert hasattr(config, 'max_radius')


# ============================================================================
# Tests MotionVector
# ============================================================================

class TestMotionVector:
    """Tests pour MotionVector."""

    def test_creation_default(self):
        """Vérifie la création avec valeurs par défaut."""
        mv = MotionVector()
        assert mv.dx == 0.0
        assert mv.dy == 0.0
        assert mv.magnitude == 0.0
        assert mv.is_moving is False

    def test_creation_with_values(self):
        """Vérifie la création avec valeurs."""
        mv = MotionVector(dx=3.0, dy=4.0)
        assert mv.dx == 3.0
        assert mv.dy == 4.0
        assert mv.magnitude == pytest.approx(5.0)
        assert mv.is_moving is True

    def test_direction(self):
        """Vérifie le calcul de direction."""
        mv = MotionVector(dx=1.0, dy=0.0)
        assert mv.direction == pytest.approx(0.0)  # Vers la droite
        
        mv = MotionVector(dx=0.0, dy=1.0)
        assert mv.direction == pytest.approx(math.pi / 2)  # Vers le bas
        
        mv = MotionVector(dx=-1.0, dy=0.0)
        assert mv.direction == pytest.approx(math.pi)  # Vers la gauche

    def test_is_moving_threshold(self):
        """Vérifie le seuil de mouvement."""
        mv = MotionVector(dx=0.3, dy=0.3)
        assert mv.magnitude < 0.5
        assert mv.is_moving is False
        
        mv = MotionVector(dx=0.5, dy=0.5)
        assert mv.magnitude > 0.5
        assert mv.is_moving is True


# ============================================================================
# Tests TrackedObject
# ============================================================================

class TestTrackedObject:
    """Tests pour TrackedObject."""

    def test_creation(self):
        """Vérifie la création d'un objet suivi."""
        obj = TrackedObject(
            id=1,
            x=50.0,
            y=50.0,
            vx=3.0,
            vy=2.0,
            size=25.0,
        )
        assert obj.id == 1
        assert obj.x == 50.0
        assert obj.y == 50.0
        assert obj.vx == 3.0
        assert obj.vy == 2.0
        assert obj.size == 25.0

    def test_default_values(self):
        """Vérifie les valeurs par défaut."""
        obj = TrackedObject(id=1, x=50.0, y=50.0)
        assert obj.vx == 0.0
        assert obj.vy == 0.0
        assert obj.size == 20.0
        assert obj.age == 0
        assert obj.confidence == 0.5

    def test_predict_position(self):
        """Vérifie la prédiction de position."""
        obj = TrackedObject(id=1, x=50.0, y=50.0, vx=10.0, vy=5.0)
        
        # Prédiction à t+1
        px, py = obj.predict_position(1.0)
        assert px == pytest.approx(60.0)
        assert py == pytest.approx(55.0)
        
        # Prédiction à t+2
        px, py = obj.predict_position(2.0)
        assert px == pytest.approx(70.0)
        assert py == pytest.approx(60.0)

    def test_update(self):
        """Vérifie la mise à jour d'un objet."""
        obj = TrackedObject(id=1, x=50.0, y=50.0)
        initial_confidence = obj.confidence
        
        obj.update(60.0, 55.0, frame=1)
        
        assert obj.x == 60.0
        assert obj.y == 55.0
        assert obj.age == 1
        assert obj.last_seen == 1
        assert obj.confidence > initial_confidence

    def test_decay(self):
        """Vérifie la décroissance de confiance."""
        obj = TrackedObject(id=1, x=50.0, y=50.0, vx=5.0, vy=3.0)
        obj.confidence = 0.9
        obj.last_seen = 0
        
        obj.decay(frame=5)
        
        assert obj.confidence < 0.9
        # Position mise à jour par prédiction
        assert obj.x == pytest.approx(50.0 + 5.0 * 5)
        assert obj.y == pytest.approx(50.0 + 3.0 * 5)


# ============================================================================
# Tests ColorFovea
# ============================================================================

class TestColorFovea:
    """Tests pour ColorFovea."""

    def test_creation(self):
        """Vérifie la création d'une fovéa couleur."""
        config = ColorFoveaConfig(num_rings=8, num_sectors=24)
        fovea = ColorFovea(config)
        assert fovea.color_config == config

    def test_inherits_fovea(self):
        """Vérifie l'héritage de Fovea."""
        config = ColorFoveaConfig()
        fovea = ColorFovea(config)
        
        # Méthodes héritées
        assert hasattr(fovea, 'sample')
        assert hasattr(fovea, 'cells')
        assert hasattr(fovea, 'gaze')

    def test_color_channels(self):
        """Vérifie les canaux de couleur."""
        config = ColorFoveaConfig(num_rings=8, num_sectors=24)
        fovea = ColorFovea(config)
        
        assert fovea.luma.shape == (8, 24)
        assert fovea.chroma_u.shape == (8, 24)
        assert fovea.chroma_v.shape == (8, 24)
        assert fovea.alpha.shape == (8, 24)
        assert fovea.motion_magnitude.shape == (8, 24)
        assert fovea.motion_direction.shape == (8, 24)

    def test_sample_color_bgr(self):
        """Vérifie l'échantillonnage couleur BGR."""
        config = ColorFoveaConfig(num_rings=8, num_sectors=24)
        fovea = ColorFovea(config)
        
        # Image test BGR (comme OpenCV)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        fovea.set_gaze(50, 50)
        
        result = fovea.sample_color(image)
        
        assert 'luma' in result
        assert 'chroma_u' in result
        assert 'chroma_v' in result
        assert 'alpha' in result
        assert 'motion_mag' in result
        assert 'motion_dir' in result
        
        assert result['luma'].shape == (8, 24)

    def test_sample_color_grayscale(self):
        """Vérifie l'échantillonnage d'une image en niveaux de gris."""
        config = ColorFoveaConfig(num_rings=8, num_sectors=24)
        fovea = ColorFovea(config)
        
        # Image grayscale
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        fovea.set_gaze(50, 50)
        
        result = fovea.sample_color(image)
        
        assert result['luma'].shape == (8, 24)
        # Pour grayscale, chroma devrait être 0
        assert np.allclose(result['chroma_u'], 0.0, atol=1.0)
        assert np.allclose(result['chroma_v'], 0.0, atol=1.0)

    def test_alpha_out_of_bounds(self):
        """Vérifie que alpha<1 pour les pixels hors de l'image."""
        config = ColorFoveaConfig(
            num_rings=8, 
            num_sectors=24,
            fovea_radius=10,
            max_radius=60,
        )
        fovea = ColorFovea(config)
        
        # Petite image avec regard près du bord
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        fovea.set_gaze(5, 5)  # Près du coin, certains rayons sortiront
        
        result = fovea.sample_color(image)
        
        # Vérifier qu'il y a des alpha < 1 (pixels hors limites)
        assert np.any(result['alpha'] < 1.0)

    def test_motion_detection(self):
        """Vérifie la détection de mouvement."""
        config = ColorFoveaConfig(
            num_rings=8, 
            num_sectors=24,
            motion_history=3,
        )
        fovea = ColorFovea(config)
        
        # Première frame
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image1[40:50, 40:50, :] = 255
        fovea.set_gaze(50, 50)
        
        result1 = fovea.sample_color(image1)
        
        # Deuxième frame avec objet déplacé
        image2 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2[42:52, 45:55, :] = 255
        
        result2 = fovea.sample_color(image2)
        
        # Le mouvement devrait être détecté après la 2ème frame
        assert result2['motion_mag'] is not None

    def test_get_motion_vectors(self):
        """Vérifie la récupération des vecteurs de mouvement."""
        config = ColorFoveaConfig()
        fovea = ColorFovea(config)
        
        # Créer du mouvement
        image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        image1[40:50, 40:50] = 200
        fovea.set_gaze(50, 50)
        fovea.sample_color(image1)
        
        image2 = np.zeros((100, 100, 3), dtype=np.uint8)
        image2[42:52, 45:55] = 200
        fovea.sample_color(image2)
        
        vectors = fovea.get_motion_vectors()
        assert isinstance(vectors, list)

    def test_get_color_signature(self):
        """Vérifie l'extraction de signature couleur."""
        config = ColorFoveaConfig()
        fovea = ColorFovea(config)
        
        # Image avec région colorée
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[30:70, 30:70, 2] = 200  # Rouge en BGR
        fovea.set_gaze(50, 50)
        
        fovea.sample_color(image)
        sig = fovea.get_color_signature()
        
        assert sig is not None
        assert len(sig) == 3  # (Y, U, V)

    def test_get_dominant_motion_none(self):
        """Vérifie qu'il n'y a pas de mouvement dominant sans mouvement."""
        config = ColorFoveaConfig()
        fovea = ColorFovea(config)
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        fovea.set_gaze(50, 50)
        fovea.sample_color(image)
        
        dominant = fovea.get_dominant_motion()
        assert dominant is None


# ============================================================================
# Tests ObjectTracker
# ============================================================================

class TestObjectTracker:
    """Tests pour ObjectTracker."""

    def test_creation(self):
        """Vérifie la création du tracker."""
        tracker = ObjectTracker()
        assert tracker.max_objects == 10
        assert tracker.min_confidence == 0.3
        assert tracker.merge_distance == 30.0

    def test_custom_creation(self):
        """Vérifie la création avec paramètres personnalisés."""
        tracker = ObjectTracker(
            max_objects=5,
            min_confidence=0.5,
            merge_distance=50.0
        )
        assert tracker.max_objects == 5
        assert tracker.min_confidence == 0.5
        assert tracker.merge_distance == 50.0

    def test_objects_initially_empty(self):
        """Vérifie que la liste d'objets est vide initialement."""
        tracker = ObjectTracker()
        assert len(tracker.objects) == 0


# ============================================================================
# Tests d'intégration
# ============================================================================

class TestColorFoveaIntegration:
    """Tests d'intégration pour ColorFovea."""

    def test_full_pipeline(self):
        """Teste le pipeline complet d'analyse couleur."""
        config = ColorFoveaConfig(
            num_rings=8,
            num_sectors=24,
            fovea_radius=30,
            max_radius=80,
            motion_history=5,
        )
        fovea = ColorFovea(config)
        
        # Simuler une séquence vidéo avec objet en mouvement
        for i in range(5):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            # Objet qui se déplace
            x = 50 + i * 5
            y = 50 + i * 3
            frame[y:y+30, x:x+30, 2] = 200  # Rouge
            frame[y:y+30, x:x+30, 1] = 50   # Peu de vert
            
            fovea.set_gaze(100, 100)
            result = fovea.sample_color(frame)
            
            assert result['luma'].shape == (8, 24)
            assert result['alpha'].shape == (8, 24)

    def test_stereo_color_sampling(self):
        """Teste l'échantillonnage couleur stéréo."""
        config = ColorFoveaConfig(num_rings=8, num_sectors=24)
        fovea_left = ColorFovea(config)
        fovea_right = ColorFovea(config)
        
        # Images stéréo simulées
        left_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        right_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        
        fovea_left.set_gaze(100, 100)
        fovea_right.set_gaze(105, 100)  # Légère disparité
        
        result_left = fovea_left.sample_color(left_image)
        result_right = fovea_right.sample_color(right_image)
        
        assert result_left['luma'].shape == result_right['luma'].shape

    def test_multiple_frames_motion(self):
        """Teste la détection de mouvement sur plusieurs frames."""
        config = ColorFoveaConfig(
            num_rings=8, 
            num_sectors=24,
            motion_history=5,
            motion_threshold=1.0,
        )
        fovea = ColorFovea(config)
        fovea.set_gaze(100, 100)
        
        # Générer une séquence avec mouvement
        for i in range(10):
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
            # Objet en mouvement rapide
            x = 50 + i * 10
            y = 50 + i * 5
            if x < 190 and y < 190:
                frame[y:y+20, x:x+20] = [100, 150, 200]
            
            result = fovea.sample_color(frame)
            
            # À partir de la 2ème frame, le mouvement devrait être calculé
            if i > 0:
                assert result['motion_mag'] is not None


# ============================================================================
# Tests de performance
# ============================================================================

class TestColorFoveaPerformance:
    """Tests de performance pour ColorFovea."""

    def test_sampling_speed(self):
        """Vérifie la vitesse d'échantillonnage."""
        import time
        
        config = ColorFoveaConfig(num_rings=8, num_sectors=24)
        fovea = ColorFovea(config)
        
        image = np.random.randint(0, 256, (640, 480, 3), dtype=np.uint8)
        fovea.set_gaze(320, 240)
        
        # Warmup
        for _ in range(10):
            fovea.sample_color(image)
        
        # Mesure
        start = time.perf_counter()
        n_samples = 100
        for _ in range(n_samples):
            fovea.sample_color(image)
        elapsed = time.perf_counter() - start
        
        fps = n_samples / elapsed
        # Devrait être > 50 FPS pour une image 640x480
        assert fps > 30, f"Trop lent: {fps:.1f} FPS"


# ============================================================================
# Tests de robustesse
# ============================================================================

class TestColorFoveaRobustness:
    """Tests de robustesse pour ColorFovea."""

    def test_empty_image(self):
        """Vérifie le comportement avec image noire."""
        config = ColorFoveaConfig()
        fovea = ColorFovea(config)
        
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        fovea.set_gaze(50, 50)
        result = fovea.sample_color(image)
        
        assert np.all(result['luma'] == 0)

    def test_saturated_image(self):
        """Vérifie le comportement avec image blanche."""
        config = ColorFoveaConfig(
            num_rings=8,
            num_sectors=16,
            fovea_radius=10,
            max_radius=40,  # Assez petit pour rester dans l'image
        )
        fovea = ColorFovea(config)
        
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        fovea.set_gaze(50, 50)
        result = fovea.sample_color(image)
        
        # Les cellules avec alpha > 0 devraient avoir luma proche de 255
        valid_mask = result['alpha'] > 0.5
        if np.any(valid_mask):
            assert np.all(result['luma'][valid_mask] > 200)

    def test_gaze_at_corner(self):
        """Vérifie le comportement avec regard au coin."""
        config = ColorFoveaConfig(max_radius=50)
        fovea = ColorFovea(config)
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        corners = [(0, 0), (0, 99), (99, 0), (99, 99)]
        for corner in corners:
            fovea.set_gaze(*corner)
            result = fovea.sample_color(image)
            # Devrait avoir des pixels hors limites
            assert np.sum(result['alpha'] < 1.0) > 0

    def test_very_small_image(self):
        """Vérifie le comportement avec très petite image."""
        config = ColorFoveaConfig(
            num_rings=4, 
            num_sectors=8, 
            fovea_radius=5, 
            max_radius=10
        )
        fovea = ColorFovea(config)
        
        image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        fovea.set_gaze(10, 10)
        result = fovea.sample_color(image)
        
        assert result['luma'].shape == (4, 8)

    def test_non_square_image(self):
        """Vérifie le comportement avec image non carrée."""
        config = ColorFoveaConfig()
        fovea = ColorFovea(config)
        
        # Image rectangulaire
        image = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        fovea.set_gaze(50, 100)
        result = fovea.sample_color(image)
        
        assert result is not None
        assert 'luma' in result
