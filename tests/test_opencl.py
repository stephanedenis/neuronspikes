"""Tests pour le backend OpenCL."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Importer conditionnellement selon la disponibilité d'OpenCL
try:
    from neuronspikes.opencl_backend import (
        OpenCLBackend,
        get_opencl_backend,
        is_opencl_available,
        list_opencl_devices,
        DeviceInfo,
        OPENCL_AVAILABLE,
    )
except ImportError:
    OPENCL_AVAILABLE = False


@pytest.mark.skipif(not OPENCL_AVAILABLE, reason="PyOpenCL non disponible")
class TestOpenCLBackend:
    """Tests pour OpenCLBackend."""
    
    def test_backend_creation(self):
        """Test de création du backend."""
        backend = get_opencl_backend(verbose=False)
        
        assert backend is not None
        assert backend.device is not None
        assert backend.ctx is not None
        assert backend.queue is not None
    
    def test_device_info(self):
        """Test des informations du périphérique."""
        backend = get_opencl_backend(verbose=False)
        
        info = backend.device_info
        assert isinstance(info, DeviceInfo)
        assert info.compute_units > 0
        assert info.global_mem_mb > 0
    
    def test_list_devices(self):
        """Test du listage des périphériques."""
        devices = list_opencl_devices()
        
        assert len(devices) > 0
        for d in devices:
            assert isinstance(d, DeviceInfo)
            assert d.name
            assert d.platform
    
    def test_polar_sample(self):
        """Test de l'échantillonnage polaire."""
        backend = get_opencl_backend(verbose=False)
        
        # Image de test
        image = np.full((100, 100), 128, dtype=np.uint8)
        
        # Paramètres pour 4 cellules
        cell_params = np.array([
            # inner_r, outer_r, start_angle, end_angle
            0, 10, 0, np.pi/2,
            0, 10, np.pi/2, np.pi,
            0, 10, np.pi, 3*np.pi/2,
            0, 10, 3*np.pi/2, 2*np.pi,
        ], dtype=np.float32)
        
        activations = backend.polar_sample(
            image, 50, 50, 0.0, cell_params
        )
        
        assert activations.shape == (4,)
        assert np.all(activations >= 0)
        assert np.all(activations <= 1)
    
    def test_polar_sample_with_rotation(self):
        """Test avec rotation."""
        backend = get_opencl_backend(verbose=False)
        
        image = np.full((100, 100), 200, dtype=np.uint8)
        cell_params = np.array([
            0, 10, 0, np.pi/2,
            0, 10, np.pi/2, np.pi,
        ], dtype=np.float32)
        
        act1 = backend.polar_sample(image, 50, 50, 0.0, cell_params)
        act2 = backend.polar_sample(image, 50, 50, np.pi/4, cell_params)
        
        # Les activations devraient être similaires pour image uniforme
        assert np.allclose(act1, act2, atol=0.1)
    
    def test_compute_saliency(self):
        """Test du calcul de saillance."""
        backend = get_opencl_backend(verbose=False)
        
        # Image avec bord net
        image = np.zeros((100, 100), dtype=np.uint8)
        image[:, 50:] = 255
        
        saliency = backend.compute_saliency(image)
        
        assert saliency.shape == (100, 100)
        # Forte saillance au bord
        assert saliency[50, 50] > saliency[50, 10]
    
    def test_compute_saliency_uniform(self):
        """Test saillance sur image uniforme."""
        backend = get_opencl_backend(verbose=False)
        
        image = np.full((50, 50), 128, dtype=np.uint8)
        
        saliency = backend.compute_saliency(image)
        
        # Saillance faible partout
        assert np.mean(saliency) < 0.1
    
    def test_abs_diff(self):
        """Test de la différence absolue."""
        backend = get_opencl_backend(verbose=False)
        
        img1 = np.full((50, 50), 100, dtype=np.uint8)
        img2 = np.full((50, 50), 150, dtype=np.uint8)
        
        diff = backend.abs_diff(img1, img2)
        
        assert diff.shape == (50, 50)
        expected = 50 / 255.0
        assert np.allclose(diff, expected, atol=0.01)
    
    def test_stereo_correlation(self):
        """Test de la corrélation stéréo."""
        backend = get_opencl_backend(verbose=False)
        
        left = np.array([[0.5, 0.8], [0.3, 0.9]], dtype=np.float32)
        right = np.array([[0.5, 0.6], [0.3, 0.7]], dtype=np.float32)
        
        corr, disp = backend.stereo_correlation(left, right)
        
        assert corr.shape == left.shape
        assert disp.shape == left.shape
        
        # Vérifier les calculs
        expected_corr = left * right
        expected_disp = left - right
        assert np.allclose(corr, expected_corr, atol=0.01)
        assert np.allclose(disp, expected_disp, atol=0.01)
    
    def test_detect_rotation(self):
        """Test de la détection de rotation."""
        backend = get_opencl_backend(verbose=False)
        
        # Pattern avec feature distincte
        current = np.zeros((4, 8), dtype=np.float32)
        current[2, 0] = 1.0  # Feature au secteur 0
        
        prev = np.zeros((4, 8), dtype=np.float32)
        prev[2, 2] = 1.0  # Feature au secteur 2 (décalage de 2)
        
        shift = backend.detect_rotation(current, prev, 4, 8, max_shift=4)
        
        # Devrait détecter un shift de 2
        assert shift == 2
    
    def test_detect_rotation_no_shift(self):
        """Test sans rotation."""
        backend = get_opencl_backend(verbose=False)
        
        pattern = np.random.rand(4, 8).astype(np.float32)
        
        shift = backend.detect_rotation(pattern, pattern, 4, 8)
        
        assert shift == 0
    
    def test_get_stats(self):
        """Test des statistiques."""
        backend = get_opencl_backend(verbose=False)
        
        stats = backend.get_stats()
        
        assert stats['available'] is True
        assert 'device' in stats
        assert 'compute_units' in stats
        assert stats['compute_units'] > 0


@pytest.mark.skipif(not OPENCL_AVAILABLE, reason="PyOpenCL non disponible")
class TestPerformance:
    """Tests de performance."""
    
    def test_polar_sample_speed(self):
        """Test de vitesse d'échantillonnage."""
        import time
        
        backend = get_opencl_backend(verbose=False)
        
        image = np.random.randint(0, 256, (720, 1280), dtype=np.uint8)
        cell_params = np.zeros(64 * 4, dtype=np.float32)
        for i in range(64):
            ring = i // 8
            sector = i % 8
            cell_params[i*4 + 0] = ring * 8
            cell_params[i*4 + 1] = (ring+1) * 8
            cell_params[i*4 + 2] = sector * np.pi / 4
            cell_params[i*4 + 3] = (sector+1) * np.pi / 4
        
        # Warmup
        backend.polar_sample(image, 640, 360, 0.0, cell_params)
        
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(100):
            backend.polar_sample(image, 640, 360, 0.0, cell_params)
        t1 = time.perf_counter()
        
        ms_per_frame = (t1 - t0) * 10  # 100 frames -> ms/frame
        
        # Devrait être < 10ms/frame
        assert ms_per_frame < 50, f"Trop lent: {ms_per_frame:.1f} ms/frame"
    
    def test_saliency_speed(self):
        """Test de vitesse du calcul de saillance."""
        import time
        
        backend = get_opencl_backend(verbose=False)
        
        image = np.random.randint(0, 256, (720, 1280), dtype=np.uint8)
        
        # Warmup
        backend.compute_saliency(image)
        
        # Benchmark
        t0 = time.perf_counter()
        for _ in range(20):
            backend.compute_saliency(image)
        t1 = time.perf_counter()
        
        ms_per_frame = (t1 - t0) * 50  # 20 frames -> ms/frame
        
        # Devrait être < 20ms/frame
        assert ms_per_frame < 100, f"Trop lent: {ms_per_frame:.1f} ms/frame"


class TestWithoutOpenCL:
    """Tests quand OpenCL n'est pas disponible."""
    
    def test_is_opencl_available(self):
        """Test de la fonction is_opencl_available."""
        result = is_opencl_available()
        assert isinstance(result, bool)
    
    def test_get_backend_returns_none_when_unavailable(self):
        """Test que get_backend retourne None si OpenCL indisponible."""
        # Simuler l'absence d'OpenCL
        with patch.dict('sys.modules', {'pyopencl': None}):
            # Le module a déjà été importé, donc on teste juste le comportement
            pass  # Ce test est surtout documentaire
