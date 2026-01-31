"""Tests pour le module LUT (lookup tables)."""

import numpy as np
import pytest

from neuronspikes.lut import (
    BIT_REVERSAL_LUT,
    INTENSITY_TO_SPIKES,
    generate_bit_reversal_lut,
    generate_temporal_pattern,
    intensity_to_spike_train,
    frame_to_spike_trains,
)


class TestBitReversalLUT:
    """Tests pour la LUT d'inversion de bits."""
    
    def test_lut_shape(self):
        """La LUT doit avoir 256 entrées."""
        assert len(BIT_REVERSAL_LUT) == 256
        assert BIT_REVERSAL_LUT.dtype == np.uint8
    
    def test_lut_zero(self):
        """L'inversion de 0 doit être 0."""
        assert BIT_REVERSAL_LUT[0] == 0
    
    def test_lut_one(self):
        """00000001 inversé donne 10000000 = 128."""
        assert BIT_REVERSAL_LUT[1] == 128
    
    def test_lut_two(self):
        """00000010 inversé donne 01000000 = 64."""
        assert BIT_REVERSAL_LUT[2] == 64
    
    def test_lut_128(self):
        """10000000 inversé donne 00000001 = 1."""
        assert BIT_REVERSAL_LUT[128] == 1
    
    def test_lut_255(self):
        """11111111 inversé donne 11111111 = 255."""
        assert BIT_REVERSAL_LUT[255] == 255
    
    def test_lut_bijective(self):
        """La LUT doit être une bijection (permutation)."""
        # Chaque valeur doit apparaître exactement une fois
        unique_values = np.unique(BIT_REVERSAL_LUT)
        assert len(unique_values) == 256
    
    def test_lut_involutive(self):
        """Appliquer l'inversion deux fois doit redonner la valeur originale."""
        for i in range(256):
            reversed_once = BIT_REVERSAL_LUT[i]
            reversed_twice = BIT_REVERSAL_LUT[reversed_once]
            assert reversed_twice == i, f"Failed for {i}"
    
    def test_lut_deterministic(self):
        """La génération doit être déterministe."""
        lut1 = generate_bit_reversal_lut()
        lut2 = generate_bit_reversal_lut()
        assert np.array_equal(lut1, lut2)


class TestTemporalPattern:
    """Tests pour la génération de patterns temporels."""
    
    def test_zero_intensity(self):
        """Intensité 0 = aucune impulsion."""
        pattern = generate_temporal_pattern(0, BIT_REVERSAL_LUT)
        assert pattern.sum() == 0
    
    def test_max_intensity(self):
        """Intensité 255 = 255 impulsions."""
        pattern = generate_temporal_pattern(255, BIT_REVERSAL_LUT)
        assert pattern.sum() == 255
    
    def test_intensity_equals_spike_count(self):
        """Le nombre d'impulsions doit égaler l'intensité."""
        for intensity in [1, 10, 50, 100, 200, 255]:
            pattern = generate_temporal_pattern(intensity, BIT_REVERSAL_LUT)
            assert pattern.sum() == intensity
    
    def test_pattern_shape(self):
        """Le pattern doit avoir 256 slots."""
        pattern = generate_temporal_pattern(100, BIT_REVERSAL_LUT)
        assert len(pattern) == 256
        assert pattern.dtype == np.bool_


class TestIntensityToSpikesMatrix:
    """Tests pour la matrice pré-calculée."""
    
    def test_matrix_shape(self):
        """La matrice doit être 256x256."""
        assert INTENSITY_TO_SPIKES.shape == (256, 256)
    
    def test_matrix_dtype(self):
        """La matrice doit être de type bool."""
        assert INTENSITY_TO_SPIKES.dtype == np.bool_
    
    def test_row_sum_equals_intensity(self):
        """La somme de chaque ligne doit égaler l'intensité."""
        for intensity in range(256):
            row_sum = INTENSITY_TO_SPIKES[intensity].sum()
            assert row_sum == intensity


class TestConvenienceFunctions:
    """Tests pour les fonctions utilitaires."""
    
    def test_intensity_to_spike_train(self):
        """Vérifier la fonction de conversion simple."""
        train = intensity_to_spike_train(100)
        assert len(train) == 256
        assert train.sum() == 100
    
    def test_frame_to_spike_trains_shape(self):
        """Vérifier la forme de sortie pour une frame."""
        frame = np.zeros((64, 64), dtype=np.uint8)
        frame[0, 0] = 100
        frame[10, 10] = 200
        
        trains = frame_to_spike_trains(frame)
        
        assert trains.shape == (64, 64, 256)
        assert trains[0, 0].sum() == 100
        assert trains[10, 10].sum() == 200
        assert trains[5, 5].sum() == 0  # Pixel noir
    
    def test_frame_deterministic(self):
        """La conversion doit être déterministe."""
        frame = np.arange(256, dtype=np.uint8).reshape(16, 16)
        
        trains1 = frame_to_spike_trains(frame)
        trains2 = frame_to_spike_trains(frame)
        
        assert np.array_equal(trains1, trains2)


class TestTemporalDistribution:
    """Tests pour vérifier la distribution temporelle uniforme."""
    
    def test_low_intensity_middle_slot(self):
        """Intensité 1 doit activer le slot 128 (milieu)."""
        train = intensity_to_spike_train(1)
        assert train[128] == True
        assert train.sum() == 1
    
    def test_distribution_uniformity(self):
        """Vérifier que les slots sont bien distribués."""
        # Pour intensité maximale, tous les slots sauf 0 sont utilisés
        train_255 = intensity_to_spike_train(255)
        
        # Diviser en 4 quartiles
        q1 = train_255[0:64].sum()
        q2 = train_255[64:128].sum()
        q3 = train_255[128:192].sum()
        q4 = train_255[192:256].sum()
        
        # La distribution devrait être relativement uniforme
        # (pas exactement égale à cause de la nature du bit-reversal)
        total = q1 + q2 + q3 + q4
        assert total == 255
        
        # Chaque quartile devrait avoir environ 64 impulsions (±16)
        for q in [q1, q2, q3, q4]:
            assert 48 <= q <= 80, f"Quartile {q} hors limites"
