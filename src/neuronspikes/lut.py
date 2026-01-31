"""
Lookup Tables pour NeuronSpikes.

Ce module contient les LUT déterministes utilisées pour convertir
les intensités lumineuses en trains d'impulsions temporellement distribués.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_bit_reversal_lut() -> NDArray[np.uint8]:
    """Génère la LUT de permutation par inversion de bits.
    
    Cette LUT permet une distribution temporelle homogène des impulsions.
    Pour chaque intensité i (0-255), on obtient l'index temporel où
    l'impulsion doit être émise.
    
    Principe:
        - Intensité 1:   00000001 → 10000000 → slot 128 (milieu)
        - Intensité 2:   00000010 → 01000000 → slot 64  (quart)
        - Intensité 128: 10000000 → 00000001 → slot 1   (début)
    
    Returns:
        NDArray[np.uint8]: LUT de 256 valeurs (index temporels)
    """
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Inverser l'ordre des 8 bits
        reversed_bits = int('{:08b}'.format(i)[::-1], 2)
        lut[i] = reversed_bits
    return lut


def generate_temporal_pattern(intensity: int, lut: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Génère le pattern temporel d'impulsions pour une intensité donnée.
    
    Pour une intensité I, génère un train de I impulsions réparties
    uniformément sur 256 slots temporels.
    
    Args:
        intensity: Valeur d'intensité 0-255
        lut: LUT de permutation bit-reversal
    
    Returns:
        NDArray[np.bool_]: Tableau de 256 booléens indiquant les impulsions
    """
    pattern = np.zeros(256, dtype=np.bool_)
    
    # Pour chaque niveau d'intensité de 1 à intensity, 
    # marquer le slot correspondant via la LUT
    for level in range(1, intensity + 1):
        slot = lut[level]
        pattern[slot] = True
    
    return pattern


def generate_intensity_to_spikes_matrix(lut: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Génère la matrice complète intensité → pattern d'impulsions.
    
    Pré-calcule tous les patterns possibles pour éviter les calculs
    en temps réel.
    
    Args:
        lut: LUT de permutation bit-reversal
    
    Returns:
        NDArray[np.bool_]: Matrice 256x256 où [i, t] indique si
                          l'intensité i produit une impulsion au slot t
    """
    matrix = np.zeros((256, 256), dtype=np.bool_)
    
    for intensity in range(256):
        matrix[intensity] = generate_temporal_pattern(intensity, lut)
    
    return matrix


# LUT pré-calculée au chargement du module
BIT_REVERSAL_LUT: NDArray[np.uint8] = generate_bit_reversal_lut()

# Matrice pré-calculée (256 intensités × 256 slots temporels)
INTENSITY_TO_SPIKES: NDArray[np.bool_] = generate_intensity_to_spikes_matrix(BIT_REVERSAL_LUT)


def intensity_to_spike_train(intensity: int) -> NDArray[np.bool_]:
    """Convertit une intensité en train d'impulsions.
    
    Utilise la matrice pré-calculée pour une performance optimale.
    
    Args:
        intensity: Valeur 0-255
        
    Returns:
        NDArray[np.bool_]: Train de 256 impulsions
    """
    return INTENSITY_TO_SPIKES[intensity].copy()


def frame_to_spike_trains(frame: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Convertit une frame d'image en trains d'impulsions.
    
    Args:
        frame: Image monochrome (H, W) en uint8
        
    Returns:
        NDArray[np.bool_]: Trains d'impulsions (H, W, 256)
    """
    h, w = frame.shape
    # Utiliser l'indexation directe dans la matrice pré-calculée
    return INTENSITY_TO_SPIKES[frame.flatten()].reshape(h, w, 256)
