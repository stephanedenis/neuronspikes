"""Voies rétiniennes bio-inspirées.

Implémente les traitements parallèles du système visuel humain:
- Voie Magnocellulaire (M): mouvement, contraste transitoire, réponse rapide
- Voie Parvocellulaire (P): couleur, détails fins, réponse lente
- Voie Koniocellulaire (K): opposition bleu-jaune

Architecture inspirée de:
- Rétine → LGN (thalamus) → V1 (cortex strié)
- Séparation en voie ventrale (QUOI) et dorsale (OÙ/COMMENT)

Références:
- Hubel & Wiesel (1962): Cellules simples et complexes
- Marr (1982): Vision computationnelle
- Itti & Koch (2001): Saillance visuelle
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
import math

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class PathwayConfig:
    """Configuration des voies rétiniennes."""
    
    # Voie Magnocellulaire
    magno_sigma: float = 2.0          # Sigma du flou gaussien (large réceptive field)
    magno_threshold: float = 0.05     # Seuil de détection de mouvement
    magno_temporal_decay: float = 0.7 # Décroissance temporelle
    
    # Voie Parvocellulaire  
    parvo_sigma: float = 1.0          # Sigma fin pour détails
    parvo_color_gain: float = 1.5     # Gain des canaux couleur
    
    # Inhibition latérale (cellules horizontales/amacrines)
    lateral_sigma: float = 3.0        # Sigma du surround
    lateral_strength: float = 0.5     # Force de l'inhibition
    
    # Filtres de Gabor (V1)
    gabor_num_orientations: int = 8   # Nombre d'orientations (0°, 22.5°, 45°, ...)
    gabor_num_scales: int = 3         # Nombre d'échelles spatiales
    gabor_sigma: float = 3.0          # Sigma de l'enveloppe gaussienne
    gabor_wavelength: float = 8.0     # Longueur d'onde de base
    gabor_gamma: float = 0.5          # Ratio d'aspect


class MagnocellularPathway:
    """Voie Magnocellulaire - Mouvement et contraste transitoire.
    
    Caractéristiques biologiques:
    - Grands champs récepteurs
    - Réponse transitoire (sensible au changement)
    - Haute sensibilité au contraste
    - Faible résolution spatiale
    - Traitement rapide (~40ms)
    
    Projette vers la voie dorsale (MT/V5 → Cortex pariétal)
    pour le traitement du mouvement et de l'action.
    """
    
    def __init__(self, config: PathwayConfig = None):
        self.config = config or PathwayConfig()
        self.prev_frame: Optional[np.ndarray] = None
        self.motion_accumulator: Optional[np.ndarray] = None
        
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Traite une frame et retourne la réponse magnocellulaire.
        
        Args:
            frame: Image en niveaux de gris ou couleur
            
        Returns:
            Tuple (motion_energy, direction_map):
            - motion_energy: Énergie de mouvement (0-1)
            - direction_map: Direction du mouvement (angle en radians)
        """
        # Convertir en gris si nécessaire
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        # Normaliser [0, 1]
        gray = gray / 255.0 if gray.max() > 1.0 else gray
        
        # Flou gaussien (grands champs récepteurs)
        if HAS_CV2:
            ksize = int(self.config.magno_sigma * 4) | 1  # Impair
            blurred = cv2.GaussianBlur(gray, (ksize, ksize), self.config.magno_sigma)
        else:
            blurred = self._gaussian_blur_numpy(gray, self.config.magno_sigma)
        
        # Première frame ou changement de taille: réinitialiser
        if self.prev_frame is None or self.prev_frame.shape != blurred.shape:
            self.prev_frame = blurred.copy()
            self.motion_accumulator = np.zeros_like(blurred)
            return np.zeros_like(blurred), np.zeros_like(blurred)
        
        # Différence temporelle (réponse transitoire)
        diff = blurred - self.prev_frame
        
        # Énergie de mouvement (valeur absolue)
        motion_energy = np.abs(diff)
        
        # Seuillage adaptatif
        threshold = self.config.magno_threshold
        motion_energy = np.where(motion_energy > threshold, motion_energy, 0)
        
        # Direction du mouvement (signe de la différence)
        # Positif = apparition (ON), Négatif = disparition (OFF)
        direction = np.sign(diff) * np.pi / 2  # ±90° pour simplifier
        
        # Accumulation temporelle avec décroissance
        decay = self.config.magno_temporal_decay
        self.motion_accumulator = decay * self.motion_accumulator + (1 - decay) * motion_energy
        
        # Mettre à jour la frame précédente
        self.prev_frame = blurred.copy()
        
        return self.motion_accumulator, direction
    
    def _gaussian_blur_numpy(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """Fallback Gaussien en NumPy pur."""
        size = int(sigma * 4) | 1
        x = np.arange(size) - size // 2
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        
        # Séparable: horizontal puis vertical
        result = np.apply_along_axis(lambda row: np.convolve(row, kernel_1d, 'same'), 1, img)
        result = np.apply_along_axis(lambda col: np.convolve(col, kernel_1d, 'same'), 0, result)
        return result
    
    def reset(self):
        """Réinitialise l'état temporel."""
        self.prev_frame = None
        self.motion_accumulator = None


class ParvocellularPathway:
    """Voie Parvocellulaire - Couleur et détails fins.
    
    Caractéristiques biologiques:
    - Petits champs récepteurs (haute résolution)
    - Réponse soutenue (maintenue dans le temps)
    - Sensibilité à la couleur (opposition R-G)
    - Traitement lent (~80ms)
    
    Projette vers la voie ventrale (V4 → Cortex temporal inférieur)
    pour la reconnaissance des formes et objets.
    """
    
    def __init__(self, config: PathwayConfig = None):
        self.config = config or PathwayConfig()
        
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Traite une frame et retourne les canaux parvocellulaires.
        
        Args:
            frame: Image couleur BGR ou RGB
            
        Returns:
            Tuple (luminance, red_green, blue_yellow):
            - luminance: Canal achromatique (détails)
            - red_green: Opposition rouge-vert
            - blue_yellow: Opposition bleu-jaune
        """
        if len(frame.shape) != 3 or frame.shape[2] < 3:
            # Image en gris: retourner luminance seulement
            gray = frame.astype(np.float32) / 255.0 if frame.max() > 1 else frame.astype(np.float32)
            return gray, np.zeros_like(gray), np.zeros_like(gray)
        
        # Séparer les canaux (assumé BGR ou RGB)
        b = frame[:, :, 0].astype(np.float32)
        g = frame[:, :, 1].astype(np.float32)
        r = frame[:, :, 2].astype(np.float32)
        
        # Normaliser [0, 1]
        max_val = max(b.max(), g.max(), r.max(), 1.0)
        if max_val > 1.0:
            b, g, r = b / 255.0, g / 255.0, r / 255.0
        
        # Luminance (pondération perceptuelle)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        
        # Opposition rouge-vert (cellules P type I)
        # Centre R, surround G (ou inverse)
        gain = self.config.parvo_color_gain
        red_green = gain * (r - g)
        red_green = np.clip(red_green, -1, 1)
        
        # Opposition bleu-jaune (cellules K)
        yellow = (r + g) / 2
        blue_yellow = gain * (b - yellow)
        blue_yellow = np.clip(blue_yellow, -1, 1)
        
        return luminance, red_green, blue_yellow
    
    def get_edges(self, luminance: np.ndarray) -> np.ndarray:
        """Détecte les bords haute résolution.
        
        Args:
            luminance: Canal de luminance
            
        Returns:
            Carte des bords
        """
        if HAS_CV2:
            # Laplacien pour détection de bords
            lap = cv2.Laplacian(luminance.astype(np.float32), cv2.CV_32F)
            return np.abs(lap)
        else:
            # Différences finies
            dx = np.diff(luminance, axis=1, prepend=luminance[:, :1])
            dy = np.diff(luminance, axis=0, prepend=luminance[:1, :])
            return np.sqrt(dx**2 + dy**2)


class LateralInhibition:
    """Inhibition latérale centre-surround.
    
    Implémente le mécanisme des cellules horizontales et amacrines
    de la rétine qui créent l'effet centre-surround (DOG).
    
    Cela améliore:
    - Détection des contours
    - Adaptation au contraste local
    - Normalisation de l'activité
    """
    
    def __init__(self, config: PathwayConfig = None):
        self.config = config or PathwayConfig()
        
    def apply(self, activation: np.ndarray) -> np.ndarray:
        """Applique l'inhibition latérale.
        
        Args:
            activation: Carte d'activation
            
        Returns:
            Activation avec inhibition latérale
        """
        center = activation
        
        # Calculer le surround (voisinage flou)
        sigma = self.config.lateral_sigma
        if HAS_CV2:
            ksize = int(sigma * 4) | 1
            surround = cv2.GaussianBlur(activation.astype(np.float32), (ksize, ksize), sigma)
        else:
            surround = self._gaussian_blur_numpy(activation, sigma)
        
        # Centre - Surround (DOG-like)
        strength = self.config.lateral_strength
        result = center - strength * surround
        
        # Rectification (ReLU biologique)
        return np.clip(result, 0, None)
    
    def _gaussian_blur_numpy(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """Fallback Gaussien."""
        size = int(sigma * 4) | 1
        x = np.arange(size) - size // 2
        kernel_1d = np.exp(-x**2 / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        result = np.apply_along_axis(lambda row: np.convolve(row, kernel_1d, 'same'), 1, img)
        result = np.apply_along_axis(lambda col: np.convolve(col, kernel_1d, 'same'), 0, result)
        return result


class GaborFilterBank:
    """Banc de filtres de Gabor - Modèle de V1.
    
    Implémente les cellules simples du cortex visuel primaire
    découvertes par Hubel & Wiesel (1962).
    
    Chaque filtre est sensible à:
    - Une orientation spécifique
    - Une fréquence spatiale
    - Une phase (ON/OFF)
    """
    
    def __init__(self, config: PathwayConfig = None):
        self.config = config or PathwayConfig()
        self.filters: List[np.ndarray] = []
        self.orientations: List[float] = []
        self.scales: List[float] = []
        self._build_filters()
        
    def _build_filters(self):
        """Construit le banc de filtres de Gabor."""
        cfg = self.config
        
        self.filters = []
        self.orientations = []
        self.scales = []
        
        for scale_idx in range(cfg.gabor_num_scales):
            # Longueur d'onde augmente avec l'échelle
            wavelength = cfg.gabor_wavelength * (2 ** scale_idx)
            sigma = cfg.gabor_sigma * (1.5 ** scale_idx)
            
            for ori_idx in range(cfg.gabor_num_orientations):
                # Orientation en radians
                theta = ori_idx * np.pi / cfg.gabor_num_orientations
                
                if HAS_CV2:
                    # Taille du kernel (doit être impaire)
                    ksize = int(sigma * 6) | 1
                    
                    gabor = cv2.getGaborKernel(
                        ksize=(ksize, ksize),
                        sigma=sigma,
                        theta=theta,
                        lambd=wavelength,
                        gamma=cfg.gabor_gamma,
                        psi=0  # Phase 0 (paire)
                    )
                    self.filters.append(gabor)
                else:
                    gabor = self._make_gabor_numpy(sigma, theta, wavelength, cfg.gabor_gamma)
                    self.filters.append(gabor)
                
                self.orientations.append(theta)
                self.scales.append(scale_idx)
    
    def _make_gabor_numpy(self, sigma: float, theta: float, wavelength: float, gamma: float) -> np.ndarray:
        """Crée un filtre de Gabor en NumPy pur."""
        size = int(sigma * 6) | 1
        half = size // 2
        
        y, x = np.mgrid[-half:half+1, -half:half+1].astype(np.float32)
        
        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        # Gabor = Gaussienne × Sinusoïde
        gaussian = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
        sinusoid = np.cos(2 * np.pi * x_theta / wavelength)
        
        gabor = gaussian * sinusoid
        gabor -= gabor.mean()  # Zero-mean
        gabor /= np.abs(gabor).sum() + 1e-6  # Normaliser
        
        return gabor
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """Applique tous les filtres de Gabor.
        
        Args:
            image: Image en niveaux de gris
            
        Returns:
            Tensor (H, W, num_filters) des réponses
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Normaliser
        gray = gray / 255.0 if gray.max() > 1.0 else gray
        
        h, w = gray.shape
        num_filters = len(self.filters)
        responses = np.zeros((h, w, num_filters), dtype=np.float32)
        
        for i, kernel in enumerate(self.filters):
            if HAS_CV2:
                resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
            else:
                # Convolution 2D manuelle (lent mais fonctionnel)
                resp = self._convolve2d(gray, kernel)
            
            responses[:, :, i] = resp
        
        return responses
    
    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Convolution 2D simple (fallback sans OpenCV)."""
        from scipy.signal import convolve2d
        return convolve2d(image, kernel, mode='same')
    
    def get_orientation_energy(self, responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule l'énergie et l'orientation dominante par pixel.
        
        Args:
            responses: Sortie de process()
            
        Returns:
            Tuple (energy, orientation):
            - energy: Énergie totale
            - orientation: Angle dominant (0 à π)
        """
        # Somme des carrés par orientation (toutes échelles)
        num_ori = self.config.gabor_num_orientations
        num_scales = self.config.gabor_num_scales
        
        h, w, _ = responses.shape
        ori_energy = np.zeros((h, w, num_ori), dtype=np.float32)
        
        for scale_idx in range(num_scales):
            for ori_idx in range(num_ori):
                filter_idx = scale_idx * num_ori + ori_idx
                ori_energy[:, :, ori_idx] += responses[:, :, filter_idx] ** 2
        
        # Énergie totale
        energy = np.sqrt(np.sum(ori_energy, axis=2))
        
        # Orientation dominante
        dominant_ori_idx = np.argmax(ori_energy, axis=2)
        orientation = dominant_ori_idx.astype(np.float32) * np.pi / num_ori
        
        return energy, orientation


class RetinalProcessor:
    """Processeur rétinien complet combinant toutes les voies.
    
    Pipeline:
    1. Séparation Magno/Parvo/Konio
    2. Inhibition latérale
    3. Filtres de Gabor (V1)
    
    Usage:
        processor = RetinalProcessor()
        result = processor.process(frame)
        
        # Accès aux différentes voies
        motion = result['magno']['motion']
        color_rg = result['parvo']['red_green']
        orientation = result['v1']['orientation']
    """
    
    def __init__(self, config: PathwayConfig = None):
        self.config = config or PathwayConfig()
        
        self.magno = MagnocellularPathway(self.config)
        self.parvo = ParvocellularPathway(self.config)
        self.lateral = LateralInhibition(self.config)
        self.gabor = GaborFilterBank(self.config)
        
    def process(self, frame: np.ndarray) -> dict:
        """Traite une frame à travers toutes les voies.
        
        Args:
            frame: Image BGR/RGB ou grayscale
            
        Returns:
            Dictionnaire avec les résultats de chaque voie
        """
        results = {}
        
        # Voie Magnocellulaire (mouvement)
        motion_energy, motion_dir = self.magno.process(frame)
        motion_inhibited = self.lateral.apply(motion_energy)
        results['magno'] = {
            'motion': motion_energy,
            'motion_inhibited': motion_inhibited,
            'direction': motion_dir,
        }
        
        # Voie Parvocellulaire (couleur et détails)
        luminance, red_green, blue_yellow = self.parvo.process(frame)
        luminance_inhibited = self.lateral.apply(luminance)
        edges = self.parvo.get_edges(luminance)
        results['parvo'] = {
            'luminance': luminance,
            'luminance_inhibited': luminance_inhibited,
            'red_green': red_green,
            'blue_yellow': blue_yellow,
            'edges': edges,
        }
        
        # V1 - Filtres de Gabor sur la luminance
        gabor_responses = self.gabor.process(luminance)
        energy, orientation = self.gabor.get_orientation_energy(gabor_responses)
        results['v1'] = {
            'responses': gabor_responses,
            'energy': energy,
            'orientation': orientation,
        }
        
        # Saillance combinée (bottom-up)
        saliency = self._compute_saliency(results)
        results['saliency'] = saliency
        
        return results
    
    def _compute_saliency(self, results: dict) -> np.ndarray:
        """Calcule une carte de saillance bottom-up.
        
        Combine:
        - Mouvement (très saillant)
        - Contraste de couleur
        - Énergie d'orientation
        - Bords
        """
        motion = results['magno']['motion_inhibited']
        color_rg = np.abs(results['parvo']['red_green'])
        color_by = np.abs(results['parvo']['blue_yellow'])
        edges = results['parvo']['edges']
        energy = results['v1']['energy']
        
        # Redimensionner si nécessaire (Gabor peut avoir shape différente)
        target_shape = motion.shape
        
        def resize_if_needed(arr):
            if arr.shape != target_shape:
                if HAS_CV2:
                    return cv2.resize(arr, (target_shape[1], target_shape[0]))
                else:
                    # Sous-échantillonnage simple
                    return arr[:target_shape[0], :target_shape[1]]
            return arr
        
        # Normaliser chaque canal [0, 1]
        def normalize(arr):
            arr = resize_if_needed(arr)
            min_v, max_v = arr.min(), arr.max()
            if max_v - min_v < 1e-6:
                return np.zeros_like(arr)
            return (arr - min_v) / (max_v - min_v)
        
        # Pondération bio-inspirée
        # Le mouvement est très saillant (survie!)
        saliency = (
            0.4 * normalize(motion) +
            0.15 * normalize(color_rg) +
            0.10 * normalize(color_by) +
            0.15 * normalize(edges) +
            0.20 * normalize(energy)
        )
        
        return saliency
    
    def reset(self):
        """Réinitialise l'état temporel."""
        self.magno.reset()


# Export pour le module
__all__ = [
    'PathwayConfig',
    'MagnocellularPathway',
    'ParvocellularPathway',
    'LateralInhibition',
    'GaborFilterBank',
    'RetinalProcessor',
]
