"""
Fovea - Rétine polaire bio-inspirée avec attention.

Inspiré de:
- TestVisionCS/Vision.cs (Stéphane Denis, ~2010s)
- Observation VOR chat Peyo (1995)

Architecture de la rétine biologique:
- Fovéa centrale: haute résolution, cônes
- Périphérie: basse résolution, bâtonnets
- Organisation polaire (secteurs angulaires + anneaux)

Caractéristiques:
- Résolution décroissante du centre vers la périphérie
- Divisions angulaires discrètes (groupes de rotation)
- Point de fixation mobile (saccades)
- Support stéréoscopique (2 fovéas)
- Compensation de rotation (VOR)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray


@dataclass
class FoveaConfig:
    """Configuration de la fovéa.
    
    Attributes:
        num_rings: Nombre d'anneaux concentriques
        num_sectors: Nombre de secteurs angulaires (doit être puissance de 2)
        fovea_radius: Rayon de la zone fovéale (haute résolution)
        max_radius: Rayon maximum de la rétine
        center_resolution: Résolution au centre (pixels par cellule)
        peripheral_falloff: Facteur de décroissance de résolution
    """
    num_rings: int = 32
    num_sectors: int = 16  # Divisions angulaires discrètes
    fovea_radius: int = 16  # Zone centrale haute résolution
    max_radius: int = 128
    center_resolution: float = 1.0  # 1 pixel = 1 cellule au centre
    peripheral_falloff: float = 1.5  # Résolution décroît en r^falloff


@dataclass
class GazePoint:
    """Point de fixation du regard.
    
    Attributes:
        x, y: Position du point de fixation dans l'image source
        theta: Angle de rotation de la rétine (compensation VOR)
        vergence: Angle de convergence pour stéréo (0 = parallèle)
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # Rotation en radians
    vergence: float = 0.0  # Pour stéréo
    
    def move(self, dx: float, dy: float):
        """Déplace le point de fixation (saccade)."""
        self.x += dx
        self.y += dy
    
    def rotate(self, dtheta: float):
        """Applique une rotation (compensation VOR)."""
        self.theta += dtheta
        # Normaliser dans [-π, π]
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
    
    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class PolarCell:
    """Une cellule de la rétine polaire.
    
    Attributes:
        ring: Index de l'anneau (0 = centre)
        sector: Index du secteur angulaire
        inner_radius: Rayon intérieur
        outer_radius: Rayon extérieur
        start_angle: Angle de début (radians)
        end_angle: Angle de fin (radians)
    """
    ring: int
    sector: int
    inner_radius: float
    outer_radius: float
    start_angle: float
    end_angle: float
    
    @property
    def center_radius(self) -> float:
        return (self.inner_radius + self.outer_radius) / 2
    
    @property
    def center_angle(self) -> float:
        return (self.start_angle + self.end_angle) / 2
    
    @property
    def area(self) -> float:
        """Surface approximative de la cellule."""
        dr = self.outer_radius - self.inner_radius
        dtheta = self.end_angle - self.start_angle
        r = self.center_radius
        return r * dr * dtheta
    
    def to_cartesian(self, cx: float, cy: float, rotation: float = 0.0) -> Tuple[float, float]:
        """Convertit le centre de la cellule en coordonnées cartésiennes.
        
        Args:
            cx, cy: Centre de la fovéa
            rotation: Rotation additionnelle (compensation VOR)
        """
        angle = self.center_angle + rotation
        r = self.center_radius
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        return (x, y)


class Fovea:
    """Rétine polaire avec fovéa centrale.
    
    Simule l'organisation de la rétine biologique:
    - Zone fovéale centrale à haute résolution
    - Périphérie à résolution décroissante
    - Organisation en secteurs angulaires (groupes de rotation discrets)
    """
    
    def __init__(self, config: Optional[FoveaConfig] = None):
        """Initialise la fovéa.
        
        Args:
            config: Configuration de la fovéa
        """
        self.config = config or FoveaConfig()
        
        # Point de fixation actuel
        self.gaze = GazePoint()
        
        # Grille de cellules polaires
        self.cells: List[List[PolarCell]] = []
        
        # Buffer d'activation
        self._activations: NDArray[np.float32] = np.zeros(
            (self.config.num_rings, self.config.num_sectors),
            dtype=np.float32
        )
        
        # Pré-calculer la géométrie des cellules
        self._build_cells()
        
        # Pré-calculer les masques de sampling pour chaque cellule
        self._sampling_masks: Dict[Tuple[int, int], NDArray[np.bool_]] = {}
        
        # Statistiques
        self._frame_count = 0
        self._total_samples = 0
    
    def _build_cells(self):
        """Construit la grille de cellules polaires."""
        cfg = self.config
        self.cells = []
        
        # Calcul des rayons avec résolution décroissante
        # r(i) = fovea_radius * (1 + i)^falloff pour simuler log-polar
        radii = [0.0]
        for i in range(cfg.num_rings):
            if i < cfg.num_rings // 4:
                # Zone fovéale: espacement linéaire
                r = cfg.fovea_radius * (i + 1) / (cfg.num_rings // 4)
            else:
                # Zone périphérique: espacement exponentiel
                t = (i - cfg.num_rings // 4) / (cfg.num_rings * 0.75)
                r = cfg.fovea_radius + (cfg.max_radius - cfg.fovea_radius) * (t ** cfg.peripheral_falloff)
            radii.append(min(r, cfg.max_radius))
        
        # Créer les cellules
        sector_angle = 2 * math.pi / cfg.num_sectors
        
        for ring_idx in range(cfg.num_rings):
            ring = []
            inner_r = radii[ring_idx]
            outer_r = radii[ring_idx + 1]
            
            for sector_idx in range(cfg.num_sectors):
                start_angle = sector_idx * sector_angle
                end_angle = (sector_idx + 1) * sector_angle
                
                cell = PolarCell(
                    ring=ring_idx,
                    sector=sector_idx,
                    inner_radius=inner_r,
                    outer_radius=outer_r,
                    start_angle=start_angle,
                    end_angle=end_angle
                )
                ring.append(cell)
            
            self.cells.append(ring)
    
    def set_gaze(self, x: float, y: float, theta: float = 0.0):
        """Définit le point de fixation.
        
        Args:
            x, y: Position dans l'image source
            theta: Rotation de la rétine (compensation VOR)
        """
        self.gaze.x = x
        self.gaze.y = y
        self.gaze.theta = theta
    
    def saccade(self, dx: float, dy: float):
        """Effectue une saccade (mouvement rapide du regard).
        
        Args:
            dx, dy: Déplacement du point de fixation
        """
        self.gaze.move(dx, dy)
    
    def rotate(self, dtheta: float):
        """Applique une rotation (compensation VOR).
        
        Args:
            dtheta: Angle de rotation en radians
        """
        self.gaze.rotate(dtheta)
    
    def sample(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Échantillonne l'image selon la géométrie fovéale.
        
        Args:
            image: Image source (grayscale H×W ou RGB H×W×3)
            
        Returns:
            Matrice d'activation (num_rings × num_sectors)
        """
        self._frame_count += 1
        
        # Convertir en grayscale si nécessaire
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        h, w = gray.shape
        cx, cy = self.gaze.x, self.gaze.y
        theta = self.gaze.theta
        
        # Échantillonner chaque cellule
        for ring_idx, ring in enumerate(self.cells):
            for sector_idx, cell in enumerate(ring):
                value = self._sample_cell(gray, cell, cx, cy, theta)
                self._activations[ring_idx, sector_idx] = value
        
        return self._activations.copy()
    
    def _sample_cell(
        self, 
        image: NDArray[np.uint8], 
        cell: PolarCell,
        cx: float, 
        cy: float, 
        theta: float
    ) -> float:
        """Échantillonne une cellule polaire.
        
        Utilise un échantillonnage par points dans la zone de la cellule.
        """
        h, w = image.shape
        
        # Nombre de points d'échantillonnage proportionnel à l'aire
        num_samples = max(1, int(cell.area / 4))
        self._total_samples += num_samples
        
        total = 0.0
        valid_samples = 0
        
        # Échantillonner dans la cellule
        for i in range(num_samples):
            # Position dans la cellule (interpolation)
            t_r = (i + 0.5) / num_samples
            t_a = (i % 3 + 0.5) / 3  # Variation angulaire
            
            r = cell.inner_radius + t_r * (cell.outer_radius - cell.inner_radius)
            angle = cell.start_angle + t_a * (cell.end_angle - cell.start_angle) + theta
            
            # Convertir en coordonnées image
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            
            # Vérifier les bornes
            if 0 <= x < w and 0 <= y < h:
                total += image[y, x]
                valid_samples += 1
        
        if valid_samples > 0:
            return total / (valid_samples * 255.0)
        return 0.0
    
    def get_cell_positions(self) -> NDArray[np.float32]:
        """Retourne les positions cartésiennes de toutes les cellules.
        
        Returns:
            Array (num_rings × num_sectors × 2) avec (x, y) pour chaque cellule
        """
        positions = np.zeros(
            (self.config.num_rings, self.config.num_sectors, 2),
            dtype=np.float32
        )
        
        cx, cy = self.gaze.x, self.gaze.y
        theta = self.gaze.theta
        
        for ring_idx, ring in enumerate(self.cells):
            for sector_idx, cell in enumerate(ring):
                x, y = cell.to_cartesian(cx, cy, theta)
                positions[ring_idx, sector_idx] = [x, y]
        
        return positions
    
    def get_fovea_mask(self) -> NDArray[np.bool_]:
        """Retourne un masque de la zone fovéale (haute résolution).
        
        Returns:
            Masque booléen (num_rings × num_sectors)
        """
        fovea_rings = self.config.num_rings // 4
        mask = np.zeros(
            (self.config.num_rings, self.config.num_sectors),
            dtype=bool
        )
        mask[:fovea_rings, :] = True
        return mask
    
    def get_sector_activations(self, sector: int) -> NDArray[np.float32]:
        """Retourne les activations d'un secteur angulaire.
        
        Utile pour détecter les patterns radiaux.
        
        Args:
            sector: Index du secteur (0 à num_sectors-1)
        """
        return self._activations[:, sector].copy()
    
    def get_ring_activations(self, ring: int) -> NDArray[np.float32]:
        """Retourne les activations d'un anneau.
        
        Utile pour détecter les patterns circulaires (rotation).
        
        Args:
            ring: Index de l'anneau (0 = centre)
        """
        return self._activations[ring, :].copy()
    
    def detect_rotation(self, prev_activations: NDArray[np.float32]) -> int:
        """Détecte la rotation entre deux frames.
        
        Compare les activations actuelles avec les précédentes
        pour trouver le décalage angulaire optimal.
        
        Args:
            prev_activations: Activations de la frame précédente
            
        Returns:
            Nombre de secteurs de décalage (-num_sectors/2 à +num_sectors/2)
        """
        best_shift = 0
        best_correlation = -1.0
        
        num_sectors = self.config.num_sectors
        
        # Tester chaque décalage possible
        for shift in range(-num_sectors // 2, num_sectors // 2 + 1):
            # Décaler les activations actuelles
            shifted = np.roll(self._activations, shift, axis=1)
            
            # Calculer la corrélation
            correlation = np.sum(shifted * prev_activations)
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_shift = shift
        
        return best_shift
    
    def compensate_rotation(self, prev_activations: NDArray[np.float32]) -> float:
        """Compense la rotation détectée (VOR).
        
        Détecte et applique la compensation de rotation.
        
        Args:
            prev_activations: Activations de la frame précédente
            
        Returns:
            Angle de compensation appliqué (radians)
        """
        shift = self.detect_rotation(prev_activations)
        
        # Convertir en angle
        angle_per_sector = 2 * math.pi / self.config.num_sectors
        compensation = -shift * angle_per_sector
        
        # Appliquer la compensation
        self.gaze.rotate(compensation)
        
        return compensation
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de la fovéa."""
        return {
            'num_rings': self.config.num_rings,
            'num_sectors': self.config.num_sectors,
            'total_cells': self.config.num_rings * self.config.num_sectors,
            'gaze_x': self.gaze.x,
            'gaze_y': self.gaze.y,
            'gaze_theta': self.gaze.theta,
            'frame_count': self._frame_count,
            'total_samples': self._total_samples,
            'avg_activation': float(np.mean(self._activations)),
        }
    
    def reset(self):
        """Réinitialise la fovéa."""
        self._activations.fill(0)
        self.gaze = GazePoint()
        self._frame_count = 0
        self._total_samples = 0


class StereoFovea:
    """Système stéréoscopique avec deux fovéas.
    
    Simule la vision binoculaire avec:
    - Deux fovéas (gauche et droite)
    - Disparité binoculaire pour la profondeur
    - Vergence (convergence des axes optiques)
    """
    
    def __init__(
        self,
        config: Optional[FoveaConfig] = None,
        baseline: float = 60.0  # Distance inter-oculaire en pixels
    ):
        """Initialise le système stéréo.
        
        Args:
            config: Configuration des fovéas
            baseline: Distance entre les deux caméras/yeux
        """
        self.config = config or FoveaConfig()
        self.baseline = baseline
        
        # Deux fovéas
        self.left = Fovea(self.config)
        self.right = Fovea(self.config)
        
        # Point de fixation commun (dans l'espace 3D)
        self._target_x = 0.0
        self._target_y = 0.0
        self._target_depth = 100.0  # Distance focale
        
        # Vergence actuelle
        self._vergence = 0.0
    
    def set_target(self, x: float, y: float, depth: float = 100.0):
        """Définit le point de fixation 3D.
        
        Args:
            x, y: Position dans le plan image central
            depth: Distance du point de fixation
        """
        self._target_x = x
        self._target_y = y
        self._target_depth = max(1.0, depth)
        
        # Calculer les positions fovéales pour chaque œil
        # Vergence = angle de convergence
        half_baseline = self.baseline / 2
        self._vergence = math.atan2(half_baseline, self._target_depth)
        
        # Position gauche (décalage vers la droite pour converger)
        offset_x = half_baseline * (1 - depth / self._target_depth) if self._target_depth > 0 else 0
        self.left.set_gaze(x - offset_x, y)
        
        # Position droite (décalage vers la gauche)
        self.right.set_gaze(x + offset_x, y)
    
    def sample(
        self, 
        left_image: NDArray[np.uint8], 
        right_image: NDArray[np.uint8]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Échantillonne les deux images.
        
        Args:
            left_image: Image de la caméra gauche
            right_image: Image de la caméra droite
            
        Returns:
            Tuple (activations_gauche, activations_droite)
        """
        left_act = self.left.sample(left_image)
        right_act = self.right.sample(right_image)
        return left_act, right_act
    
    def compute_disparity(self) -> NDArray[np.float32]:
        """Calcule la disparité entre les deux fovéas.
        
        La disparité est la différence d'activation qui encode la profondeur.
        
        Returns:
            Matrice de disparité (num_rings × num_sectors)
        """
        return self.left._activations - self.right._activations
    
    def estimate_depth_map(self) -> NDArray[np.float32]:
        """Estime la carte de profondeur relative.
        
        Utilise la disparité pour estimer la distance relative.
        
        Returns:
            Carte de profondeur relative (num_rings × num_sectors)
        """
        disparity = self.compute_disparity()
        
        # Éviter division par zéro
        disparity_safe = np.where(
            np.abs(disparity) < 0.01,
            0.01 * np.sign(disparity + 0.001),
            disparity
        )
        
        # Profondeur inversement proportionnelle à la disparité
        # (simplification - en réalité dépend de la géométrie)
        depth = self.baseline / (np.abs(disparity_safe) + 0.1)
        
        return depth
    
    def saccade(self, dx: float, dy: float):
        """Saccade coordonnée des deux yeux."""
        self.left.saccade(dx, dy)
        self.right.saccade(dx, dy)
    
    def rotate(self, dtheta: float):
        """Rotation coordonnée des deux yeux (VOR)."""
        self.left.rotate(dtheta)
        self.right.rotate(dtheta)
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du système stéréo."""
        return {
            'baseline': self.baseline,
            'vergence': self._vergence,
            'target_depth': self._target_depth,
            'left': self.left.get_stats(),
            'right': self.right.get_stats(),
        }


def visualize_fovea(
    fovea: Fovea,
    size: int = 256,
    show_grid: bool = True
) -> NDArray[np.uint8]:
    """Visualise l'état de la fovéa.
    
    Args:
        fovea: Fovéa à visualiser
        size: Taille de l'image de sortie
        show_grid: Afficher la grille polaire
        
    Returns:
        Image RGB de visualisation
    """
    import numpy as np
    
    # Créer image de sortie
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    center = size // 2
    scale = size / (2 * fovea.config.max_radius)
    
    # Dessiner les cellules
    for ring_idx, ring in enumerate(fovea.cells):
        for sector_idx, cell in enumerate(ring):
            # Couleur basée sur l'activation
            activation = fovea._activations[ring_idx, sector_idx]
            color_val = int(activation * 255)
            
            # Zone fovéale en teinte chaude, périphérie en froid
            if ring_idx < fovea.config.num_rings // 4:
                color = (color_val, color_val // 2, 0)  # Orange
            else:
                color = (0, color_val // 2, color_val)  # Cyan
            
            # Dessiner l'arc de la cellule
            inner_r = int(cell.inner_radius * scale)
            outer_r = int(cell.outer_radius * scale)
            start_deg = int(math.degrees(cell.start_angle + fovea.gaze.theta))
            end_deg = int(math.degrees(cell.end_angle + fovea.gaze.theta))
            
            # Remplir avec des lignes radiales (approximation)
            for r in range(inner_r, outer_r + 1):
                for angle_deg in range(start_deg, end_deg + 1):
                    angle_rad = math.radians(angle_deg)
                    x = int(center + r * math.cos(angle_rad))
                    y = int(center + r * math.sin(angle_rad))
                    if 0 <= x < size and 0 <= y < size:
                        img[y, x] = color
    
    # Dessiner la grille si demandé
    if show_grid:
        grid_color = (64, 64, 64)
        
        # Cercles concentriques
        for ring in fovea.cells:
            if ring:
                r = int(ring[0].outer_radius * scale)
                for angle in range(360):
                    angle_rad = math.radians(angle)
                    x = int(center + r * math.cos(angle_rad))
                    y = int(center + r * math.sin(angle_rad))
                    if 0 <= x < size and 0 <= y < size:
                        img[y, x] = grid_color
        
        # Rayons
        for sector_idx in range(fovea.config.num_sectors):
            angle = sector_idx * 2 * math.pi / fovea.config.num_sectors + fovea.gaze.theta
            for r in range(int(fovea.config.max_radius * scale)):
                x = int(center + r * math.cos(angle))
                y = int(center + r * math.sin(angle))
                if 0 <= x < size and 0 <= y < size:
                    img[y, x] = grid_color
    
    # Marquer le centre (point de fixation)
    cv_center = 3
    for dx in range(-cv_center, cv_center + 1):
        for dy in range(-cv_center, cv_center + 1):
            if dx*dx + dy*dy <= cv_center*cv_center:
                x, y = center + dx, center + dy
                if 0 <= x < size and 0 <= y < size:
                    img[y, x] = (255, 255, 255)
    
    return img
