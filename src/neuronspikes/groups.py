"""
Détection de groupes d'activation (clusters) dans les patterns de spikes.

Les groupes d'activation sont des ensembles de neurones qui s'activent
simultanément. Ces patterns de co-activation sont les "corrélations candidates"
pour la formation de nouveaux neurones dans les couches supérieures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


@dataclass
class ActivationGroup:
    """Un groupe de neurones co-activés.
    
    Attributes:
        pixels: Ensemble des coordonnées (y, x) des neurones activés
        slot: Index du slot temporel où le groupe s'est formé
        frame: Index de la frame
        size: Nombre de neurones dans le groupe
        centroid: Centre de masse du groupe (y, x)
        bounding_box: Boîte englobante (y_min, x_min, y_max, x_max)
    """
    pixels: Set[Tuple[int, int]]
    slot: int
    frame: int
    size: int = field(init=False)
    centroid: Tuple[float, float] = field(init=False)
    bounding_box: Tuple[int, int, int, int] = field(init=False)
    
    def __post_init__(self):
        self.size = len(self.pixels)
        if self.pixels:
            ys = [p[0] for p in self.pixels]
            xs = [p[1] for p in self.pixels]
            self.centroid = (sum(ys) / len(ys), sum(xs) / len(xs))
            self.bounding_box = (min(ys), min(xs), max(ys), max(xs))
        else:
            self.centroid = (0.0, 0.0)
            self.bounding_box = (0, 0, 0, 0)
    
    def overlaps(self, other: 'ActivationGroup', threshold: float = 0.5) -> bool:
        """Vérifie si deux groupes se chevauchent significativement.
        
        Args:
            other: Autre groupe à comparer
            threshold: Fraction minimale de pixels communs (Jaccard)
            
        Returns:
            True si les groupes se chevauchent au-delà du seuil
        """
        if not self.pixels or not other.pixels:
            return False
        
        intersection = len(self.pixels & other.pixels)
        union = len(self.pixels | other.pixels)
        
        return (intersection / union) >= threshold if union > 0 else False
    
    def distance_to(self, other: 'ActivationGroup') -> float:
        """Distance euclidienne entre les centroïdes."""
        dy = self.centroid[0] - other.centroid[0]
        dx = self.centroid[1] - other.centroid[1]
        return (dy**2 + dx**2) ** 0.5


@dataclass
class GroupDetectorConfig:
    """Configuration du détecteur de groupes.
    
    Attributes:
        min_group_size: Taille minimale d'un groupe (pixels)
        connectivity: 4 ou 8 - voisinage pour la connexité
        track_history: Nombre de frames d'historique à garder
    """
    min_group_size: int = 2
    connectivity: int = 8  # 4 ou 8 voisins
    track_history: int = 10


class GroupDetector:
    """Détecteur de groupes d'activation par composantes connexes.
    
    Identifie les clusters de neurones qui s'activent simultanément
    dans chaque slot temporel.
    """
    
    def __init__(self, config: Optional[GroupDetectorConfig] = None):
        """Initialise le détecteur.
        
        Args:
            config: Configuration du détecteur
        """
        self.config = config or GroupDetectorConfig()
        
        # Historique des groupes détectés
        self.history: List[List[ActivationGroup]] = []
        
        # Statistiques
        self.stats = {
            'total_groups': 0,
            'total_pixels_in_groups': 0,
            'max_group_size': 0,
            'frames_processed': 0,
        }
        
        # Offsets pour le voisinage
        if self.config.connectivity == 4:
            self._neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # 8-connectivity
            self._neighbors = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
    
    def detect_groups(
        self, 
        activations: NDArray[np.bool_],
        slot: int = 0,
        frame: int = 0
    ) -> List[ActivationGroup]:
        """Détecte les groupes de neurones co-activés.
        
        Utilise un algorithme de composantes connexes (flood fill).
        
        Args:
            activations: Masque d'activation (H, W) booléen
            slot: Index du slot temporel
            frame: Index de la frame
            
        Returns:
            Liste des groupes détectés
        """
        h, w = activations.shape
        visited = np.zeros_like(activations, dtype=np.bool_)
        groups = []
        
        # Parcourir tous les pixels activés
        for y in range(h):
            for x in range(w):
                if activations[y, x] and not visited[y, x]:
                    # Nouveau groupe trouvé - flood fill
                    pixels = self._flood_fill(activations, visited, y, x, h, w)
                    
                    if len(pixels) >= self.config.min_group_size:
                        group = ActivationGroup(
                            pixels=pixels,
                            slot=slot,
                            frame=frame
                        )
                        groups.append(group)
                        
                        # Stats
                        self.stats['total_groups'] += 1
                        self.stats['total_pixels_in_groups'] += group.size
                        self.stats['max_group_size'] = max(
                            self.stats['max_group_size'], 
                            group.size
                        )
        
        return groups
    
    def _flood_fill(
        self,
        activations: NDArray[np.bool_],
        visited: NDArray[np.bool_],
        start_y: int,
        start_x: int,
        h: int,
        w: int
    ) -> Set[Tuple[int, int]]:
        """Flood fill pour trouver les pixels connectés.
        
        Args:
            activations: Masque d'activation
            visited: Masque des pixels déjà visités
            start_y, start_x: Point de départ
            h, w: Dimensions de l'image
            
        Returns:
            Ensemble des pixels du groupe
        """
        pixels = set()
        stack = [(start_y, start_x)]
        
        while stack:
            y, x = stack.pop()
            
            if visited[y, x]:
                continue
            
            visited[y, x] = True
            pixels.add((y, x))
            
            # Explorer les voisins
            for dy, dx in self._neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if activations[ny, nx] and not visited[ny, nx]:
                        stack.append((ny, nx))
        
        return pixels
    
    def process_frame_activations(
        self,
        activations_list: List[NDArray[np.bool_]],
        frame: int = 0
    ) -> List[List[ActivationGroup]]:
        """Traite toutes les activations d'une frame (256 slots).
        
        Args:
            activations_list: Liste de 256 masques d'activation
            frame: Index de la frame
            
        Returns:
            Liste de groupes par slot
        """
        all_groups = []
        
        for slot, activations in enumerate(activations_list):
            groups = self.detect_groups(activations, slot, frame)
            all_groups.append(groups)
        
        self.stats['frames_processed'] += 1
        
        # Maintenir l'historique
        self.history.append(all_groups)
        if len(self.history) > self.config.track_history:
            self.history.pop(0)
        
        return all_groups
    
    def get_recurring_patterns(
        self,
        min_occurrences: int = 3,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[ActivationGroup, int]]:
        """Trouve les patterns de groupes qui se répètent dans l'historique.
        
        C'est ici que les "corrélations candidates" deviennent des
        candidats pour la création de nouveaux neurones.
        
        Args:
            min_occurrences: Nombre minimum d'apparitions
            similarity_threshold: Seuil de similarité (Jaccard)
            
        Returns:
            Liste de (groupe représentatif, nombre d'occurrences)
        """
        if not self.history:
            return []
        
        # Collecter tous les groupes de l'historique
        all_groups = []
        for frame_groups in self.history:
            for slot_groups in frame_groups:
                all_groups.extend(slot_groups)
        
        if not all_groups:
            return []
        
        # Clustering simple par similarité
        clusters = []
        used = set()
        
        for i, group in enumerate(all_groups):
            if i in used:
                continue
            
            cluster = [group]
            used.add(i)
            
            for j, other in enumerate(all_groups[i+1:], i+1):
                if j not in used and group.overlaps(other, similarity_threshold):
                    cluster.append(other)
                    used.add(j)
            
            if len(cluster) >= min_occurrences:
                # Prendre le premier comme représentatif
                clusters.append((cluster[0], len(cluster)))
        
        # Trier par nombre d'occurrences
        clusters.sort(key=lambda x: x[1], reverse=True)
        
        return clusters
    
    def reset(self):
        """Réinitialise le détecteur."""
        self.history.clear()
        self.stats = {
            'total_groups': 0,
            'total_pixels_in_groups': 0,
            'max_group_size': 0,
            'frames_processed': 0,
        }


def visualize_groups(
    activations: NDArray[np.bool_],
    groups: List[ActivationGroup]
) -> NDArray[np.uint8]:
    """Crée une image de visualisation des groupes.
    
    Chaque groupe reçoit une couleur différente.
    
    Args:
        activations: Masque d'activation original
        groups: Groupes détectés
        
    Returns:
        Image RGB (H, W, 3) avec les groupes colorés
    """
    h, w = activations.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Couleurs pour les groupes (cycle de couleurs vives)
    colors = [
        (255, 0, 0),    # Rouge
        (0, 255, 0),    # Vert
        (0, 0, 255),    # Bleu
        (255, 255, 0),  # Jaune
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Violet
    ]
    
    for i, group in enumerate(groups):
        color = colors[i % len(colors)]
        for y, x in group.pixels:
            vis[y, x] = color
    
    # Pixels activés mais pas dans un groupe (trop petits)
    for y in range(h):
        for x in range(w):
            if activations[y, x] and vis[y, x].sum() == 0:
                vis[y, x] = (64, 64, 64)  # Gris foncé
    
    return vis
