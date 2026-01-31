"""
Corrélations temporelles - Détection de patterns récurrents entre frames.

Ce module implémente la détection de co-activations qui se répètent
dans le temps, base pour la genèse de nouveaux neurones.

Concept:
- Un pattern temporel = groupe de pixels qui s'activent ensemble
  de manière répétée sur plusieurs frames consécutifs ou proches
- La récurrence d'un pattern indique une structure stable du monde
- Ces patterns récurrents déclenchent la création de neurones

Aucun hasard: tout est basé sur des statistiques observées.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import deque

from .groups import ActivationGroup, GroupDetector


@dataclass
class TemporalPattern:
    """Pattern d'activation récurrent dans le temps.
    
    Attributes:
        pattern_id: Identifiant unique du pattern
        signature: Masque binaire du pattern (pixels qui co-activent)
        first_seen: Numéro du premier frame où le pattern a été vu
        last_seen: Numéro du dernier frame où le pattern a été vu
        occurrences: Nombre de fois que le pattern a été détecté
        confidence: Score de confiance (0-1) basé sur la récurrence
        centroid: Centre de masse du pattern
    """
    pattern_id: int
    signature: np.ndarray  # Masque binaire (H, W)
    first_seen: int
    last_seen: int
    occurrences: int = 1
    confidence: float = 0.0
    centroid: tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        """Calcule le centroïde à partir de la signature."""
        if self.signature is not None and self.signature.any():
            ys, xs = np.where(self.signature)
            self.centroid = (float(np.mean(xs)), float(np.mean(ys)))
    
    @property
    def pixel_count(self) -> int:
        """Nombre de pixels dans le pattern."""
        return int(np.sum(self.signature))
    
    @property
    def age(self) -> int:
        """Âge du pattern en frames depuis sa première observation."""
        return self.last_seen - self.first_seen
    
    @property
    def frequency(self) -> float:
        """Fréquence d'apparition (occurrences / âge)."""
        if self.age == 0:
            return 1.0
        return self.occurrences / (self.age + 1)


@dataclass
class CorrelationConfig:
    """Configuration pour la détection de corrélations temporelles.
    
    Attributes:
        history_size: Nombre de frames à garder en mémoire
        min_overlap: Chevauchement minimum (0-1) pour considérer deux groupes similaires
        min_occurrences: Nombre minimum d'occurrences pour qu'un pattern soit stable
        confidence_threshold: Seuil de confiance pour considérer un pattern valide
        decay_rate: Taux de décroissance de la confiance si pattern non vu
    """
    history_size: int = 30  # 0.5 seconde à 60 fps
    min_overlap: float = 0.7  # 70% de chevauchement
    min_occurrences: int = 3  # Vu au moins 3 fois
    confidence_threshold: float = 0.5
    decay_rate: float = 0.95  # Décroissance par frame


class TemporalCorrelator:
    """Détecteur de corrélations temporelles entre frames.
    
    Analyse les groupes d'activation sur plusieurs frames pour
    détecter les patterns qui se répètent régulièrement.
    """
    
    def __init__(self, shape: tuple[int, int], config: Optional[CorrelationConfig] = None):
        """Initialise le corrélateur.
        
        Args:
            shape: Dimensions (H, W) des frames à analyser
            config: Configuration optionnelle
        """
        self.shape = shape
        self.config = config or CorrelationConfig()
        
        # Historique des groupes détectés (frames récents)
        self._history: deque[list[ActivationGroup]] = deque(maxlen=self.config.history_size)
        
        # Patterns temporels détectés
        self._patterns: dict[int, TemporalPattern] = {}
        self._next_pattern_id = 0
        
        # Compteur de frames
        self._frame_count = 0
        
        # Statistiques
        self._total_patterns_created = 0
        self._total_patterns_pruned = 0
    
    @property
    def pattern_count(self) -> int:
        """Nombre de patterns actifs."""
        return len(self._patterns)
    
    @property
    def stable_patterns(self) -> list[TemporalPattern]:
        """Retourne les patterns stables (confiance > seuil)."""
        return [
            p for p in self._patterns.values()
            if p.confidence >= self.config.confidence_threshold
            and p.occurrences >= self.config.min_occurrences
        ]
    
    def process_groups(self, groups: list[ActivationGroup]) -> list[TemporalPattern]:
        """Traite un nouveau frame de groupes et détecte les corrélations.
        
        Args:
            groups: Liste des groupes d'activation du frame courant
            
        Returns:
            Liste des patterns mis à jour ou créés
        """
        self._frame_count += 1
        updated_patterns = []
        
        # Pour chaque groupe, chercher un pattern existant qui correspond
        matched_pattern_ids = set()
        
        for group in groups:
            best_match = self._find_matching_pattern(group)
            
            if best_match is not None:
                # Mettre à jour le pattern existant
                self._update_pattern(best_match, group)
                matched_pattern_ids.add(best_match.pattern_id)
                updated_patterns.append(best_match)
            else:
                # Créer un nouveau pattern
                new_pattern = self._create_pattern(group)
                updated_patterns.append(new_pattern)
        
        # Décroître la confiance des patterns non matchés
        self._decay_unmatched_patterns(matched_pattern_ids)
        
        # Ajouter à l'historique
        self._history.append(groups)
        
        # Élaguer les patterns morts (confiance trop basse et anciens)
        self._prune_dead_patterns()
        
        return updated_patterns
    
    def _find_matching_pattern(self, group: ActivationGroup) -> Optional[TemporalPattern]:
        """Trouve le pattern existant le plus similaire au groupe.
        
        Args:
            group: Groupe d'activation à matcher
            
        Returns:
            Pattern correspondant ou None
        """
        best_match = None
        best_overlap = self.config.min_overlap
        
        # Créer le masque du groupe à partir des pixels
        group_mask = self._group_to_mask(group)
        
        for pattern in self._patterns.values():
            overlap = self._compute_overlap(group_mask, pattern.signature)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = pattern
        
        return best_match
    
    def _group_to_mask(self, group: ActivationGroup) -> np.ndarray:
        """Convertit un groupe en masque booléen.
        
        Args:
            group: Groupe d'activation
            
        Returns:
            Masque booléen de la taille de l'image
        """
        mask = np.zeros(self.shape, dtype=bool)
        for y, x in group.pixels:
            if 0 <= y < self.shape[0] and 0 <= x < self.shape[1]:
                mask[y, x] = True
        return mask
    
    def _compute_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calcule le chevauchement (IoU) entre deux masques.
        
        Args:
            mask1: Premier masque binaire
            mask2: Second masque binaire
            
        Returns:
            Coefficient de Jaccard (intersection / union)
        """
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    def _create_pattern(self, group: ActivationGroup) -> TemporalPattern:
        """Crée un nouveau pattern à partir d'un groupe.
        
        Args:
            group: Groupe d'activation source
            
        Returns:
            Nouveau pattern temporel
        """
        # Créer le masque pleine taille à partir des pixels
        signature = self._group_to_mask(group)
        
        pattern = TemporalPattern(
            pattern_id=self._next_pattern_id,
            signature=signature,
            first_seen=self._frame_count,
            last_seen=self._frame_count,
            occurrences=1,
            confidence=0.1  # Confiance initiale faible
        )
        
        self._patterns[pattern.pattern_id] = pattern
        self._next_pattern_id += 1
        self._total_patterns_created += 1
        
        return pattern
    
    def _update_pattern(self, pattern: TemporalPattern, group: ActivationGroup):
        """Met à jour un pattern avec un nouveau groupe correspondant.
        
        Args:
            pattern: Pattern à mettre à jour
            group: Groupe correspondant
        """
        # Créer le masque du groupe à partir des pixels
        group_mask = self._group_to_mask(group)
        
        # Fusionner les signatures (moyenne pondérée vers le nouveau)
        # Plus le pattern est ancien et fréquent, plus il est stable
        alpha = min(0.3, 1.0 / (pattern.occurrences + 1))
        pattern.signature = (
            (1 - alpha) * pattern.signature.astype(float) +
            alpha * group_mask.astype(float)
        ) > 0.5
        
        pattern.last_seen = self._frame_count
        pattern.occurrences += 1
        
        # Augmenter la confiance
        pattern.confidence = min(1.0, pattern.confidence + 0.1)
        
        # Recalculer le centroïde
        if pattern.signature.any():
            ys, xs = np.where(pattern.signature)
            pattern.centroid = (float(np.mean(xs)), float(np.mean(ys)))
    
    def _decay_unmatched_patterns(self, matched_ids: set[int]):
        """Décroît la confiance des patterns non observés.
        
        Args:
            matched_ids: IDs des patterns qui ont été matchés ce frame
        """
        for pattern_id, pattern in self._patterns.items():
            if pattern_id not in matched_ids:
                pattern.confidence *= self.config.decay_rate
    
    def _prune_dead_patterns(self):
        """Supprime les patterns morts (confiance très basse et anciens)."""
        dead_ids = []
        
        for pattern_id, pattern in self._patterns.items():
            # Pattern mort: confiance < 0.01 et pas vu depuis longtemps
            frames_since_seen = self._frame_count - pattern.last_seen
            
            if pattern.confidence < 0.01 and frames_since_seen > self.config.history_size:
                dead_ids.append(pattern_id)
        
        for pattern_id in dead_ids:
            del self._patterns[pattern_id]
            self._total_patterns_pruned += 1
    
    def get_pattern_map(self) -> np.ndarray:
        """Retourne une carte des patterns stables.
        
        Returns:
            Image (H, W) où chaque pixel indique le pattern_id + 1
            (0 = pas de pattern)
        """
        pattern_map = np.zeros(self.shape, dtype=np.int32)
        
        for pattern in self.stable_patterns:
            # Superposer les patterns (les plus récents en dernier)
            pattern_map[pattern.signature] = pattern.pattern_id + 1
        
        return pattern_map
    
    def get_confidence_map(self) -> np.ndarray:
        """Retourne une carte de confiance des patterns.
        
        Returns:
            Image (H, W) avec la confiance maximale à chaque pixel [0-1]
        """
        confidence_map = np.zeros(self.shape, dtype=np.float32)
        
        for pattern in self._patterns.values():
            # Prendre le maximum de confiance à chaque pixel
            confidence_map = np.maximum(
                confidence_map,
                pattern.signature.astype(np.float32) * pattern.confidence
            )
        
        return confidence_map
    
    def reset(self):
        """Réinitialise le corrélateur."""
        self._history.clear()
        self._patterns.clear()
        self._next_pattern_id = 0
        self._frame_count = 0
        self._total_patterns_created = 0
        self._total_patterns_pruned = 0
    
    def get_stats(self) -> dict:
        """Retourne les statistiques du corrélateur.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        return {
            'frame_count': self._frame_count,
            'active_patterns': len(self._patterns),
            'stable_patterns': len(self.stable_patterns),
            'total_created': self._total_patterns_created,
            'total_pruned': self._total_patterns_pruned,
            'history_size': len(self._history),
        }


def visualize_patterns(
    pattern_map: np.ndarray,
    confidence_map: Optional[np.ndarray] = None
) -> np.ndarray:
    """Visualise les patterns temporels avec des couleurs.
    
    Args:
        pattern_map: Carte des pattern_ids (0 = pas de pattern)
        confidence_map: Carte optionnelle de confiance [0-1]
        
    Returns:
        Image RGB (H, W, 3) uint8
    """
    h, w = pattern_map.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Couleurs distinctes pour les patterns
    colors = np.array([
        [255, 100, 100],  # Rouge clair
        [100, 255, 100],  # Vert clair
        [100, 100, 255],  # Bleu clair
        [255, 255, 100],  # Jaune
        [255, 100, 255],  # Magenta
        [100, 255, 255],  # Cyan
        [255, 180, 100],  # Orange
        [180, 100, 255],  # Violet
    ], dtype=np.uint8)
    
    unique_ids = np.unique(pattern_map)
    
    for pattern_id in unique_ids:
        if pattern_id == 0:
            continue
        
        mask = pattern_map == pattern_id
        color = colors[(pattern_id - 1) % len(colors)]
        
        # Appliquer la couleur
        output[mask] = color
    
    # Moduler par la confiance si disponible
    if confidence_map is not None:
        # Convertir confiance en luminosité (0.3 à 1.0)
        brightness = 0.3 + 0.7 * confidence_map
        output = (output * brightness[:, :, np.newaxis]).astype(np.uint8)
    
    return output
