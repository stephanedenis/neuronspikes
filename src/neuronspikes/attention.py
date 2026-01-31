"""
Attention - Système d'attention visuelle avec zoom et mémoire.

Ce module implémente:
- Zoom virtuel (zoom-in/zoom-out) tout en restant dans l'image source
- Mémoire des positions relatives explorées (carte d'exploration)
- Rétroaction inhibitrice (habituation aux zones déjà vues)
- Suivi des points d'intérêt découverts

Inspiré de:
- Système vestibulo-oculaire (VOR) observé chez le chat Peyo (1995)
- Attention sélective et habituation biologique
- Inhibition de retour (IOR - Inhibition of Return)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Deque
from collections import deque
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray

from .fovea import FoveaConfig


class ZoomLevel(Enum):
    """Niveaux de zoom prédéfinis."""
    WIDE = auto()      # Vue large, basse résolution
    NORMAL = auto()    # Vue normale
    CLOSE = auto()     # Vue rapprochée
    DETAIL = auto()    # Vue de détail maximum


@dataclass
class ZoomConfig:
    """Configuration du zoom virtuel.
    
    Attributes:
        min_scale: Échelle minimum (zoom-out max, vue large)
        max_scale: Échelle maximum (zoom-in max, vue détail)
        num_levels: Nombre de niveaux discrets de zoom
        zoom_speed: Vitesse de transition entre niveaux
    """
    min_scale: float = 0.25   # Vue 4× plus large
    max_scale: float = 2.0    # Vue 2× plus proche
    num_levels: int = 8       # Niveaux discrets
    zoom_speed: float = 0.15  # Transition lisse


@dataclass
class AttentionConfig:
    """Configuration du système d'attention.
    
    Attributes:
        memory_size: Nombre de positions mémorisées
        inhibition_decay: Décroissance de l'inhibition (par frame)
        inhibition_radius: Rayon d'effet de l'inhibition (pixels)
        habituation_rate: Vitesse d'habituation aux stimuli répétés
        novelty_threshold: Seuil de nouveauté pour déclencher l'attention
        exploration_weight: Poids de l'exploration vs exploitation
    """
    memory_size: int = 100
    inhibition_decay: float = 0.95
    inhibition_radius: float = 32.0
    habituation_rate: float = 0.1
    novelty_threshold: float = 0.3
    exploration_weight: float = 0.3


@dataclass
class GazeMemory:
    """Mémoire d'une position de regard.
    
    Attributes:
        x, y: Position dans l'image source
        scale: Niveau de zoom lors de l'observation
        frame: Frame de l'observation
        saliency: Score de saillance observé
        correlation: Corrélation stéréo (si applicable)
        visits: Nombre de visites à cette position
    """
    x: float
    y: float
    scale: float
    frame: int
    saliency: float = 0.0
    correlation: float = 0.0
    visits: int = 1
    
    def distance_to(self, other_x: float, other_y: float) -> float:
        """Distance euclidienne vers une autre position."""
        return math.sqrt((self.x - other_x)**2 + (self.y - other_y)**2)


@dataclass
class PointOfInterest:
    """Point d'intérêt découvert.
    
    Attributes:
        x, y: Position
        confidence: Confiance (augmente avec les observations)
        first_seen: Frame de première observation
        last_seen: Frame de dernière observation
        observations: Nombre d'observations
        features: Signature visuelle (activations fovéales moyennes)
    """
    x: float
    y: float
    confidence: float = 0.0
    first_seen: int = 0
    last_seen: int = 0
    observations: int = 0
    features: Optional[NDArray[np.float32]] = None
    
    def update(self, frame: int, new_features: Optional[NDArray[np.float32]] = None):
        """Met à jour le POI avec une nouvelle observation."""
        self.last_seen = frame
        self.observations += 1
        # Confiance augmente avec le logarithme des observations
        self.confidence = min(1.0, 0.3 + 0.7 * math.log1p(self.observations) / 5)
        
        # Moyenne mobile exponentielle des features
        if new_features is not None:
            if self.features is None:
                self.features = new_features.copy()
            else:
                alpha = 0.2
                self.features = alpha * new_features + (1 - alpha) * self.features


class VirtualZoom:
    """Zoom virtuel contraint à l'image source.
    
    Permet de "zoomer" sur une région en ajustant dynamiquement
    le rayon de la fovéa, tout en s'assurant que le cercle
    de vision reste entièrement dans l'image.
    """
    
    def __init__(
        self,
        config: ZoomConfig,
        image_width: int,
        image_height: int,
        base_radius: int = 64
    ):
        """Initialise le zoom virtuel.
        
        Args:
            config: Configuration du zoom
            image_width, image_height: Dimensions de l'image source
            base_radius: Rayon de base de la fovéa (à scale=1.0)
        """
        self.config = config
        self.image_width = image_width
        self.image_height = image_height
        self.base_radius = base_radius
        
        # État courant
        self._current_scale: float = 1.0
        self._target_scale: float = 1.0
        self._level_index: int = config.num_levels // 2  # Niveau normal
        
        # Pré-calculer les échelles discrètes
        self._scales = self._compute_scales()
    
    def _compute_scales(self) -> List[float]:
        """Calcule les échelles discrètes de zoom."""
        cfg = self.config
        scales = []
        for i in range(cfg.num_levels):
            t = i / (cfg.num_levels - 1)
            # Distribution logarithmique pour naturel
            scale = cfg.min_scale * ((cfg.max_scale / cfg.min_scale) ** t)
            scales.append(scale)
        return scales
    
    @property
    def current_scale(self) -> float:
        """Échelle courante."""
        return self._current_scale
    
    @property
    def current_radius(self) -> int:
        """Rayon effectif de la fovéa."""
        return int(self.base_radius / self._current_scale)
    
    @property
    def level_index(self) -> int:
        """Index du niveau de zoom courant."""
        return self._level_index
    
    @property
    def zoom_level(self) -> ZoomLevel:
        """Niveau de zoom symbolique."""
        n = self.config.num_levels
        if self._level_index < n // 4:
            return ZoomLevel.WIDE
        elif self._level_index < n // 2:
            return ZoomLevel.NORMAL
        elif self._level_index < 3 * n // 4:
            return ZoomLevel.CLOSE
        else:
            return ZoomLevel.DETAIL
    
    def zoom_in(self, steps: int = 1):
        """Zoom avant (plus de détails, champ plus étroit)."""
        new_level = min(self._level_index + steps, self.config.num_levels - 1)
        if new_level != self._level_index:
            self._level_index = new_level
            self._target_scale = self._scales[self._level_index]
    
    def zoom_out(self, steps: int = 1):
        """Zoom arrière (moins de détails, champ plus large)."""
        new_level = max(self._level_index - steps, 0)
        if new_level != self._level_index:
            self._level_index = new_level
            self._target_scale = self._scales[self._level_index]
    
    def set_level(self, level: int):
        """Définit le niveau de zoom directement."""
        level = max(0, min(level, self.config.num_levels - 1))
        self._level_index = level
        self._target_scale = self._scales[level]
    
    def update(self) -> float:
        """Met à jour l'échelle (transition lisse).
        
        Returns:
            Nouvelle échelle courante
        """
        if abs(self._current_scale - self._target_scale) > 0.001:
            diff = self._target_scale - self._current_scale
            self._current_scale += diff * self.config.zoom_speed
        else:
            self._current_scale = self._target_scale
        return self._current_scale
    
    def constrain_gaze(self, x: float, y: float) -> Tuple[float, float]:
        """Contraint la position du regard pour rester dans l'image.
        
        Le cercle de vision (rayon = current_radius) doit rester
        entièrement dans les limites de l'image.
        
        Args:
            x, y: Position désirée du regard
            
        Returns:
            Position contrainte (x, y)
        """
        margin = self.current_radius
        
        # Contraindre x
        min_x = margin
        max_x = self.image_width - margin
        x = max(min_x, min(x, max_x))
        
        # Contraindre y
        min_y = margin
        max_y = self.image_height - margin
        y = max(min_y, min(y, max_y))
        
        return (x, y)
    
    def get_effective_fovea_config(self, base_config: FoveaConfig) -> FoveaConfig:
        """Retourne une config de fovéa ajustée au zoom courant.
        
        Args:
            base_config: Configuration de base
            
        Returns:
            Configuration adaptée au niveau de zoom
        """
        scale = self._current_scale
        return FoveaConfig(
            num_rings=base_config.num_rings,
            num_sectors=base_config.num_sectors,
            fovea_radius=max(4, int(base_config.fovea_radius / scale)),
            max_radius=max(8, int(base_config.max_radius / scale)),
            center_resolution=base_config.center_resolution * scale,
            peripheral_falloff=base_config.peripheral_falloff,
        )


class InhibitionMap:
    """Carte d'inhibition de retour (IOR).
    
    Les zones récemment visitées sont inhibées pour encourager
    l'exploration de nouvelles régions.
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        config: AttentionConfig
    ):
        """Initialise la carte d'inhibition.
        
        Args:
            width, height: Dimensions (résolution réduite)
            config: Configuration de l'attention
        """
        self.width = width
        self.height = height
        self.config = config
        
        # Carte d'inhibition (0 = pas d'inhibition, 1 = inhibition max)
        self._map: NDArray[np.float32] = np.zeros((height, width), dtype=np.float32)
        
        # Historique des positions de regard
        self._gaze_history: Deque[Tuple[float, float, int]] = deque(
            maxlen=config.memory_size
        )
    
    @property
    def inhibition_map(self) -> NDArray[np.float32]:
        """Retourne la carte d'inhibition courante."""
        return self._map.copy()
    
    def add_gaze(self, x: float, y: float, frame: int, intensity: float = 1.0):
        """Ajoute une position de regard avec inhibition.
        
        Args:
            x, y: Position en coordonnées image source
            frame: Numéro de frame
            intensity: Intensité de l'inhibition (0-1)
        """
        self._gaze_history.append((x, y, frame))
        
        # Convertir en coordonnées carte (résolution réduite)
        # Supposons que l'image source fait map_scale × plus grand
        # Pour l'instant, on garde les mêmes coordonnées
        map_x = int(x * self.width / (self.width * 8))  # Échelle arbitraire
        map_y = int(y * self.height / (self.height * 8))
        
        # Appliquer l'inhibition gaussienne
        radius = int(self.config.inhibition_radius / 8)  # Échelle réduite
        self._apply_gaussian_inhibition(map_x, map_y, radius, intensity)
    
    def _apply_gaussian_inhibition(
        self,
        cx: int,
        cy: int,
        radius: int,
        intensity: float
    ):
        """Applique une inhibition gaussienne centrée sur (cx, cy)."""
        radius = max(1, radius)
        
        # Créer un masque gaussien
        y_range = np.arange(max(0, cy - radius * 2), min(self.height, cy + radius * 2 + 1))
        x_range = np.arange(max(0, cx - radius * 2), min(self.width, cx + radius * 2 + 1))
        
        if len(x_range) == 0 or len(y_range) == 0:
            return
        
        yy, xx = np.meshgrid(y_range, x_range, indexing='ij')
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        sigma_sq = radius**2
        gaussian = intensity * np.exp(-dist_sq / (2 * sigma_sq))
        
        # Ajouter à la carte (max pour éviter la saturation)
        self._map[y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1] = np.maximum(
            self._map[y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1],
            gaussian.astype(np.float32)
        )
    
    def decay(self):
        """Applique la décroissance temporelle de l'inhibition."""
        self._map *= self.config.inhibition_decay
        # Seuil minimum
        self._map[self._map < 0.01] = 0
    
    def get_inhibition_at(self, x: float, y: float) -> float:
        """Retourne le niveau d'inhibition à une position.
        
        Args:
            x, y: Position en coordonnées image source
            
        Returns:
            Niveau d'inhibition (0-1)
        """
        map_x = int(x * self.width / (self.width * 8))
        map_y = int(y * self.height / (self.height * 8))
        
        map_x = max(0, min(map_x, self.width - 1))
        map_y = max(0, min(map_y, self.height - 1))
        
        return float(self._map[map_y, map_x])
    
    def modulate_saliency(
        self,
        saliency: NDArray[np.float32],
        scale: float = 1.0
    ) -> NDArray[np.float32]:
        """Module une carte de saillance par l'inhibition.
        
        Les zones inhibées voient leur saillance réduite.
        
        Args:
            saliency: Carte de saillance originale
            scale: Facteur d'échelle pour l'inhibition
            
        Returns:
            Saillance modulée
        """
        # Redimensionner l'inhibition à la taille de la saillance
        import cv2
        inhibition_resized = cv2.resize(
            self._map,
            (saliency.shape[1], saliency.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Moduler: saliency * (1 - inhibition * scale)
        modulation = 1.0 - np.clip(inhibition_resized * scale, 0, 0.9)
        return saliency * modulation
    
    def reset(self):
        """Réinitialise la carte d'inhibition."""
        self._map.fill(0)
        self._gaze_history.clear()


class AttentionMemory:
    """Mémoire des positions explorées et points d'intérêt.
    
    Maintient:
    - Historique des positions de regard
    - Points d'intérêt découverts
    - Carte d'exploration (zones visitées/non visitées)
    """
    
    def __init__(self, config: AttentionConfig):
        """Initialise la mémoire d'attention.
        
        Args:
            config: Configuration de l'attention
        """
        self.config = config
        
        # Historique des positions
        self._history: Deque[GazeMemory] = deque(maxlen=config.memory_size)
        
        # Points d'intérêt découverts
        self._pois: List[PointOfInterest] = []
        self._max_pois = 50
        
        # Statistiques
        self._frame_count = 0
        self._total_distance = 0.0
    
    def record_gaze(
        self,
        x: float,
        y: float,
        scale: float,
        saliency: float = 0.0,
        correlation: float = 0.0
    ):
        """Enregistre une position de regard.
        
        Args:
            x, y: Position
            scale: Niveau de zoom
            saliency: Score de saillance observé
            correlation: Corrélation stéréo
        """
        self._frame_count += 1
        
        # Calculer la distance depuis la dernière position
        if self._history:
            last = self._history[-1]
            dist = last.distance_to(x, y)
            self._total_distance += dist
        
        # Vérifier si position déjà visitée (fusionner)
        existing = self._find_nearby_memory(x, y, threshold=16.0)
        if existing:
            existing.visits += 1
            existing.saliency = 0.7 * existing.saliency + 0.3 * saliency
            existing.correlation = 0.7 * existing.correlation + 0.3 * correlation
        else:
            memory = GazeMemory(
                x=x, y=y,
                scale=scale,
                frame=self._frame_count,
                saliency=saliency,
                correlation=correlation
            )
            self._history.append(memory)
    
    def _find_nearby_memory(
        self,
        x: float,
        y: float,
        threshold: float
    ) -> Optional[GazeMemory]:
        """Trouve une mémoire proche de la position donnée."""
        for memory in reversed(self._history):
            if memory.distance_to(x, y) < threshold:
                return memory
        return None
    
    def register_poi(
        self,
        x: float,
        y: float,
        features: Optional[NDArray[np.float32]] = None
    ):
        """Enregistre ou met à jour un point d'intérêt.
        
        Args:
            x, y: Position du POI
            features: Signature visuelle
        """
        # Chercher un POI existant proche
        merge_distance = 24.0
        for poi in self._pois:
            dist = math.sqrt((poi.x - x)**2 + (poi.y - y)**2)
            if dist < merge_distance:
                poi.update(self._frame_count, features)
                return
        
        # Nouveau POI
        if len(self._pois) < self._max_pois:
            poi = PointOfInterest(
                x=x, y=y,
                first_seen=self._frame_count,
                last_seen=self._frame_count,
                observations=1,
                features=features.copy() if features is not None else None
            )
            poi.confidence = 0.3
            self._pois.append(poi)
        else:
            # Remplacer le POI le moins confiant
            min_poi = min(self._pois, key=lambda p: p.confidence)
            if min_poi.confidence < 0.3:
                self._pois.remove(min_poi)
                self.register_poi(x, y, features)
    
    def get_pois(self, min_confidence: float = 0.0) -> List[PointOfInterest]:
        """Retourne les points d'intérêt.
        
        Args:
            min_confidence: Confiance minimum
            
        Returns:
            Liste des POIs filtrés
        """
        return [p for p in self._pois if p.confidence >= min_confidence]
    
    def get_recent_positions(self, n: int = 10) -> List[Tuple[float, float]]:
        """Retourne les n dernières positions de regard."""
        return [(m.x, m.y) for m in list(self._history)[-n:]]
    
    def get_exploration_score(
        self,
        x: float,
        y: float,
        radius: float = 32.0
    ) -> float:
        """Score d'exploration (0 = très visité, 1 = jamais visité).
        
        Args:
            x, y: Position à évaluer
            radius: Rayon de recherche
            
        Returns:
            Score de nouveauté (0-1)
        """
        visits = 0
        for memory in self._history:
            if memory.distance_to(x, y) < radius:
                visits += memory.visits
        
        # Décroissance exponentielle
        return math.exp(-visits * 0.1)
    
    def suggest_exploration_target(
        self,
        current_x: float,
        current_y: float,
        image_width: int,
        image_height: int,
        margin: int = 64
    ) -> Tuple[float, float]:
        """Suggère une cible d'exploration (zone peu visitée).
        
        Args:
            current_x, current_y: Position courante
            image_width, image_height: Dimensions de l'image
            margin: Marge par rapport aux bords
            
        Returns:
            Position suggérée (x, y)
        """
        best_score = -1.0
        best_pos = (current_x, current_y)
        
        # Échantillonner quelques positions
        for _ in range(20):
            x = np.random.uniform(margin, image_width - margin)
            y = np.random.uniform(margin, image_height - margin)
            
            score = self.get_exploration_score(x, y)
            
            # Bonus pour distance modérée (pas trop loin, pas trop près)
            dist = math.sqrt((x - current_x)**2 + (y - current_y)**2)
            distance_bonus = 1.0 - abs(dist - 200) / 400  # Optimal ~200px
            
            combined = score * 0.7 + distance_bonus * 0.3
            
            if combined > best_score:
                best_score = combined
                best_pos = (x, y)
        
        return best_pos
    
    def get_stats(self) -> Dict[str, float]:
        """Retourne les statistiques de la mémoire."""
        return {
            'num_memories': len(self._history),
            'num_pois': len(self._pois),
            'total_distance': self._total_distance,
            'avg_visits': np.mean([m.visits for m in self._history]) if self._history else 0,
            'avg_poi_confidence': np.mean([p.confidence for p in self._pois]) if self._pois else 0,
        }
    
    def reset(self):
        """Réinitialise la mémoire."""
        self._history.clear()
        self._pois.clear()
        self._frame_count = 0
        self._total_distance = 0.0


class AttentionController:
    """Contrôleur d'attention intégré.
    
    Combine:
    - Zoom virtuel adaptatif
    - Inhibition de retour
    - Mémoire des positions
    - Décision de saccade/exploration
    """
    
    def __init__(
        self,
        image_width: int,
        image_height: int,
        base_fovea_config: FoveaConfig,
        zoom_config: Optional[ZoomConfig] = None,
        attention_config: Optional[AttentionConfig] = None
    ):
        """Initialise le contrôleur d'attention.
        
        Args:
            image_width, image_height: Dimensions de l'image source
            base_fovea_config: Configuration de base de la fovéa
            zoom_config: Configuration du zoom
            attention_config: Configuration de l'attention
        """
        self.image_width = image_width
        self.image_height = image_height
        self.base_config = base_fovea_config
        
        zoom_config = zoom_config or ZoomConfig()
        attention_config = attention_config or AttentionConfig()
        
        # Sous-systèmes
        self.zoom = VirtualZoom(
            config=zoom_config,
            image_width=image_width,
            image_height=image_height,
            base_radius=base_fovea_config.max_radius
        )
        
        # Carte d'inhibition (résolution réduite pour efficacité)
        self.inhibition = InhibitionMap(
            width=image_width // 8,
            height=image_height // 8,
            config=attention_config
        )
        
        self.memory = AttentionMemory(attention_config)
        self.attention_config = attention_config
        
        # État courant
        self._current_x: float = image_width / 2
        self._current_y: float = image_height / 2
        self._frame_count: int = 0
        
        # Mode d'attention
        self._exploration_mode: bool = False
        self._frames_since_poi: int = 0
    
    @property
    def current_gaze(self) -> Tuple[float, float]:
        """Position courante du regard."""
        return (self._current_x, self._current_y)
    
    @property
    def current_scale(self) -> float:
        """Échelle de zoom courante."""
        return self.zoom.current_scale
    
    @property
    def current_radius(self) -> int:
        """Rayon effectif de la fovéa."""
        return self.zoom.current_radius
    
    def get_effective_config(self) -> FoveaConfig:
        """Retourne la config de fovéa ajustée au zoom."""
        return self.zoom.get_effective_fovea_config(self.base_config)
    
    def update(
        self,
        saliency: Optional[NDArray[np.float32]] = None,
        correlation: float = 0.0,
        features: Optional[NDArray[np.float32]] = None
    ) -> Tuple[float, float]:
        """Met à jour le système d'attention.
        
        Args:
            saliency: Carte de saillance courante
            correlation: Score de corrélation stéréo
            features: Activations fovéales courantes
            
        Returns:
            Nouvelle position du regard (x, y)
        """
        self._frame_count += 1
        
        # Mettre à jour le zoom (transition lisse)
        self.zoom.update()
        
        # Décroissance de l'inhibition
        self.inhibition.decay()
        
        # Enregistrer la position courante
        current_saliency = 0.0
        if saliency is not None:
            # Saillance au point de regard actuel
            sx = int(self._current_x * saliency.shape[1] / self.image_width)
            sy = int(self._current_y * saliency.shape[0] / self.image_height)
            sx = max(0, min(sx, saliency.shape[1] - 1))
            sy = max(0, min(sy, saliency.shape[0] - 1))
            current_saliency = float(saliency[sy, sx])
        
        self.memory.record_gaze(
            self._current_x, self._current_y,
            scale=self.zoom.current_scale,
            saliency=current_saliency,
            correlation=correlation
        )
        
        # Enregistrer POI si forte saillance et corrélation
        if current_saliency > 0.5 and correlation > 0.4:
            self.memory.register_poi(
                self._current_x, self._current_y,
                features=features
            )
            self._frames_since_poi = 0
        else:
            self._frames_since_poi += 1
        
        # Ajouter inhibition à la position courante
        self.inhibition.add_gaze(
            self._current_x, self._current_y,
            self._frame_count,
            intensity=0.3
        )
        
        return (self._current_x, self._current_y)
    
    def move_to(self, x: float, y: float):
        """Déplace le regard vers une position.
        
        La position est automatiquement contrainte aux limites de l'image.
        """
        x, y = self.zoom.constrain_gaze(x, y)
        self._current_x = x
        self._current_y = y
    
    def select_next_target(
        self,
        saliency: NDArray[np.float32],
        correlation: float
    ) -> Tuple[float, float]:
        """Sélectionne la prochaine cible de regard.
        
        Combine:
        - Pics de saillance (exploitation)
        - Zones non explorées (exploration)
        - Inhibition de retour
        
        Args:
            saliency: Carte de saillance
            correlation: Corrélation stéréo courante
            
        Returns:
            Position cible (x, y)
        """
        # Moduler la saillance par l'inhibition
        modulated = self.inhibition.modulate_saliency(saliency, scale=0.7)
        
        # Décider mode exploration vs exploitation
        exploration_weight = self.attention_config.exploration_weight
        if correlation < 0.3:
            # Faible corrélation → plus d'exploration
            exploration_weight = min(0.8, exploration_weight + 0.3)
        if self._frames_since_poi > 60:
            # Pas de POI depuis longtemps → plus d'exploration
            exploration_weight = min(0.9, exploration_weight + 0.2)
        
        if np.random.random() < exploration_weight:
            # Mode exploration
            target = self.memory.suggest_exploration_target(
                self._current_x, self._current_y,
                self.image_width, self.image_height,
                margin=self.current_radius + 10
            )
        else:
            # Mode exploitation: aller vers le pic de saillance
            peak_y, peak_x = np.unravel_index(np.argmax(modulated), modulated.shape)
            target = (
                peak_x * self.image_width / modulated.shape[1],
                peak_y * self.image_height / modulated.shape[0]
            )
        
        # Contraindre aux limites
        return self.zoom.constrain_gaze(*target)
    
    def should_zoom_in(self, correlation: float, saliency: float) -> bool:
        """Détermine si on devrait zoomer."""
        # Zoom si forte corrélation et saillance
        return correlation > 0.6 and saliency > 0.4
    
    def should_zoom_out(self, correlation: float) -> bool:
        """Détermine si on devrait dézoomer."""
        # Dézoom si faible corrélation ou exploration prolongée
        return correlation < 0.2 or self._frames_since_poi > 100
    
    def auto_zoom(self, correlation: float, saliency: float):
        """Ajuste automatiquement le zoom.
        
        Args:
            correlation: Corrélation stéréo
            saliency: Saillance au point de regard
        """
        if self.should_zoom_in(correlation, saliency):
            self.zoom.zoom_in()
        elif self.should_zoom_out(correlation):
            self.zoom.zoom_out()
    
    def get_stats(self) -> Dict[str, any]:
        """Retourne les statistiques du système."""
        memory_stats = self.memory.get_stats()
        return {
            'frame': self._frame_count,
            'gaze': (self._current_x, self._current_y),
            'zoom_level': self.zoom.level_index,
            'zoom_scale': self.zoom.current_scale,
            'radius': self.zoom.current_radius,
            'exploration_mode': self._exploration_mode,
            'frames_since_poi': self._frames_since_poi,
            **memory_stats
        }
    
    def reset(self):
        """Réinitialise le contrôleur."""
        self._current_x = self.image_width / 2
        self._current_y = self.image_height / 2
        self._frame_count = 0
        self._frames_since_poi = 0
        self._exploration_mode = False
        self.zoom.set_level(self.zoom.config.num_levels // 2)
        self.inhibition.reset()
        self.memory.reset()
