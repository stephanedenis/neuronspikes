"""
ColorFovea - Fovéa avec support couleur (YUV), alpha et détection de mouvement.

Ce module étend la fovéa polaire avec:
- Canaux Luma (Y) et Chroma (U, V) pour la couleur
- Canal Alpha pour les pixels hors image (permet le regard libre)
- Détection de mouvement par flux optique
- Suivi d'objets basé sur le mouvement

Espace colorimétrique YUV:
- Y (Luma): Luminosité (0-255)
- U (Cb): Chrominance bleu-jaune (-128 à 127)
- V (Cr): Chrominance rouge-vert (-128 à 127)

L'alpha permet de positionner l'attention n'importe où, même partiellement
hors de l'image source, en marquant les pixels hors limites.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Deque
from collections import deque
from enum import Enum, auto
import numpy as np
from numpy.typing import NDArray

from .fovea import FoveaConfig, GazePoint, PolarCell, Fovea


class ColorChannel(Enum):
    """Canaux de couleur disponibles."""
    LUMA = auto()      # Y - Luminosité
    CHROMA_U = auto()  # U/Cb - Bleu-Jaune
    CHROMA_V = auto()  # V/Cr - Rouge-Vert
    ALPHA = auto()     # Masque de validité
    MOTION = auto()    # Magnitude du mouvement
    MOTION_DIR = auto()  # Direction du mouvement


@dataclass
class ColorFoveaConfig(FoveaConfig):
    """Configuration étendue pour la fovéa couleur.
    
    Attributes:
        use_color: Activer les canaux chroma (sinon grayscale uniquement)
        motion_history: Nombre de frames pour l'historique de mouvement
        motion_threshold: Seuil de détection de mouvement
        alpha_padding: Marge de padding alpha autour de l'image
    """
    use_color: bool = True
    motion_history: int = 5
    motion_threshold: float = 2.0
    alpha_padding: int = 0  # Pixels de marge hors image autorisés


@dataclass
class MotionVector:
    """Vecteur de mouvement pour une cellule.
    
    Attributes:
        dx, dy: Composantes du mouvement (pixels/frame)
        magnitude: Vitesse du mouvement
        direction: Direction en radians
        confidence: Confiance de la mesure (0-1)
    """
    dx: float = 0.0
    dy: float = 0.0
    
    @property
    def magnitude(self) -> float:
        return math.sqrt(self.dx**2 + self.dy**2)
    
    @property
    def direction(self) -> float:
        return math.atan2(self.dy, self.dx)
    
    @property
    def is_moving(self) -> bool:
        return self.magnitude > 0.5


@dataclass
class TrackedObject:
    """Objet en mouvement suivi.
    
    Attributes:
        id: Identifiant unique
        x, y: Position courante (centre)
        vx, vy: Vitesse (pixels/frame)
        size: Taille estimée (rayon)
        color_signature: Signature couleur (Y, U, V moyens)
        age: Nombre de frames depuis création
        last_seen: Frame de dernière observation
        confidence: Confiance du tracking (0-1)
    """
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    size: float = 20.0
    color_signature: Tuple[float, float, float] = (128.0, 0.0, 0.0)
    age: int = 0
    last_seen: int = 0
    confidence: float = 0.5
    
    def predict_position(self, dt: float = 1.0) -> Tuple[float, float]:
        """Prédit la position future."""
        return (self.x + self.vx * dt, self.y + self.vy * dt)
    
    def update(self, x: float, y: float, frame: int):
        """Met à jour la position et vitesse."""
        # Filtre de Kalman simplifié (moyenne mobile)
        alpha = 0.3
        self.vx = alpha * (x - self.x) + (1 - alpha) * self.vx
        self.vy = alpha * (y - self.y) + (1 - alpha) * self.vy
        self.x = x
        self.y = y
        self.age += 1
        self.last_seen = frame
        self.confidence = min(1.0, self.confidence + 0.1)
    
    def decay(self, frame: int):
        """Décroissance si non observé."""
        frames_since = frame - self.last_seen
        if frames_since > 0:
            self.confidence *= 0.9 ** frames_since
            # Prédire la position
            self.x += self.vx * frames_since
            self.y += self.vy * frames_since


class ColorFovea(Fovea):
    """Fovéa étendue avec couleur, alpha et mouvement.
    
    Hérite de Fovea et ajoute:
    - Échantillonnage en espace YUV (luma + chroma)
    - Canal alpha pour pixels hors limites
    - Détection de mouvement par différence temporelle
    - Flux optique simplifié pour direction du mouvement
    """
    
    def __init__(self, config: Optional[ColorFoveaConfig] = None):
        """Initialise la fovéa couleur.
        
        Args:
            config: Configuration étendue
        """
        if config is None:
            config = ColorFoveaConfig()
        
        # Initialiser la classe parente
        super().__init__(config)
        self.color_config = config
        
        # Buffers d'activation étendus
        shape = (self.config.num_rings, self.config.num_sectors)
        self._luma: NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
        self._chroma_u: NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
        self._chroma_v: NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
        self._alpha: NDArray[np.float32] = np.ones(shape, dtype=np.float32)
        self._motion_mag: NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
        self._motion_dir: NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
        
        # Historique pour détection de mouvement
        self._luma_history: Deque[NDArray[np.float32]] = deque(
            maxlen=config.motion_history
        )
        
        # Frame courante
        self._current_frame: int = 0
    
    @property
    def luma(self) -> NDArray[np.float32]:
        """Canal luminosité (Y)."""
        return self._luma.copy()
    
    @property
    def chroma_u(self) -> NDArray[np.float32]:
        """Canal chrominance bleu-jaune (U/Cb)."""
        return self._chroma_u.copy()
    
    @property
    def chroma_v(self) -> NDArray[np.float32]:
        """Canal chrominance rouge-vert (V/Cr)."""
        return self._chroma_v.copy()
    
    @property
    def alpha(self) -> NDArray[np.float32]:
        """Canal alpha (1 = valide, 0 = hors image)."""
        return self._alpha.copy()
    
    @property
    def motion_magnitude(self) -> NDArray[np.float32]:
        """Magnitude du mouvement par cellule."""
        return self._motion_mag.copy()
    
    @property
    def motion_direction(self) -> NDArray[np.float32]:
        """Direction du mouvement par cellule (radians)."""
        return self._motion_dir.copy()
    
    def sample_color(
        self,
        image: NDArray[np.uint8],
        compute_motion: bool = True
    ) -> Dict[str, NDArray[np.float32]]:
        """Échantillonne l'image en couleur avec alpha et mouvement.
        
        Args:
            image: Image BGR (H×W×3) ou grayscale (H×W)
            compute_motion: Calculer le mouvement
            
        Returns:
            Dictionnaire avec tous les canaux:
            - 'luma': Luminosité (num_rings × num_sectors)
            - 'chroma_u': Chrominance U (num_rings × num_sectors)
            - 'chroma_v': Chrominance V (num_rings × num_sectors)
            - 'alpha': Masque de validité (num_rings × num_sectors)
            - 'motion_mag': Magnitude mouvement (num_rings × num_sectors)
            - 'motion_dir': Direction mouvement (num_rings × num_sectors)
        """
        self._current_frame += 1
        
        # Convertir en YUV si couleur
        if len(image.shape) == 3 and image.shape[2] == 3:
            yuv = self._bgr_to_yuv(image)
            y_channel = yuv[:, :, 0]
            u_channel = yuv[:, :, 1].astype(np.float32) - 128  # Centrer sur 0
            v_channel = yuv[:, :, 2].astype(np.float32) - 128
        else:
            y_channel = image if len(image.shape) == 2 else image[:, :, 0]
            u_channel = np.zeros_like(y_channel, dtype=np.float32)
            v_channel = np.zeros_like(y_channel, dtype=np.float32)
        
        h, w = y_channel.shape
        cx, cy = self.gaze.x, self.gaze.y
        theta = self.gaze.theta
        
        # Échantillonner chaque cellule
        for ring_idx, ring in enumerate(self.cells):
            for sector_idx, cell in enumerate(ring):
                # Échantillonner avec alpha
                luma, u, v, alpha = self._sample_cell_color(
                    y_channel, u_channel, v_channel,
                    cell, cx, cy, theta, w, h
                )
                self._luma[ring_idx, sector_idx] = luma
                self._chroma_u[ring_idx, sector_idx] = u
                self._chroma_v[ring_idx, sector_idx] = v
                self._alpha[ring_idx, sector_idx] = alpha
        
        # Calculer le mouvement
        if compute_motion and len(self._luma_history) > 0:
            self._compute_motion()
        
        # Ajouter à l'historique
        self._luma_history.append(self._luma.copy())
        
        return {
            'luma': self._luma.copy(),
            'chroma_u': self._chroma_u.copy(),
            'chroma_v': self._chroma_v.copy(),
            'alpha': self._alpha.copy(),
            'motion_mag': self._motion_mag.copy(),
            'motion_dir': self._motion_dir.copy(),
        }
    
    def _bgr_to_yuv(self, bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Convertit BGR vers YUV."""
        import cv2
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    
    def _sample_cell_color(
        self,
        y_channel: NDArray,
        u_channel: NDArray,
        v_channel: NDArray,
        cell: PolarCell,
        cx: float,
        cy: float,
        theta: float,
        w: int,
        h: int
    ) -> Tuple[float, float, float, float]:
        """Échantillonne une cellule avec couleur et alpha.
        
        Returns:
            (luma, chroma_u, chroma_v, alpha)
        """
        # Nombre de points d'échantillonnage proportionnel à l'aire
        num_samples = max(1, int(cell.area / 4))
        
        luma_sum = 0.0
        u_sum = 0.0
        v_sum = 0.0
        valid_samples = 0
        total_samples = 0
        
        # Échantillonner dans la cellule
        for i in range(num_samples):
            t_r = (i + 0.5) / num_samples
            t_a = (i % 3 + 0.5) / 3
            
            r = cell.inner_radius + t_r * (cell.outer_radius - cell.inner_radius)
            angle = cell.start_angle + t_a * (cell.end_angle - cell.start_angle) + theta
            
            # Convertir en coordonnées image
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            
            total_samples += 1
            
            # Vérifier les limites (avec marge alpha)
            padding = self.color_config.alpha_padding
            if -padding <= x < w + padding and -padding <= y < h + padding:
                if 0 <= x < w and 0 <= y < h:
                    # Pixel valide
                    luma_sum += float(y_channel[y, x])
                    u_sum += float(u_channel[y, x])
                    v_sum += float(v_channel[y, x])
                    valid_samples += 1
                # else: pixel dans la marge (alpha partiel)
        
        # Calculer les moyennes
        if valid_samples > 0:
            luma = luma_sum / valid_samples
            u = u_sum / valid_samples
            v = v_sum / valid_samples
        else:
            luma = 0.0
            u = 0.0
            v = 0.0
        
        # Alpha = proportion de samples valides
        alpha = valid_samples / total_samples if total_samples > 0 else 0.0
        
        return (luma, u, v, alpha)
    
    def _compute_motion(self):
        """Calcule le mouvement par différence temporelle et flux optique simplifié."""
        if len(self._luma_history) < 2:
            return
        
        prev_luma = self._luma_history[-1]
        curr_luma = self._luma
        
        # Différence temporelle (magnitude)
        diff = np.abs(curr_luma - prev_luma)
        self._motion_mag = diff
        
        # Flux optique simplifié par corrélation de phase
        # Pour chaque cellule, chercher le décalage optimal
        num_rings, num_sectors = curr_luma.shape
        
        for ring_idx in range(num_rings):
            for sector_idx in range(num_sectors):
                if diff[ring_idx, sector_idx] > self.color_config.motion_threshold:
                    # Chercher la meilleure correspondance dans le voisinage
                    best_shift = 0
                    best_corr = -1
                    curr_val = curr_luma[ring_idx, sector_idx]
                    
                    for shift in range(-2, 3):
                        neighbor_idx = (sector_idx + shift) % num_sectors
                        prev_val = prev_luma[ring_idx, neighbor_idx]
                        corr = 1.0 / (1.0 + abs(curr_val - prev_val))
                        if corr > best_corr:
                            best_corr = corr
                            best_shift = shift
                    
                    # Direction basée sur le décalage sectoriel
                    if best_shift != 0:
                        sector_angle = 2 * math.pi / num_sectors
                        self._motion_dir[ring_idx, sector_idx] = best_shift * sector_angle
                else:
                    self._motion_dir[ring_idx, sector_idx] = 0.0
    
    def get_motion_vectors(self) -> List[MotionVector]:
        """Retourne les vecteurs de mouvement par cellule.
        
        Returns:
            Liste de MotionVector pour les cellules en mouvement
        """
        vectors = []
        num_rings, num_sectors = self._motion_mag.shape
        
        for ring_idx in range(num_rings):
            for sector_idx in range(num_sectors):
                mag = self._motion_mag[ring_idx, sector_idx]
                if mag > self.color_config.motion_threshold:
                    direction = self._motion_dir[ring_idx, sector_idx]
                    vectors.append(MotionVector(
                        dx=mag * math.cos(direction),
                        dy=mag * math.sin(direction)
                    ))
        
        return vectors
    
    def get_color_signature(self) -> Tuple[float, float, float]:
        """Retourne la signature couleur moyenne de la fovéa.
        
        Returns:
            (Y_moyen, U_moyen, V_moyen)
        """
        # Pondérer par alpha (ignorer pixels hors image)
        weights = self._alpha
        total_weight = np.sum(weights) + 1e-6
        
        y_mean = np.sum(self._luma * weights) / total_weight
        u_mean = np.sum(self._chroma_u * weights) / total_weight
        v_mean = np.sum(self._chroma_v * weights) / total_weight
        
        return (float(y_mean), float(u_mean), float(v_mean))
    
    def get_dominant_motion(self) -> Optional[MotionVector]:
        """Retourne le mouvement dominant global.
        
        Returns:
            MotionVector dominant ou None si pas de mouvement
        """
        # Seuiller les cellules en mouvement
        mask = self._motion_mag > self.color_config.motion_threshold
        if not np.any(mask):
            return None
        
        # Moyenne pondérée par magnitude
        weights = self._motion_mag * mask
        total_weight = np.sum(weights) + 1e-6
        
        # Convertir directions en vecteurs et moyenner
        dx_sum = np.sum(weights * np.cos(self._motion_dir))
        dy_sum = np.sum(weights * np.sin(self._motion_dir))
        
        avg_mag = np.sum(self._motion_mag * mask) / (np.sum(mask) + 1e-6)
        
        return MotionVector(
            dx=dx_sum / total_weight * avg_mag,
            dy=dy_sum / total_weight * avg_mag
        )


class ObjectTracker:
    """Traqueur d'objets basé sur le mouvement et la couleur.
    
    Détecte et suit les objets en mouvement en utilisant:
    - Segmentation par mouvement (cellules avec même direction)
    - Signature couleur pour identification
    - Prédiction de trajectoire
    """
    
    def __init__(
        self,
        max_objects: int = 10,
        min_confidence: float = 0.3,
        merge_distance: float = 30.0
    ):
        """Initialise le traqueur.
        
        Args:
            max_objects: Nombre maximum d'objets suivis
            min_confidence: Confiance minimum pour garder un objet
            merge_distance: Distance pour fusionner les détections
        """
        self.max_objects = max_objects
        self.min_confidence = min_confidence
        self.merge_distance = merge_distance
        
        self._objects: Dict[int, TrackedObject] = {}
        self._next_id: int = 0
        self._current_frame: int = 0
    
    @property
    def objects(self) -> List[TrackedObject]:
        """Liste des objets actuellement suivis."""
        return list(self._objects.values())
    
    def update(
        self,
        motion_mag: NDArray[np.float32],
        motion_dir: NDArray[np.float32],
        color_sig: Tuple[float, float, float],
        gaze_x: float,
        gaze_y: float,
        cell_positions: List[Tuple[float, float]],
        motion_threshold: float = 2.0
    ) -> List[TrackedObject]:
        """Met à jour le tracking avec les nouvelles observations.
        
        Args:
            motion_mag: Magnitudes de mouvement par cellule
            motion_dir: Directions de mouvement par cellule
            color_sig: Signature couleur de la fovéa
            gaze_x, gaze_y: Position du regard
            cell_positions: Positions cartésiennes des cellules
            motion_threshold: Seuil de mouvement
            
        Returns:
            Liste des objets mis à jour
        """
        self._current_frame += 1
        
        # Décroître tous les objets existants
        for obj in self._objects.values():
            obj.decay(self._current_frame)
        
        # Détecter les clusters de mouvement
        clusters = self._detect_motion_clusters(
            motion_mag, motion_dir, cell_positions,
            gaze_x, gaze_y, motion_threshold
        )
        
        # Associer les clusters aux objets existants
        for cluster in clusters:
            matched = self._match_to_existing(cluster, color_sig)
            if matched is None:
                # Nouvel objet
                if len(self._objects) < self.max_objects:
                    self._create_object(cluster, color_sig)
        
        # Purger les objets perdus
        self._purge_lost_objects()
        
        return self.objects
    
    def _detect_motion_clusters(
        self,
        motion_mag: NDArray[np.float32],
        motion_dir: NDArray[np.float32],
        cell_positions: List[Tuple[float, float]],
        gaze_x: float,
        gaze_y: float,
        threshold: float
    ) -> List[Dict]:
        """Détecte les clusters de cellules en mouvement cohérent.
        
        Returns:
            Liste de clusters avec position et direction moyennes
        """
        # Trouver les cellules en mouvement
        mask = motion_mag > threshold
        if not np.any(mask):
            return []
        
        # Pour simplifier, on considère un seul cluster centré
        # Une version plus avancée ferait du clustering spatial
        moving_indices = np.where(mask.flatten())[0]
        
        if len(moving_indices) == 0:
            return []
        
        # Calculer le centroïde
        x_sum, y_sum = 0.0, 0.0
        dx_sum, dy_sum = 0.0, 0.0
        count = 0
        
        num_rings, num_sectors = motion_mag.shape
        for idx in moving_indices:
            ring = idx // num_sectors
            sector = idx % num_sectors
            
            # Position de la cellule
            if idx < len(cell_positions):
                cx, cy = cell_positions[idx]
            else:
                # Calculer approximativement
                sector_angle = 2 * math.pi / num_sectors
                cell_angle = sector * sector_angle
                cell_radius = (ring + 0.5) / num_rings * 64  # Approximation
                cx = gaze_x + cell_radius * math.cos(cell_angle)
                cy = gaze_y + cell_radius * math.sin(cell_angle)
            
            mag = motion_mag[ring, sector]
            direction = motion_dir[ring, sector]
            
            x_sum += cx * mag
            y_sum += cy * mag
            dx_sum += mag * math.cos(direction)
            dy_sum += mag * math.sin(direction)
            count += mag
        
        if count < 1e-6:
            return []
        
        return [{
            'x': x_sum / count,
            'y': y_sum / count,
            'dx': dx_sum / count,
            'dy': dy_sum / count,
            'size': len(moving_indices) * 5,  # Taille approximative
        }]
    
    def _match_to_existing(
        self,
        cluster: Dict,
        color_sig: Tuple[float, float, float]
    ) -> Optional[TrackedObject]:
        """Associe un cluster à un objet existant.
        
        Returns:
            L'objet associé ou None
        """
        best_match = None
        best_dist = float('inf')
        
        for obj in self._objects.values():
            # Prédire la position
            pred_x, pred_y = obj.predict_position()
            
            # Distance
            dist = math.sqrt(
                (cluster['x'] - pred_x)**2 + 
                (cluster['y'] - pred_y)**2
            )
            
            if dist < self.merge_distance and dist < best_dist:
                best_dist = dist
                best_match = obj
        
        if best_match is not None:
            best_match.update(
                cluster['x'], cluster['y'],
                self._current_frame
            )
            best_match.vx = 0.7 * best_match.vx + 0.3 * cluster['dx']
            best_match.vy = 0.7 * best_match.vy + 0.3 * cluster['dy']
            best_match.color_signature = (
                0.8 * best_match.color_signature[0] + 0.2 * color_sig[0],
                0.8 * best_match.color_signature[1] + 0.2 * color_sig[1],
                0.8 * best_match.color_signature[2] + 0.2 * color_sig[2],
            )
        
        return best_match
    
    def _create_object(
        self,
        cluster: Dict,
        color_sig: Tuple[float, float, float]
    ):
        """Crée un nouvel objet suivi."""
        obj = TrackedObject(
            id=self._next_id,
            x=cluster['x'],
            y=cluster['y'],
            vx=cluster['dx'],
            vy=cluster['dy'],
            size=cluster.get('size', 20),
            color_signature=color_sig,
            last_seen=self._current_frame
        )
        self._objects[self._next_id] = obj
        self._next_id += 1
    
    def _purge_lost_objects(self):
        """Supprime les objets perdus."""
        to_remove = []
        for obj_id, obj in self._objects.items():
            if obj.confidence < self.min_confidence:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self._objects[obj_id]
    
    def get_strongest_object(self) -> Optional[TrackedObject]:
        """Retourne l'objet avec la plus haute confiance."""
        if not self._objects:
            return None
        return max(self._objects.values(), key=lambda o: o.confidence)
    
    def get_fastest_object(self) -> Optional[TrackedObject]:
        """Retourne l'objet le plus rapide."""
        if not self._objects:
            return None
        return max(
            self._objects.values(),
            key=lambda o: math.sqrt(o.vx**2 + o.vy**2)
        )
    
    def reset(self):
        """Réinitialise le tracking."""
        self._objects.clear()
        self._next_id = 0
        self._current_frame = 0


def visualize_color_fovea(
    fovea_data: Dict[str, NDArray[np.float32]],
    size: int = 256,
    show_motion: bool = True
) -> NDArray[np.uint8]:
    """Visualise les données de la fovéa couleur.
    
    Args:
        fovea_data: Dictionnaire des canaux (luma, chroma_u, chroma_v, alpha, motion)
        size: Taille de l'image de sortie
        show_motion: Afficher les vecteurs de mouvement
        
    Returns:
        Image BGR de visualisation
    """
    import cv2
    
    # Reconstruire YUV
    luma = fovea_data['luma']
    u = fovea_data['chroma_u'] + 128  # Recentrer
    v = fovea_data['chroma_v'] + 128
    alpha = fovea_data['alpha']
    
    num_rings, num_sectors = luma.shape
    
    # Créer l'image polaire
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    max_radius = size // 2 - 10
    
    sector_angle = 2 * math.pi / num_sectors
    
    for ring_idx in range(num_rings):
        inner_r = int(ring_idx / num_rings * max_radius)
        outer_r = int((ring_idx + 1) / num_rings * max_radius)
        
        for sector_idx in range(num_sectors):
            # Couleur YUV → BGR
            y_val = int(luma[ring_idx, sector_idx])
            u_val = int(u[ring_idx, sector_idx])
            v_val = int(v[ring_idx, sector_idx])
            a_val = alpha[ring_idx, sector_idx]
            
            # Convertir YUV vers BGR (approximation)
            # B = Y + 1.773 * (U - 128)
            # G = Y - 0.344 * (U - 128) - 0.714 * (V - 128)  
            # R = Y + 1.403 * (V - 128)
            b = int(np.clip(y_val + 1.773 * (u_val - 128), 0, 255))
            g = int(np.clip(y_val - 0.344 * (u_val - 128) - 0.714 * (v_val - 128), 0, 255))
            r = int(np.clip(y_val + 1.403 * (v_val - 128), 0, 255))
            
            # Appliquer alpha (assombrir les pixels hors image)
            b = int(b * a_val)
            g = int(g * a_val)
            r = int(r * a_val)
            
            # Dessiner le secteur
            start_angle = int(np.degrees(sector_idx * sector_angle))
            end_angle = int(np.degrees((sector_idx + 1) * sector_angle))
            
            cv2.ellipse(
                img, (center, center),
                (outer_r, outer_r),
                0, start_angle, end_angle,
                (b, g, r), -1
            )
            
            if inner_r > 0:
                cv2.ellipse(
                    img, (center, center),
                    (inner_r, inner_r),
                    0, start_angle, end_angle,
                    (0, 0, 0), -1
                )
    
    # Dessiner les vecteurs de mouvement
    if show_motion and 'motion_mag' in fovea_data:
        motion_mag = fovea_data['motion_mag']
        motion_dir = fovea_data['motion_dir']
        
        for ring_idx in range(num_rings):
            mid_r = ((ring_idx + 0.5) / num_rings) * max_radius
            
            for sector_idx in range(num_sectors):
                mag = motion_mag[ring_idx, sector_idx]
                if mag > 2.0:  # Seuil
                    direction = motion_dir[ring_idx, sector_idx]
                    mid_angle = (sector_idx + 0.5) * sector_angle
                    
                    # Point de départ
                    sx = int(center + mid_r * math.cos(mid_angle))
                    sy = int(center + mid_r * math.sin(mid_angle))
                    
                    # Point d'arrivée (longueur proportionnelle à la magnitude)
                    arrow_len = min(20, mag * 3)
                    ex = int(sx + arrow_len * math.cos(mid_angle + direction))
                    ey = int(sy + arrow_len * math.sin(mid_angle + direction))
                    
                    cv2.arrowedLine(img, (sx, sy), (ex, ey), (0, 255, 255), 1)
    
    return img
