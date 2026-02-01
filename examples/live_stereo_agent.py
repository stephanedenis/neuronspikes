#!/usr/bin/env python3
"""
Live Stereo Agent - Yeux virtuels autonomes avec attention.

Utilise la résolution maximale de la caméra stéréo (2560x720)
et simule des fovéas mobiles qui cherchent activement les
détails communs entre les deux yeux.

Comportements agentifs:
1. Saillance: Détection des zones à fort contraste
2. Corrélation: Recherche des patterns communs gauche/droite
3. Saccades: Mouvements rapides vers les zones d'intérêt
4. Poursuite: Suivi lent des objets en mouvement
5. VOR: Compensation de la rotation (stabilisation)

Usage:
    python examples/live_stereo_agent.py [options]
    
Options:
    -c, --camera INDEX    Index de la caméra stéréo (défaut: 1)
    -r, --rings N         Nombre d'anneaux de la fovéa (défaut: 24)
    -s, --sectors N       Nombre de secteurs angulaires (défaut: 16)

Touches:
    q, ESC  Quitter
    SPACE   Pause/Reprendre
    a       Toggle mode autonome / manuel
    r       Reset des fovéas au centre
    d       Mode disparité / corrélation
    s       Forcer une saccade aléatoire
"""

import argparse
import time
import sys
import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

# Ajouter le chemin du projet
sys.path.insert(0, str(__file__).rsplit('/', 2)[0] + '/src')

from neuronspikes import (
    Fovea,
    FoveaConfig,
    visualize_fovea,
    get_opencl_backend,
    is_opencl_available,
    # Système d'attention avec zoom et mémoire
    AttentionController,
    AttentionConfig,
    ZoomConfig,
    # Fovéa couleur avec mouvement
    ColorFovea,
    ColorFoveaConfig,
    # Genèse dynamique de neurones
    NeuronStack,
    GenesisConfig,
    NeuronConfig,
)


@dataclass
class SaliencyPoint:
    """Point saillant détecté."""
    x: float
    y: float
    strength: float  # Force de la saillance (0-1)
    age: int = 0  # Nombre de frames depuis détection
    
    def decay(self, factor: float = 0.95):
        """Décroissance de la saillance avec le temps."""
        self.strength *= factor
        self.age += 1


@dataclass 
class GazeController:
    """Contrôleur de regard avec comportement agentif.
    
    Implémente:
    - Saccades: Mouvements balistiques rapides
    - Poursuite: Suivi lent et fluide
    - Fixation: Maintien sur un point d'intérêt
    """
    x: float
    y: float
    target_x: float = 0.0
    target_y: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    
    # Limites du champ visuel
    min_x: float = 50.0
    max_x: float = 1230.0
    min_y: float = 50.0
    max_y: float = 670.0
    
    # Paramètres de mouvement
    saccade_speed: float = 150.0  # pixels/frame pendant saccade (rapide!)
    pursuit_gain: float = 0.15    # Gain de poursuite lente
    fixation_radius: float = 8.0  # Rayon de fixation (plus précis)
    
    # État
    in_saccade: bool = False
    fixation_time: int = 0  # Frames en fixation
    
    def saccade_to(self, x: float, y: float):
        """Déclenche une saccade vers la cible."""
        self.target_x = np.clip(x, self.min_x, self.max_x)
        self.target_y = np.clip(y, self.min_y, self.max_y)
        self.in_saccade = True
        self.fixation_time = 0
    
    def pursue(self, x: float, y: float):
        """Poursuite lente vers la cible."""
        self.target_x = np.clip(x, self.min_x, self.max_x)
        self.target_y = np.clip(y, self.min_y, self.max_y)
        self.in_saccade = False
    
    def update(self) -> Tuple[float, float]:
        """Met à jour la position du regard.
        
        Returns:
            Nouvelle position (x, y)
        """
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if self.in_saccade:
            # Mouvement balistique rapide
            if distance > self.fixation_radius:
                # Normaliser et appliquer vitesse de saccade
                speed = min(self.saccade_speed, distance)
                self.x += (dx / distance) * speed
                self.y += (dy / distance) * speed
            else:
                # Fin de saccade
                self.in_saccade = False
                self.x = self.target_x
                self.y = self.target_y
        else:
            # Poursuite lente ou fixation
            if distance > self.fixation_radius:
                self.x += dx * self.pursuit_gain
                self.y += dy * self.pursuit_gain
                self.fixation_time = 0
            else:
                self.fixation_time += 1
        
        # Borner la position
        self.x = np.clip(self.x, self.min_x, self.max_x)
        self.y = np.clip(self.y, self.min_y, self.max_y)
        
        return self.x, self.y
    
    @property
    def is_fixating(self) -> bool:
        """Retourne True si en fixation stable."""
        return self.fixation_time > 5 and not self.in_saccade


class AttentionAgent:
    """Agent d'attention visuelle binoculaire.
    
    Cherche activement les détails communs entre les deux yeux.
    Utilise OpenCL pour accélérer le traitement GPU si disponible.
    Intègre zoom virtuel, mémoire des positions et rétroaction inhibitrice.
    Supporte le mode couleur avec ColorFovea pour détection de mouvement.
    """
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fovea_config: FoveaConfig,
        use_opencl: bool = True,
        use_color: bool = False,
    ):
        self.width = frame_width
        self.height = frame_height
        self.config = fovea_config
        self.use_color = use_color
        
        # Backend OpenCL (optionnel)
        self.opencl = None
        if use_opencl and is_opencl_available():
            try:
                self.opencl = get_opencl_backend(prefer_amd=True, verbose=True)
                # Pré-calculer les paramètres des cellules pour OpenCL
                self._build_cell_params()
            except Exception as e:
                print(f"OpenCL désactivé: {e}")
                self.opencl = None
        
        # Système d'attention avec zoom et mémoire
        zoom_cfg = ZoomConfig(
            min_scale=0.25,   # Vue 4× plus large
            max_scale=2.0,    # Vue 2× plus proche
            num_levels=8,
            zoom_speed=0.1
        )
        attention_cfg = AttentionConfig(
            memory_size=100,
            inhibition_decay=0.95,
            inhibition_radius=48.0,
            habituation_rate=0.1,
            exploration_weight=0.3
        )
        self.attention = AttentionController(
            image_width=frame_width,
            image_height=frame_height,
            base_fovea_config=fovea_config,
            zoom_config=zoom_cfg,
            attention_config=attention_cfg
        )
        
        # Deux fovéas indépendantes (couleur ou grayscale)
        if use_color:
            color_config = ColorFoveaConfig(
                num_rings=fovea_config.num_rings,
                num_sectors=fovea_config.num_sectors,
                fovea_radius=fovea_config.fovea_radius,
                max_radius=fovea_config.max_radius,
                use_color=True,
                motion_history=5,
                motion_threshold=2.0,
            )
            self.left_fovea = ColorFovea(color_config)
            self.right_fovea = ColorFovea(color_config)
            self.color_config = color_config
        else:
            self.left_fovea = Fovea(fovea_config)
            self.right_fovea = Fovea(fovea_config)
            self.color_config = None
        
        # Contrôleur de regard (commun aux deux yeux, légère vergence)
        center_x = frame_width // 2
        center_y = frame_height // 2
        self.gaze = GazeController(
            x=center_x, y=center_y,
            target_x=center_x, target_y=center_y,
            min_x=fovea_config.max_radius + 10,
            max_x=frame_width - fovea_config.max_radius - 10,
            min_y=fovea_config.max_radius + 10,
            max_y=frame_height - fovea_config.max_radius - 10,
        )
        
        # Carte de saillance
        self.saliency_points: List[SaliencyPoint] = []
        self.max_saliency_points = 20
        
        # Historique pour détection de mouvement
        self.prev_left_gray: Optional[np.ndarray] = None
        self.prev_right_gray: Optional[np.ndarray] = None
        
        # Activations précédentes pour corrélation temporelle
        self.prev_left_act: Optional[np.ndarray] = None
        self.prev_right_act: Optional[np.ndarray] = None
        
        # Vergence dynamique (convergence horizontale des yeux)
        self.vergence_offset: float = 5.0  # Offset initial en pixels
        self.vergence_velocity: float = 0.0  # Vélocité pour lissage
        self.vergence_min: float = -30.0   # Vergence max divergente
        self.vergence_max: float = 50.0    # Vergence max convergente (objets proches)
        self.vergence_speed: float = 2.0   # Vitesse d'ajustement pixels/frame
        self.vergence_search_range: int = 10  # Plage de recherche pour optimisation
        
        # État
        self.autonomous = True
        self.frame_count = 0
        self.last_saccade_frame = 0
        self.min_saccade_interval = 8  # Frames minimum entre saccades (plus réactif)
        
        # Statistiques
        self.correlation_history: List[float] = []
        self.gpu_time_ms: float = 0.0
        self.vergence_history: List[float] = []
        
        # Pile de neurones pour genèse dynamique
        # Entrée: activations de la fovéa (rings × sectors)
        fovea_shape = (fovea_config.num_rings, fovea_config.num_sectors)
        genesis_config = GenesisConfig(
            min_pattern_confidence=0.5,      # Plus permissif (était 0.7)
            min_pattern_occurrences=5,       # Plus rapide (était 10)
            max_neurons_per_layer=200,
            neuron_merge_threshold=0.5,      # Moins de fusion (était 0.6)
        )
        neuron_config = NeuronConfig(
            threshold=0.4,                   # Plus sensible (était 0.6)
            decay_rate=0.1,                  # Décroissance plus rapide (était 0.05)
            refractory_period=2,             # Plus court (était 3)
        )
        self.neuron_stack = NeuronStack(
            base_shape=fovea_shape,
            num_layers=4,
            config=genesis_config,
            neuron_config=neuron_config,
            reduction_factor=0.5,
        )
        print(f"NeuronStack initialisé: {fovea_shape} → 4 couches")
    
    def _build_cell_params(self):
        """Pré-calcule les paramètres des cellules pour OpenCL."""
        cfg = self.config
        num_cells = cfg.num_rings * cfg.num_sectors
        self.cell_params = np.zeros(num_cells * 4, dtype=np.float32)
        
        # Recalculer les rayons comme dans Fovea._build_cells()
        radii = [0.0]
        for i in range(cfg.num_rings):
            if i < cfg.num_rings // 4:
                r = cfg.fovea_radius * (i + 1) / (cfg.num_rings // 4)
            else:
                t = (i - cfg.num_rings // 4) / (cfg.num_rings * 0.75)
                r = cfg.fovea_radius + (cfg.max_radius - cfg.fovea_radius) * (t ** 1.5)
            radii.append(min(r, cfg.max_radius))
        
        sector_angle = 2 * math.pi / cfg.num_sectors
        
        idx = 0
        for ring_idx in range(cfg.num_rings):
            inner_r = radii[ring_idx]
            outer_r = radii[ring_idx + 1]
            for sector_idx in range(cfg.num_sectors):
                start_angle = sector_idx * sector_angle
                end_angle = (sector_idx + 1) * sector_angle
                
                self.cell_params[idx * 4 + 0] = inner_r
                self.cell_params[idx * 4 + 1] = outer_r
                self.cell_params[idx * 4 + 2] = start_angle
                self.cell_params[idx * 4 + 3] = end_angle
                idx += 1
    
    def compute_saliency_map(
        self, 
        gray: np.ndarray,
        prev_gray: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calcule une carte de saillance basée sur le contraste et le mouvement.
        
        Args:
            gray: Image grayscale courante
            prev_gray: Image grayscale précédente (pour mouvement)
            
        Returns:
            Carte de saillance normalisée
        """
        import time
        start_time = time.perf_counter()
        
        # Essayer OpenCL si disponible
        if self.opencl is not None:
            try:
                # Redimensionner d'abord pour le GPU (plus efficace)
                gray_small = cv2.resize(gray, (160, 120))
                
                # Saillance par gradient (contraste)
                saliency = self.opencl.compute_saliency(gray_small)
                
                # Ajouter mouvement si image précédente disponible
                if prev_gray is not None:
                    prev_small = cv2.resize(prev_gray, (160, 120))
                    motion = self.opencl.abs_diff(gray_small, prev_small)
                    # Combiner gradient (60%) + mouvement (40%)
                    saliency = saliency * 0.6 + motion * 0.4
                
                saliency_out = cv2.resize(saliency, (64, 48))
                
                self.gpu_time_ms = (time.perf_counter() - start_time) * 1000
                return saliency_out
            except Exception as e:
                print(f"OpenCL saliency fallback: {e}")
        
        # Fallback CPU
        # Saillance par contraste (gradient)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # Saillance par mouvement
        if prev_gray is not None:
            motion = cv2.absdiff(gray, prev_gray).astype(np.float32)
        else:
            motion = np.zeros_like(gradient)
        
        # Combiner (gradient + mouvement)
        saliency = gradient * 0.6 + motion * 0.4
        
        # Normaliser
        saliency = saliency / (saliency.max() + 1e-6)
        
        # Réduire la résolution pour accélérer
        saliency_small = cv2.resize(saliency, (64, 48))
        
        self.gpu_time_ms = (time.perf_counter() - start_time) * 1000
        return saliency_small
    
    def find_saliency_peaks(
        self, 
        saliency_left: np.ndarray,
        saliency_right: np.ndarray,
        threshold: float = 0.3
    ) -> List[SaliencyPoint]:
        """Trouve les pics de saillance communs aux deux yeux.
        
        Args:
            saliency_left: Carte de saillance gauche
            saliency_right: Carte de saillance droite
            threshold: Seuil de détection
            
        Returns:
            Liste de points saillants
        """
        # Corrélation des cartes de saillance
        # Les zones saillantes dans les deux yeux sont des candidats
        combined = saliency_left * saliency_right
        
        h, w = combined.shape
        scale_x = self.width / w
        scale_y = self.height / h
        
        peaks = []
        
        # Trouver les maxima locaux
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                val = combined[y, x]
                if val > threshold:
                    # Vérifier si c'est un maximum local
                    neighborhood = combined[y-1:y+2, x-1:x+2]
                    if val >= neighborhood.max():
                        real_x = x * scale_x
                        real_y = y * scale_y
                        peaks.append(SaliencyPoint(real_x, real_y, val))
        
        # Trier par force et garder les meilleurs
        peaks.sort(key=lambda p: p.strength, reverse=True)
        return peaks[:self.max_saliency_points]
    
    def compute_stereo_correlation(
        self,
        left_act: np.ndarray,
        right_act: np.ndarray
    ) -> float:
        """Calcule la corrélation entre les activations des deux fovéas.
        
        Une forte corrélation indique que les deux yeux voient le même objet.
        
        Returns:
            Score de corrélation (0-1)
        """
        # Essayer OpenCL si disponible
        if self.opencl is not None:
            try:
                # stereo_correlation retourne (correlation_array, disparity_array)
                corr_arr, _ = self.opencl.stereo_correlation(left_act, right_act)
                # Moyenne des corrélations pour obtenir un score global
                return float(np.mean(np.clip(corr_arr, 0, 1)))
            except Exception:
                pass  # Fallback CPU silencieux
        
        # CPU fallback
        # Normaliser
        left_norm = left_act / (np.linalg.norm(left_act) + 1e-6)
        right_norm = right_act / (np.linalg.norm(right_act) + 1e-6)
        
        # Produit scalaire = corrélation
        correlation = np.sum(left_norm * right_norm)
        
        return max(0.0, min(1.0, correlation))
    
    def update_vergence(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        attention_x: float,
        attention_y: float
    ) -> float:
        """Ajuste dynamiquement la vergence pour maximiser la corrélation.
        
        Cherche l'offset horizontal optimal entre les deux yeux pour
        aligner parfaitement les images sur le point d'attention.
        
        Args:
            left_img: Image gauche complète
            right_img: Image droite complète
            attention_x: Position X du regard dans l'image de référence
            attention_y: Position Y du regard
            
        Returns:
            Nouvel offset de vergence
        """
        # Taille de la fenêtre de recherche - plus grande pour meilleure corrélation
        window_size = 64
        half_w = window_size // 2
        
        # Position entière pour extraction
        cx = int(attention_x)
        cy = int(attention_y)
        
        # Vérifier les limites
        h, w = left_img.shape[:2]
        if cx - half_w < 0 or cx + half_w >= w or cy - half_w < 0 or cy + half_w >= h:
            return self.vergence_offset  # Garder la vergence actuelle
        
        # Extraire la fenêtre de référence (œil gauche)
        ref_window = left_img[cy - half_w:cy + half_w, cx - half_w:cx + half_w]
        if len(ref_window.shape) == 3:
            ref_gray = np.mean(ref_window, axis=2)
        else:
            ref_gray = ref_window.astype(float)
        
        # Normaliser la référence
        ref_norm = ref_gray - np.mean(ref_gray)
        ref_std = np.std(ref_gray) + 1e-6
        ref_norm = ref_norm / ref_std
        
        # Chercher le meilleur offset (-30 à +30 pixels)
        best_offset = self.vergence_offset
        best_corr = -1.0
        
        # Recherche grossière puis fine
        search_offsets = list(range(int(self.vergence_min), int(self.vergence_max) + 1, 3))
        # Ajouter la vergence actuelle et ses voisins
        for delta in range(-3, 4):
            candidate = int(self.vergence_offset + delta)
            if candidate not in search_offsets:
                search_offsets.append(candidate)
        
        for offset in search_offsets:
            test_x = cx + offset
            if test_x - half_w < 0 or test_x + half_w >= w:
                continue
            
            # Extraire la fenêtre test (œil droit)
            test_window = right_img[cy - half_w:cy + half_w, test_x - half_w:test_x + half_w]
            if len(test_window.shape) == 3:
                test_gray = np.mean(test_window, axis=2)
            else:
                test_gray = test_window.astype(float)
            
            # Normaliser
            test_norm = test_gray - np.mean(test_gray)
            test_std = np.std(test_gray) + 1e-6
            test_norm = test_norm / test_std
            
            # Corrélation croisée normalisée
            corr = np.sum(ref_norm * test_norm) / (window_size * window_size)
            
            if corr > best_corr:
                best_corr = corr
                best_offset = float(offset)
        
        # Lissage avec vélocité (éviter les sauts brusques)
        target_velocity = (best_offset - self.vergence_offset) * self.vergence_speed
        self.vergence_velocity = 0.7 * self.vergence_velocity + 0.3 * target_velocity
        
        # Appliquer la vélocité
        new_offset = self.vergence_offset + self.vergence_velocity
        
        # Clamp aux limites
        new_offset = max(self.vergence_min, min(self.vergence_max, new_offset))
        
        # Historique pour analyse
        self.vergence_history.append(new_offset)
        if len(self.vergence_history) > 60:
            self.vergence_history.pop(0)
        
        self.vergence_offset = new_offset
        return new_offset
    
    def decide_next_action(
        self,
        correlation: float,
        saliency_peaks: List[SaliencyPoint],
        saliency_map: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None
    ) -> Optional[Tuple[float, float]]:
        """Décide de la prochaine action du regard.
        
        Utilise le système d'attention avec:
        - Inhibition de retour (évite les zones récemment visitées)
        - Mémoire des positions (exploration vs exploitation)
        - Zoom adaptatif
        
        Returns:
            Nouvelle cible (x, y) ou None si pas de mouvement
        """
        # Pas d'action si pas en mode autonome
        if not self.autonomous:
            return None
        
        # Pas de saccade trop fréquente
        frames_since_saccade = self.frame_count - self.last_saccade_frame
        if frames_since_saccade < self.min_saccade_interval:
            return None
        
        # Mettre à jour le système d'attention
        self.attention.update(
            saliency=saliency_map,
            correlation=correlation,
            features=features
        )
        
        # Auto-zoom basé sur la corrélation et saillance
        current_saliency = 0.0
        if saliency_peaks:
            current_saliency = saliency_peaks[0].strength
        self.attention.auto_zoom(correlation, current_saliency)
        
        # Si faible corrélation ou en exploration, chercher nouvelle cible
        if correlation < 0.3 or (self.gaze.is_fixating and self.gaze.fixation_time > 15):
            if saliency_map is not None:
                # Utiliser l'attention controller pour sélectionner la cible
                # avec inhibition de retour
                target = self.attention.select_next_target(saliency_map, correlation)
                
                # Vérifier que c'est assez différent de la position actuelle
                dist = math.sqrt(
                    (target[0] - self.gaze.x)**2 + 
                    (target[1] - self.gaze.y)**2
                )
                if dist > 30:
                    return target
            elif saliency_peaks:
                # Fallback: utiliser les pics si pas de carte
                # mais moduler par l'inhibition
                for peak in saliency_peaks:
                    inhibition = self.attention.inhibition.get_inhibition_at(peak.x, peak.y)
                    if inhibition < 0.5:  # Pas trop inhibé
                        dist = math.sqrt(
                            (peak.x - self.gaze.x)**2 + 
                            (peak.y - self.gaze.y)**2
                        )
                        if dist > 30:
                            return (peak.x, peak.y)
        
        return None
    
    @property
    def current_zoom_level(self) -> int:
        """Niveau de zoom courant (0-7)."""
        return self.attention.zoom.level_index
    
    @property
    def current_zoom_scale(self) -> float:
        """Échelle de zoom courante."""
        return self.attention.zoom.current_scale
    
    @property
    def effective_radius(self) -> int:
        """Rayon effectif de la fovéa (ajusté au zoom)."""
        return self.attention.zoom.current_radius
    
    def zoom_in(self):
        """Zoom avant (plus de détails)."""
        self.attention.zoom.zoom_in()
    
    def zoom_out(self):
        """Zoom arrière (vue plus large)."""
        self.attention.zoom.zoom_out()
    
    def process(
        self,
        left_gray: np.ndarray,
        right_gray: np.ndarray,
        left_color: np.ndarray = None,
        right_color: np.ndarray = None
    ) -> dict:
        """Traite une paire d'images stéréo.
        
        Args:
            left_gray: Image gauche en grayscale
            right_gray: Image droite en grayscale
            left_color: Image gauche BGR (optionnel, pour mode couleur)
            right_color: Image droite BGR (optionnel, pour mode couleur)
            
        Returns:
            Dictionnaire avec les résultats
        """
        self.frame_count += 1
        
        # Calculer les cartes de saillance
        saliency_left = self.compute_saliency_map(
            left_gray, self.prev_left_gray
        )
        saliency_right = self.compute_saliency_map(
            right_gray, self.prev_right_gray
        )
        
        # Mettre à jour l'historique
        self.prev_left_gray = left_gray.copy()
        self.prev_right_gray = right_gray.copy()
        
        # Trouver les pics de saillance communs
        saliency_peaks = self.find_saliency_peaks(
            saliency_left, saliency_right
        )
        
        # Mettre à jour la position du regard
        gaze_x, gaze_y = self.gaze.update()
        
        # Mettre à jour la vergence dynamique pour optimiser l'alignement
        if left_color is not None and right_color is not None:
            self.update_vergence(left_color, right_color, gaze_x, gaze_y)
        else:
            self.update_vergence(left_gray, right_gray, gaze_x, gaze_y)
        
        # Positionner les fovéas avec vergence dynamique
        vergence_offset = self.vergence_offset
        self.left_fovea.set_gaze(gaze_x + vergence_offset, gaze_y)
        self.right_fovea.set_gaze(gaze_x - vergence_offset, gaze_y)
        
        # Variables pour les résultats couleur
        left_color_data = None
        right_color_data = None
        left_motion = None
        right_motion = None
        
        # Échantillonner avec les fovéas
        if self.use_color and left_color is not None and right_color is not None:
            # Mode couleur: échantillonner avec ColorFovea
            left_color_data = self.left_fovea.sample_color(left_color)
            right_color_data = self.right_fovea.sample_color(right_color)
            
            # Extraire luma pour la corrélation stéréo
            left_act = left_color_data['luma']
            right_act = right_color_data['luma']
            
            # Récupérer les infos de mouvement
            left_motion = self.left_fovea.get_dominant_motion()
            right_motion = self.right_fovea.get_dominant_motion()
        elif self.opencl is not None:
            try:
                left_act = self.opencl.polar_sample(
                    left_gray, 
                    int(gaze_x + vergence_offset), 
                    int(gaze_y),
                    self.cell_params,
                    self.config.num_rings,
                    self.config.num_sectors
                )
                right_act = self.opencl.polar_sample(
                    right_gray,
                    int(gaze_x - vergence_offset),
                    int(gaze_y),
                    self.cell_params,
                    self.config.num_rings,
                    self.config.num_sectors
                )
            except Exception as e:
                # Fallback CPU
                left_act = self.left_fovea.sample(left_gray)
                right_act = self.right_fovea.sample(right_gray)
        else:
            left_act = self.left_fovea.sample(left_gray)
            right_act = self.right_fovea.sample(right_gray)
        
        # Calculer la corrélation stéréo
        correlation = self.compute_stereo_correlation(left_act, right_act)
        self.correlation_history.append(correlation)
        if len(self.correlation_history) > 100:
            self.correlation_history.pop(0)
        
        # Combiner les cartes de saillance pour le système d'attention
        combined_saliency = (saliency_left + saliency_right) / 2
        
        # Décider de la prochaine action (avec inhibition de retour et zoom)
        next_target = self.decide_next_action(
            correlation, 
            saliency_peaks,
            saliency_map=combined_saliency,
            features=left_act
        )
        if next_target is not None:
            # Contraindre la cible aux limites (ajustées au zoom)
            x, y = self.attention.zoom.constrain_gaze(*next_target)
            self.gaze.saccade_to(x, y)
            self.last_saccade_frame = self.frame_count
        
        # Sauvegarder pour comparaison temporelle
        self.prev_left_act = left_act.copy()
        self.prev_right_act = right_act.copy()
        
        # Traitement par NeuronStack pour genèse dynamique
        # Combiner les activations gauche/droite en pattern d'entrée
        combined_act = (left_act + right_act) / 2.0
        stack_outputs = self.neuron_stack.process(combined_act, learn=True)
        stack_stats = self.neuron_stack.get_stats()
        
        # Mettre à jour le zoom (transition lisse)
        self.attention.zoom.update()
        
        return {
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'left_act': left_act,
            'right_act': right_act,
            'correlation': correlation,
            'saliency_peaks': saliency_peaks,
            'in_saccade': self.gaze.in_saccade,
            'fixating': self.gaze.is_fixating,
            'saliency_left': saliency_left,
            'saliency_right': saliency_right,
            # Nouvelles infos de zoom et attention
            'zoom_level': self.current_zoom_level,
            'zoom_scale': self.current_zoom_scale,
            'effective_radius': self.effective_radius,
            'num_pois': len(self.attention.memory.get_pois()),
            # Infos couleur et mouvement (si mode couleur)
            'left_color_data': left_color_data,
            'right_color_data': right_color_data,
            'left_motion': left_motion,
            'right_motion': right_motion,
            # Vergence dynamique
            'vergence_offset': self.vergence_offset,
            'vergence_velocity': self.vergence_velocity,
            # NeuronStack - genèse dynamique
            'stack_neurons': stack_stats['total_neurons'],
            'stack_patterns': sum(stack_stats.get('patterns_per_layer', [])),
            'stack_stable': sum(stack_stats.get('stable_patterns_per_layer', [])),
            'stack_stats': stack_stats,
        }
    
    def force_saccade_to(self, x: float, y: float):
        """Force une saccade vers une position."""
        self.gaze.saccade_to(x, y)
        self.last_saccade_frame = self.frame_count
    
    def random_saccade(self):
        """Effectue une saccade vers une position aléatoire."""
        x = np.random.uniform(self.gaze.min_x, self.gaze.max_x)
        y = np.random.uniform(self.gaze.min_y, self.gaze.max_y)
        self.force_saccade_to(x, y)
    
    def reset(self):
        """Réinitialise au centre."""
        center_x = self.width // 2
        center_y = self.height // 2
        self.gaze.x = center_x
        self.gaze.y = center_y
        self.gaze.target_x = center_x
        self.gaze.target_y = center_y
        self.left_fovea.reset()
        self.right_fovea.reset()
        self.correlation_history.clear()


class StereoCamera:
    """Capture stéréo side-by-side en haute résolution."""
    
    def __init__(self, camera_index: int = 1):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra {camera_index}")
        
        # Configurer pour résolution maximale
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Lire une frame pour obtenir les dimensions réelles
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Impossible de lire une frame")
        
        self.height, self.width = frame.shape[:2]
        self.half_width = self.width // 2
        
        print(f"Caméra stéréo HD: {self.width}x{self.height}")
        print(f"Chaque œil: {self.half_width}x{self.height}")
    
    def read(self):
        """Lit une paire stéréo.
        
        Returns:
            Tuple (success, left_frame, right_frame)
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None
        
        left = frame[:, :self.half_width]
        right = frame[:, self.half_width:]
        
        return True, left, right
    
    def release(self):
        self.cap.release()


def draw_neuron_stack_2d(stack: NeuronStack, size: int = 180) -> np.ndarray:
    """Visualise le NeuronStack en 2D.
    
    Affiche chaque couche comme une grille avec:
    - Neurones actifs en couleur vive
    - Neurones dormants en gris
    - Patterns en cours de détection en contour
    
    Args:
        stack: Le NeuronStack à visualiser
        size: Taille de l'image de sortie
        
    Returns:
        Image BGR de la visualisation
    """
    num_layers = stack.num_layers
    stats = stack.get_stats()
    
    # Créer l'image
    viz = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Diviser verticalement par couches
    layer_height = size // num_layers
    
    # Couleurs par couche
    layer_colors = [
        (100, 200, 255),  # Couche 0: Cyan
        (100, 255, 150),  # Couche 1: Vert
        (255, 200, 100),  # Couche 2: Orange
        (255, 100, 200),  # Couche 3: Rose
    ]
    
    for layer_idx, layer in enumerate(stack.layers):
        y_start = layer_idx * layer_height
        y_end = (layer_idx + 1) * layer_height
        
        # Fond de la couche (gradient subtil)
        base_color = layer_colors[layer_idx % len(layer_colors)]
        for y in range(y_start, y_end):
            intensity = 20 + 10 * (y - y_start) // layer_height
            viz[y, :] = [intensity, intensity, intensity]
        
        # Dessiner les neurones
        h, w = layer.shape
        neurons = layer.neurons
        
        if neurons:
            # Taille d'une cellule
            cell_w = size // w
            cell_h = layer_height // h
            
            for neuron in neurons:
                # Position du centroïde
                cx, cy = neuron.centroid
                px = int(cx * cell_w)
                py = y_start + int(cy * cell_h)
                
                # Taille basée sur le RF
                radius = max(2, int(math.sqrt(neuron._rf_size) * cell_w / w))
                
                # Couleur basée sur l'état
                from neuronspikes.genesis import NeuronState
                if neuron.state == NeuronState.FIRING:
                    color = (0, 255, 255)  # Jaune vif
                    radius = int(radius * 1.5)
                elif neuron.state == NeuronState.CHARGING:
                    intensity = int(128 + neuron.potential * 127)
                    color = (intensity, intensity, base_color[2])
                elif neuron.state == NeuronState.REFRACTORY:
                    color = (50, 50, 100)  # Bleu sombre
                else:
                    color = (80, 80, 80)  # Gris
                
                cv2.circle(viz, (px, py), radius, color, -1)
                
                # Contour pour mieux voir
                cv2.circle(viz, (px, py), radius, base_color, 1)
        
        # Label de la couche
        label = f"L{layer_idx}: {len(neurons)}n"
        cv2.putText(viz, label, (5, y_end - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, base_color, 1)
    
    # Stats globales en haut
    total_n = stats['total_neurons']
    patterns = sum(stats.get('patterns_per_layer', []))
    cv2.putText(viz, f"N:{total_n} P:{patterns}", (size - 70, 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    return viz


def draw_agent_overlay(
    image: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    fovea_config: FoveaConfig,
    in_saccade: bool,
    fixating: bool,
    saliency_peaks: List[SaliencyPoint],
    correlation: float,
    vergence_offset: float = 0.0,
    is_left_eye: bool = True
) -> np.ndarray:
    """Dessine l'overlay de l'agent sur l'image."""
    img = image.copy()
    
    gx, gy = int(gaze_x), int(gaze_y)
    
    # Couleur selon l'état
    if in_saccade:
        color = (0, 255, 255)  # Jaune = saccade
    elif fixating:
        color = (0, 255, 0)  # Vert = fixation
    else:
        color = (255, 255, 0)  # Cyan = poursuite
    
    # Zone fovéale
    cv2.circle(img, (gx, gy), fovea_config.fovea_radius, color, 2)
    
    # Zone périphérique
    cv2.circle(img, (gx, gy), fovea_config.max_radius, color, 1)
    
    # Croix de fixation
    cross_size = 15
    cv2.line(img, (gx - cross_size, gy), (gx + cross_size, gy), color, 2)
    cv2.line(img, (gx, gy - cross_size), (gx, gy + cross_size), color, 2)
    
    # Points de saillance
    for i, peak in enumerate(saliency_peaks[:5]):
        px, py = int(peak.x), int(peak.y)
        radius = int(5 + peak.strength * 10)
        alpha = 0.3 + peak.strength * 0.5
        cv2.circle(img, (px, py), radius, (0, 0, 255), 1)
        if i == 0:
            cv2.putText(img, "!", (px + 5, py - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Indicateur de corrélation
    bar_width = 100
    bar_height = 10
    bar_x, bar_y = 10, 10
    cv2.rectangle(img, (bar_x, bar_y), 
                 (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    fill_width = int(correlation * bar_width)
    corr_color = (0, int(255 * correlation), int(255 * (1 - correlation)))
    cv2.rectangle(img, (bar_x, bar_y),
                 (bar_x + fill_width, bar_y + bar_height), corr_color, -1)
    cv2.putText(img, f"Corr: {correlation:.2f}", (bar_x + bar_width + 5, bar_y + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Indicateur de vergence (ligne horizontale montrant le décalage)
    if abs(vergence_offset) > 0.5:
        verg_y = bar_y + bar_height + 15
        verg_center = bar_x + bar_width // 2
        verg_scale = 2.0  # pixels par unité de vergence
        verg_end = int(verg_center + vergence_offset * verg_scale * (1 if is_left_eye else -1))
        
        # Ligne de base
        cv2.line(img, (bar_x, verg_y), (bar_x + bar_width, verg_y), (80, 80, 80), 1)
        # Marqueur central
        cv2.line(img, (verg_center, verg_y - 3), (verg_center, verg_y + 3), (150, 150, 150), 1)
        # Flèche de vergence
        verg_color = (255, 200, 100) if vergence_offset > 0 else (100, 200, 255)
        cv2.arrowedLine(img, (verg_center, verg_y), (verg_end, verg_y), verg_color, 2, tipLength=0.3)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Agent stéréo avec attention')
    parser.add_argument('-c', '--camera', type=int, default=1,
                        help='Index de la caméra stéréo')
    parser.add_argument('-r', '--rings', type=int, default=16,
                        help='Nombre d\'anneaux (défaut: 16 pour meilleure résolution)')
    parser.add_argument('-s', '--sectors', type=int, default=24,
                        help='Nombre de secteurs (défaut: 24 pour meilleure résolution)')
    parser.add_argument('--color', action='store_true',
                        help='Mode couleur avec détection de mouvement (ColorFovea)')
    args = parser.parse_args()
    
    # Initialiser la caméra
    try:
        camera = StereoCamera(args.camera)
    except RuntimeError as e:
        print(f"Erreur: {e}")
        return 1
    
    # Configuration des fovéas - grande zone pour meilleure convergence stéréo
    # Plus de résolution = meilleure corrélation entre les deux yeux
    config = FoveaConfig(
        num_rings=args.rings,
        num_sectors=args.sectors,
        fovea_radius=32,      # Zone centrale haute résolution: 32px
        max_radius=128,       # Rayon total: 128px (fovéa de 256px de diamètre)
    )
    
    # Créer l'agent d'attention
    agent = AttentionAgent(
        frame_width=camera.half_width,
        frame_height=camera.height,
        fovea_config=config,
        use_color=args.color,
    )
    
    # État
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    show_disparity = False
    
    print("\n=== Agent Stéréo avec Attention ===")
    print("Mode AUTONOME activé - Les yeux cherchent les détails communs")
    print("Zoom adaptatif et inhibition de retour activés")
    print("")
    print("Touches:")
    print("  a     Toggle autonome/manuel")
    print("  s     Saccade aléatoire")
    print("  r     Reset au centre (efface mémoire)")
    print("  d     Toggle disparité/corrélation")
    print("  +/-   Zoom avant/arrière")
    print("  i     Afficher statistiques attention")
    print("  q/ESC Quitter")
    print("  Clic  Saccade vers position")
    print("")
    
    window_name = "Stereo Agent"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Calculer la position dans l'une des vues
            display_width = param['display_width']
            half = display_width // 2
            if x < half:
                # Clic dans vue gauche - convertir en coordonnées image
                scale = camera.half_width / half
                real_x = x * scale
                real_y = y * scale
            else:
                real_x = (x - half) * (camera.half_width / half)
                real_y = y * (camera.height / param['display_height'])
            
            agent.force_saccade_to(real_x, real_y)
    
    display_params = {'display_width': 1280, 'display_height': 360}
    cv2.setMouseCallback(window_name, mouse_callback, display_params)
    
    try:
        while True:
            ret, left, right = camera.read()
            if not ret:
                print("Erreur lecture caméra")
                break
            
            # Convertir en grayscale
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            
            # Traitement par l'agent (avec couleur si mode activé)
            if args.color:
                result = agent.process(
                    left_gray, right_gray,
                    left_color=left, right_color=right
                )
            else:
                result = agent.process(left_gray, right_gray)
            
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
            
            # Visualisation
            # Dessiner l'overlay sur les deux images avec vergence dynamique
            vergence = result['vergence_offset']
            left_viz = draw_agent_overlay(
                left, 
                result['gaze_x'] + vergence,  # Position œil gauche
                result['gaze_y'],
                config,
                result['in_saccade'],
                result['fixating'],
                result['saliency_peaks'],
                result['correlation'],
                vergence_offset=vergence,
                is_left_eye=True
            )
            right_viz = draw_agent_overlay(
                right,
                result['gaze_x'] - vergence,  # Position œil droit
                result['gaze_y'],
                config,
                result['in_saccade'],
                result['fixating'],
                result['saliency_peaks'],
                result['correlation'],
                vergence_offset=vergence,
                is_left_eye=False
            )
            
            # Visualisation des fovéas
            fovea_size = 180
            left_fovea_viz = visualize_fovea(agent.left_fovea, size=fovea_size)
            right_fovea_viz = visualize_fovea(agent.right_fovea, size=fovea_size)
            
            # Visualisation du NeuronStack
            stack_viz = draw_neuron_stack_2d(agent.neuron_stack, size=fovea_size)
            
            # Carte de corrélation ou disparité
            if show_disparity:
                disparity = result['left_act'] - result['right_act']
                disparity_norm = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-6)
                analysis_viz = cv2.applyColorMap(
                    (disparity_norm * 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                analysis_viz = cv2.resize(analysis_viz, (fovea_size, fovea_size))
            else:
                corr = result['left_act'] * result['right_act']
                corr_norm = corr / (corr.max() + 1e-6)
                analysis_viz = np.zeros((fovea_size, fovea_size, 3), dtype=np.uint8)
                h, w = corr_norm.shape
                cell_h, cell_w = fovea_size // h, fovea_size // w
                for i in range(h):
                    for j in range(w):
                        c = corr_norm[i, j]
                        color = (int(c * 100), int(c * 255), int(c * 200))
                        y1, y2 = i * cell_h, (i + 1) * cell_h
                        x1, x2 = j * cell_w, (j + 1) * cell_w
                        analysis_viz[y1:y2, x1:x2] = color
            
            # Assembler l'affichage
            # Redimensionner les images caméra
            display_h = 360
            scale = display_h / camera.height
            display_w = int(camera.half_width * scale)
            
            left_small = cv2.resize(left_viz, (display_w, display_h))
            right_small = cv2.resize(right_viz, (display_w, display_h))
            
            # Ligne 1: images stéréo
            row1 = np.hstack([left_small, right_small])
            
            # Ligne 2: fovéas, analyse et neurones
            fovea_row = np.hstack([left_fovea_viz, analysis_viz, stack_viz, right_fovea_viz])
            # Redimensionner pour correspondre
            fovea_row = cv2.resize(fovea_row, (row1.shape[1], fovea_size))
            
            display = np.vstack([row1, fovea_row])
            
            # Ajouter les labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            mode_text = "AUTONOME" if agent.autonomous else "MANUEL"
            mode_color = (0, 255, 0) if agent.autonomous else (0, 165, 255)
            cv2.putText(display, f"Mode: {mode_text}", (display_w - 120, 25), 
                       font, 0.5, mode_color, 1)
            
            # Affichage FPS et GPU
            gpu_info = "GPU" if agent.opencl else "CPU"
            cv2.putText(display, f"FPS: {fps:.1f} [{gpu_info}]", (10, display_h - 10),
                       font, 0.5, (0, 255, 0), 1)
            
            # Temps GPU si disponible
            if agent.gpu_time_ms > 0:
                cv2.putText(display, f"Saliency: {agent.gpu_time_ms:.1f}ms", 
                           (10, display_h - 25), font, 0.4, (180, 180, 180), 1)
            
            state_text = "SACCADE" if result['in_saccade'] else ("FIXATION" if result['fixating'] else "POURSUITE")
            cv2.putText(display, state_text, (display_w + 10, 25),
                       font, 0.5, (255, 255, 255), 1)
            
            # Affichage du zoom et POIs
            zoom_text = f"Zoom: {result['zoom_level']}/7 (x{result['zoom_scale']:.1f})"
            cv2.putText(display, zoom_text, (display_w + 10, 45),
                       font, 0.4, (200, 200, 100), 1)
            
            # Affichage vergence dynamique
            verg = result['vergence_offset']
            verg_dir = "CONV" if verg > 0 else ("DIV" if verg < 0 else "PAR")
            verg_text = f"Vergence: {verg_dir} {abs(verg):.1f}px"
            cv2.putText(display, verg_text, (display_w + 100, 45),
                       font, 0.4, (100, 200, 255), 1)
            
            if result['num_pois'] > 0:
                cv2.putText(display, f"POIs: {result['num_pois']}", 
                           (display_w + 10, 65), font, 0.4, (100, 200, 100), 1)
            
            # Affichage des neurones créés dynamiquement
            neurons_text = f"Neurons: {result['stack_neurons']} (P:{result['stack_patterns']} S:{result['stack_stable']})"
            cv2.putText(display, neurons_text, (display_w + 120, 65),
                       font, 0.4, (255, 200, 100), 1)
            
            # Affichage infos couleur et mouvement (si mode couleur)
            if args.color and result['left_motion'] is not None:
                motion = result['left_motion']
                motion_text = f"Motion: {motion.magnitude:.1f}px @{math.degrees(motion.direction):.0f}°"
                cv2.putText(display, motion_text, (display_w + 10, 85),
                           font, 0.4, (100, 150, 255), 1)
            elif args.color:
                cv2.putText(display, "Motion: --", (display_w + 10, 85),
                           font, 0.4, (100, 150, 255), 1)
            
            # Indicateur mode couleur
            if args.color:
                cv2.putText(display, "COLOR", (display_w - 100, display_h - 10),
                           font, 0.5, (100, 200, 255), 1)
            
            # Historique de corrélation
            if len(agent.correlation_history) > 10:
                avg_corr = np.mean(agent.correlation_history[-10:])
                cv2.putText(display, f"Corr moy: {avg_corr:.2f}", 
                           (display_w + 10, display_h - 10),
                           font, 0.4, (200, 200, 200), 1)
            
            cv2.imshow(window_name, display)
            display_params['display_width'] = row1.shape[1]
            display_params['display_height'] = display_h
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            
            if key in (ord('q'), 27):
                break
            elif key == ord('a'):
                agent.autonomous = not agent.autonomous
                print(f"Mode: {'AUTONOME' if agent.autonomous else 'MANUEL'}")
            elif key == ord('s'):
                agent.random_saccade()
                print("Saccade aléatoire!")
            elif key == ord('r'):
                agent.reset()
                agent.attention.reset()
                print("Reset au centre (mémoire effacée)")
            elif key == ord('d'):
                show_disparity = not show_disparity
                print(f"Affichage: {'Disparité' if show_disparity else 'Corrélation'}")
            elif key == ord('+') or key == ord('='):
                agent.zoom_in()
                print(f"Zoom IN → niveau {agent.current_zoom_level}, échelle {agent.current_zoom_scale:.2f}")
            elif key == ord('-') or key == ord('_'):
                agent.zoom_out()
                print(f"Zoom OUT → niveau {agent.current_zoom_level}, échelle {agent.current_zoom_scale:.2f}")
            elif key == ord('i'):
                # Afficher les statistiques d'attention
                stats = agent.attention.get_stats()
                print(f"=== Statistiques Attention ===")
                print(f"  POIs découverts: {stats['num_pois']}")
                print(f"  Positions mémorisées: {stats['num_memories']}")
                print(f"  Distance totale parcourue: {stats['total_distance']:.0f} px")
                print(f"  Zoom: niveau {stats['zoom_level']}, échelle {stats['zoom_scale']:.2f}")
    
    except KeyboardInterrupt:
        print("\nInterruption...")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
