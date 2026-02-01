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
    # Voies rétiniennes bio-inspirées
    RetinalProcessor,
    PathwayConfig,
)


@dataclass
class SaliencyPoint:
    """Point saillant détecté."""
    x: float
    y: float
    strength: float  # Force de la saillance (0-1)
    age: int = 0  # Nombre de frames depuis détection
    # Position dans l'œil droit (peut différer à cause de la parallaxe)
    x_right: float = None  # Si None, utilise x - vergence_offset
    
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
        self.vergence_speed: float = 3.0   # Vitesse d'ajustement pixels/frame (plus réactif)
        self.vergence_search_range: int = 10  # Plage de recherche pour optimisation
        self.vergence_continuous: bool = True  # Mode vergence continue optimale
        
        # État proprioceptif des muscles oculaires
        self.oculomotor = OculomotorState()
        
        # Mémoire saccadique - associe patterns visuels et mouvements
        self.saccade_memory = SaccadeMemory(max_items=200)
        self._in_saccade = False  # Suivi de l'état de saccade
        self._pre_saccade_pattern: Optional[np.ndarray] = None
        
        # État
        self.autonomous = True
        self.frame_count = 0
        self.last_saccade_frame = 0
        self.min_saccade_interval = 15  # Frames minimum entre saccades (plus stable)
        
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
        
        # Processeur rétinien bio-inspiré (Magno/Parvo/V1)
        # NOTE: Gardé pour visualisation, mais la saillance est maintenant
        # calculée en coordonnées POLAIRES par ColorFovea.compute_saliency()
        pathway_config = PathwayConfig(
            magno_sigma=2.0,            # Grands champs récepteurs pour mouvement
            magno_threshold=0.03,       # Sensible au mouvement
            parvo_color_gain=1.5,       # Opposition couleur amplifiée
            lateral_strength=0.4,       # Inhibition centre-surround
            gabor_num_orientations=8,   # Détecteurs d'orientation
            gabor_num_scales=2,         # 2 échelles (rapide)
        )
        self.retinal_processor = RetinalProcessor(pathway_config)
        self.use_retinal_saliency = False  # Désactivé: utiliser ColorFovea.compute_saliency()
        print(f"Saillance bio-inspirée: coordonnées POLAIRES via ColorFovea")
    
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
        
        # Saillance par luminosité (zones très brillantes = intéressantes)
        brightness = gray.astype(np.float32) / 255.0
        # Accentuer les zones très lumineuses (seuillage doux)
        bright_mask = np.clip((brightness - 0.7) * 5.0, 0, 1)
        
        # Saillance par mouvement
        if prev_gray is not None:
            motion = cv2.absdiff(gray, prev_gray).astype(np.float32)
        else:
            motion = np.zeros_like(gradient)
        
        # Combiner (gradient 40% + luminosité 30% + mouvement 30%)
        gradient_norm = gradient / (gradient.max() + 1e-6)
        motion_norm = motion / (motion.max() + 1e-6)
        saliency = gradient_norm * 0.4 + bright_mask * 0.3 + motion_norm * 0.3
        
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
        threshold: float = 0.3,
        left_gray: np.ndarray = None,
        right_gray: np.ndarray = None,
        left_color: np.ndarray = None,
        right_color: np.ndarray = None
    ) -> List[SaliencyPoint]:
        """Trouve les pics de saillance communs aux deux yeux.
        
        Priorise les zones TRÈS lumineuses (lumières, reflets).
        Calcule automatiquement la disparité si les deux images sont fournies.
        
        NOTE: La saillance bio-inspirée fine (Magno/Parvo) est calculée
        en coordonnées POLAIRES après l'échantillonnage fovéal via
        ColorFovea.compute_saliency(). Ici on fait une détection globale
        pour décider des saccades.
        
        Args:
            saliency_left: Carte de saillance gauche (image complète réduite)
            saliency_right: Carte de saillance droite
            threshold: Seuil de détection
            left_gray: Image grayscale gauche pour détecter les lumières
            right_gray: Image grayscale droite pour détecter les lumières
            left_color: Image couleur gauche (non utilisé ici, pour compat)
            right_color: Image couleur droite (non utilisé ici, pour compat)
            
        Returns:
            Liste de points saillants en coordonnées image
        """
        h, w = saliency_left.shape
        scale_x = self.width / w
        scale_y = self.height / h
        
        peaks = []
        x_right_bright = None  # Position X dans l'image droite
        
        # NOTE: La saillance bio-inspirée fine (Magno/Parvo) est maintenant
        # calculée en coordonnées POLAIRES par ColorFovea.compute_saliency()
        # après l'échantillonnage. Ici on fait une détection GLOBALE simple
        # basée sur la corrélation stéréo pour décider des saccades.
        
        # Combiner les cartes de saillance gauche/droite (corrélation stéréo)
        combined_saliency = saliency_left * saliency_right
        
        # Détecter les pics dans la saillance combinée
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                val = combined_saliency[y, x]
                if val > threshold:
                    # Vérifier si c'est un maximum local
                    neighborhood = combined_saliency[y-1:y+2, x-1:x+2]
                    if val >= neighborhood.max():
                        real_x = x * scale_x
                        real_y = y * scale_y
                        peaks.append(SaliencyPoint(real_x, real_y, val))
        
        # PRIORITÉ 1: Détecter directement les zones très lumineuses
        if left_gray is not None:
            # Réduire à la même taille que la saliency map
            gray_left_small = cv2.resize(left_gray, (w, h))
            
            # Trouver le point le plus brillant dans l'image gauche
            max_val_left = gray_left_small.max()
            if max_val_left > 200:  # Seuil de luminosité élevée
                max_loc_left = np.unravel_index(gray_left_small.argmax(), gray_left_small.shape)
                y_bright, x_bright_left = max_loc_left
                
                # Si on a aussi l'image droite, trouver la lumière indépendamment
                if right_gray is not None:
                    gray_right_small = cv2.resize(right_gray, (w, h))
                    max_val_right = gray_right_small.max()
                    
                    if max_val_right > 200:
                        max_loc_right = np.unravel_index(gray_right_small.argmax(), gray_right_small.shape)
                        y_bright_right, x_bright_right = max_loc_right
                        
                        # Position réelle dans l'image droite
                        x_right_bright = x_bright_right * scale_x
                        
                        # Calculer la disparité pour info
                        disparity_small = x_bright_left - x_bright_right
                        disparity_full = disparity_small * scale_x
                        self.vergence_offset = 0.8 * self.vergence_offset + 0.2 * disparity_full
                
                real_x = x_bright_left * scale_x
                real_y = y_bright * scale_y
                # Ajouter avec haute priorité ET la position droite
                brightness_strength = max_val_left / 255.0
                peaks.append(SaliencyPoint(
                    x=real_x, 
                    y=real_y, 
                    strength=brightness_strength * 1.5, 
                    age=0,
                    x_right=x_right_bright
                ))
        
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
        
        Utilise une recherche par corrélation croisée normalisée (NCC)
        pour trouver l'offset horizontal optimal.
        
        IMPORTANT: Cette vergence est utilisée uniquement quand on n'a pas
        de détection explicite de la cible dans les deux yeux. Sinon,
        on utilise la différence de position détectée directement.
        
        Args:
            left_img: Image gauche complète
            right_img: Image droite complète
            attention_x: Position X du regard dans l'image gauche
            attention_y: Position Y du regard
            
        Returns:
            Nouvel offset de vergence
        """
        # Taille de la fenêtre - plus petite pour plus de précision locale
        window_size = 48
        half_w = window_size // 2
        
        # Position entière pour extraction
        cx = int(attention_x)
        cy = int(attention_y)
        
        # Vérifier les limites
        h, w = left_img.shape[:2]
        if cx - half_w < 0 or cx + half_w >= w or cy - half_w < 0 or cy + half_w >= h:
            return self.vergence_offset
        
        # Extraire la fenêtre de référence (œil gauche)
        ref_window = left_img[cy - half_w:cy + half_w, cx - half_w:cx + half_w]
        if len(ref_window.shape) == 3:
            ref_gray = np.mean(ref_window, axis=2)
        else:
            ref_gray = ref_window.astype(float)
        
        # Normaliser la référence
        ref_mean = np.mean(ref_gray)
        ref_std = np.std(ref_gray) + 1e-6
        ref_norm = (ref_gray - ref_mean) / ref_std
        
        # Recherche sub-pixel autour de la vergence actuelle
        # Plage de recherche adaptative: plus étroite si on est stable
        vergence_stable = len(self.vergence_history) > 10 and \
                         np.std(self.vergence_history[-10:]) < 3.0
        
        if vergence_stable:
            # Recherche fine autour de la position actuelle
            search_range = 10
            search_step = 1
        else:
            # Recherche large
            search_range = int(self.vergence_max - self.vergence_min)
            search_step = 2
        
        best_offset = self.vergence_offset
        best_corr = -1.0
        
        # Chercher le meilleur offset
        start = max(int(self.vergence_min), int(self.vergence_offset - search_range))
        end = min(int(self.vergence_max), int(self.vergence_offset + search_range))
        
        for offset in range(start, end + 1, search_step):
            test_x = cx - offset  # Œil droit regarde à gauche pour converger
            if test_x - half_w < 0 or test_x + half_w >= w:
                continue
            
            # Extraire la fenêtre test (œil droit)
            test_window = right_img[cy - half_w:cy + half_w, test_x - half_w:test_x + half_w]
            if len(test_window.shape) == 3:
                test_gray = np.mean(test_window, axis=2)
            else:
                test_gray = test_window.astype(float)
            
            # Normaliser
            test_mean = np.mean(test_gray)
            test_std = np.std(test_gray) + 1e-6
            test_norm = (test_gray - test_mean) / test_std
            
            # Corrélation croisée normalisée
            corr = np.mean(ref_norm * test_norm)
            
            if corr > best_corr:
                best_corr = corr
                best_offset = float(offset)
        
        # Raffinement sub-pixel par interpolation parabolique
        if best_corr > 0.3 and search_step == 1:
            # Calculer les corrélations voisines
            offsets_to_check = [best_offset - 1, best_offset, best_offset + 1]
            corrs = []
            for off in offsets_to_check:
                test_x = cx - int(off)
                if test_x - half_w >= 0 and test_x + half_w < w:
                    tw = right_img[cy - half_w:cy + half_w, test_x - half_w:test_x + half_w]
                    if len(tw.shape) == 3:
                        tg = np.mean(tw, axis=2)
                    else:
                        tg = tw.astype(float)
                    tn = (tg - np.mean(tg)) / (np.std(tg) + 1e-6)
                    corrs.append(np.mean(ref_norm * tn))
                else:
                    corrs.append(0)
            
            if len(corrs) == 3 and corrs[0] != corrs[2]:
                # Interpolation parabolique
                denom = 2 * (corrs[0] - 2 * corrs[1] + corrs[2])
                if abs(denom) > 1e-6:
                    sub_offset = (corrs[0] - corrs[2]) / denom
                    best_offset += np.clip(sub_offset, -0.5, 0.5)
        
        # Lissage temporel (plus fort si corrélation bonne)
        alpha = 0.3 if best_corr > 0.5 else 0.1
        target_velocity = (best_offset - self.vergence_offset) * alpha
        self.vergence_velocity = 0.8 * self.vergence_velocity + 0.2 * target_velocity
        
        # Appliquer la vélocité avec damping
        new_offset = self.vergence_offset + self.vergence_velocity
        
        # Clamp aux limites
        new_offset = max(self.vergence_min, min(self.vergence_max, new_offset))
        
        # Historique
        self.vergence_history.append(new_offset)
        if len(self.vergence_history) > 60:
            self.vergence_history.pop(0)
        
        self.vergence_offset = new_offset
        self._last_vergence_corr = best_corr  # Pour debug
        return new_offset
    
    def decide_next_action(
        self,
        correlation: float,
        saliency_peaks: List[SaliencyPoint],
        saliency_map: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None
    ) -> Optional[Tuple[float, float]]:
        """Décide de la prochaine action du regard.
        
        Mode "vergence optimale continue":
        1. Priorité à la vergence: ne saccade que si corrélation OK (> 0.4)
        2. Saccade vers le MEILLEUR point d'intérêt (pas aléatoire)
        3. Long temps de fixation pour laisser la vergence s'optimiser
        4. Construit une représentation d'ensemble par exploration méthodique
        5. LIMITE les saccades à la périphérie visible pour maintenir
           la continuité perceptive (chevauchement des patterns)
        
        Returns:
            Nouvelle cible (x, y) ou None si pas de mouvement
        """
        # Pas d'action si pas en mode autonome
        if not self.autonomous:
            return None
        
        # Pas de saccade trop fréquente - laisser le temps à la vergence
        frames_since_saccade = self.frame_count - self.last_saccade_frame
        if frames_since_saccade < self.min_saccade_interval:
            return None
        
        # Distance max de saccade = rayon périphérique de la fovéa
        # Ceci assure un chevauchement entre avant/après pour la mémoire saccadique
        max_saccade_dist = self.config.max_radius * 1.5  # ~périphérie visible
        min_saccade_dist = self.config.fovea_radius * 0.5  # Éviter micro-saccades
        
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
        
        # MODE VERGENCE OPTIMALE CONTINUE:
        # Ne saccade que si la vergence actuelle est bonne (corrélation haute)
        # Sinon, rester sur place et laisser la vergence s'optimiser
        
        # Si corrélation trop faible, attendre que la vergence s'améliore
        if correlation < 0.4:
            # Rester sur place - la vergence va s'optimiser automatiquement
            return None
        
        # Saccade seulement si:
        # 1. Bonne corrélation (vergence OK)
        # 2. Fixation assez longue (a bien exploré cette zone)
        # 3. Il existe un meilleur point d'intérêt ailleurs
        # 4. La cible est dans la zone périphérique (pas trop loin!)
        
        min_fixation_for_saccade = 30  # ~1 seconde à 30fps
        if not self.gaze.is_fixating or self.gaze.fixation_time < min_fixation_for_saccade:
            return None
        
        # Chercher le meilleur point d'intérêt DANS LA PÉRIPHÉRIE
        if saliency_map is not None:
            # Utiliser l'attention controller pour sélectionner la cible
            # avec inhibition de retour (évite les zones déjà visitées)
            target = self.attention.select_next_target(saliency_map, correlation)
            
            # Vérifier que la cible est dans la zone périphérique
            dist = math.sqrt(
                (target[0] - self.gaze.x)**2 + 
                (target[1] - self.gaze.y)**2
            )
            # Saccade seulement si dans la zone périphérique [min, max]
            if min_saccade_dist < dist < max_saccade_dist:
                return target
        
        # Fallback ou saliency_peaks: chercher dans la périphérie
        if saliency_peaks:
            # Utiliser le meilleur pic non-inhibé DANS LA PÉRIPHÉRIE
            best_peak = None
            best_score = -1
            
            for peak in saliency_peaks:
                dist = math.sqrt(
                    (peak.x - self.gaze.x)**2 + 
                    (peak.y - self.gaze.y)**2
                )
                
                # IGNORER les cibles hors de la zone périphérique
                if dist < min_saccade_dist or dist > max_saccade_dist:
                    continue
                
                inhibition = self.attention.inhibition.get_inhibition_at(peak.x, peak.y)
                # Score = saillance × (1 - inhibition)
                score = peak.strength * (1 - inhibition)
                
                # Préférer les cibles à distance optimale (milieu de la périphérie)
                optimal_dist = (min_saccade_dist + max_saccade_dist) / 2
                dist_score = 1.0 - abs(dist - optimal_dist) / max_saccade_dist
                score *= (0.8 + 0.4 * dist_score)
                
                if score > best_score:
                    best_score = score
                    best_peak = peak
            
            if best_peak is not None:
                return (best_peak.x, best_peak.y)
        
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
            saliency_left, saliency_right, 
            left_gray=left_gray, right_gray=right_gray,
            left_color=left_color, right_color=right_color
        )
        
        # Mettre à jour la position du regard
        gaze_x, gaze_y = self.gaze.update()
        
        # Variable pour savoir si on a une cible binoculaire directe
        self._direct_vergence = False
        
        # Si on a détecté une lumière brillante, FORCER le regard dessus
        if saliency_peaks and saliency_peaks[0].strength > 1.0:
            # C'est une lumière (strength > 1.0 = brightness * 1.5 > 1.0)
            bright_peak = saliency_peaks[0]
            gaze_x = bright_peak.x
            gaze_y = bright_peak.y
            
            if bright_peak.x_right is not None:
                # On a détecté la lumière dans les DEUX yeux!
                # Utiliser directement la disparité mesurée pour la vergence
                direct_disparity = gaze_x - bright_peak.x_right
                
                # Mise à jour DIRECTE de la vergence (pas par corrélation)
                # Lissage doux pour éviter les sauts
                alpha = 0.4  # Plus réactif car mesure directe fiable
                self.vergence_velocity = 0.7 * self.vergence_velocity + 0.3 * (direct_disparity - self.vergence_offset)
                self.vergence_offset = 0.6 * self.vergence_offset + 0.4 * direct_disparity
                self.vergence_offset = max(self.vergence_min, min(self.vergence_max, self.vergence_offset))
                
                # L'œil droit regarde directement la position détectée
                gaze_x_right = bright_peak.x_right
                self._direct_vergence = True
            else:
                # Lumière détectée seulement dans l'œil gauche
                gaze_x_right = gaze_x - self.vergence_offset
            
            # Mettre à jour le contrôleur de gaze pour qu'il suive
            self.gaze.x = gaze_x
            self.gaze.y = gaze_y
        else:
            # Pas de lumière brillante, utiliser vergence normale
            gaze_x_right = gaze_x - self.vergence_offset
        
        # Mettre à jour la vergence par corrélation SEULEMENT si pas de vergence directe
        if not self._direct_vergence:
            if left_color is not None and right_color is not None:
                self.update_vergence(left_color, right_color, gaze_x, gaze_y)
            else:
                self.update_vergence(left_gray, right_gray, gaze_x, gaze_y)
            # Recalculer gaze_x_right avec la vergence mise à jour
            gaze_x_right = gaze_x - self.vergence_offset
        
        # Positionner les fovéas - chaque œil regarde sa propre cible
        self.left_fovea.set_gaze(gaze_x, gaze_y)
        self.right_fovea.set_gaze(gaze_x_right, gaze_y)
        
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
                    int(gaze_x), 
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
        
        # SAILLANCE BIO-INSPIRÉE en coordonnées POLAIRES
        # Calculée APRÈS l'échantillonnage fovéal (ce qui est bio-fidèle)
        if self.use_color and hasattr(self.left_fovea, 'compute_saliency'):
            left_polar_saliency = self.left_fovea.compute_saliency()
            right_polar_saliency = self.right_fovea.compute_saliency()
            
            # Trouver les pics de saillance en coordonnées polaires
            left_peaks = self.left_fovea.get_saliency_peaks(left_polar_saliency)
            right_peaks = self.right_fovea.get_saliency_peaks(right_polar_saliency)
            
            # Stocker pour visualisation et analyse
            self._last_polar_saliency = {
                'left': left_polar_saliency,
                'right': right_polar_saliency,
                'left_peaks': left_peaks,
                'right_peaks': right_peaks,
            }
        else:
            self._last_polar_saliency = None
        
        # Combiner les cartes de saillance pour le système d'attention
        combined_saliency = (saliency_left + saliency_right) / 2
        
        # Combiner patterns gauche/droite pour la mémoire
        combined_pattern = (left_act + right_act) / 2.0
        
        # Vérifier si on vient de finir une saccade
        was_in_saccade = self._in_saccade
        self._in_saccade = self.gaze.in_saccade
        
        if was_in_saccade and not self._in_saccade:
            # Fin de saccade - enregistrer l'association
            motor_vec = self.oculomotor.get_motor_vector()
            self.saccade_memory.end_saccade(combined_pattern, motor_vec)
        
        # Décider de la prochaine action (avec inhibition de retour et zoom)
        next_target = self.decide_next_action(
            correlation, 
            saliency_peaks,
            saliency_map=combined_saliency,
            features=left_act
        )
        if next_target is not None:
            # Début de saccade - enregistrer le pattern actuel
            motor_vec = self.oculomotor.get_motor_vector()
            self.saccade_memory.start_saccade(combined_pattern, motor_vec)
            self._in_saccade = True
            
            # Contraindre la cible aux limites (ajustées au zoom)
            x, y = self.attention.zoom.constrain_gaze(*next_target)
            self.gaze.saccade_to(x, y)
            self.last_saccade_frame = self.frame_count
        
        # Sauvegarder pour comparaison temporelle
        self.prev_left_act = left_act.copy()
        self.prev_right_act = right_act.copy()
        
        # Mettre à jour l'état proprioceptif des muscles oculaires
        self.oculomotor.update(
            gaze_x, gaze_y,
            self.vergence_offset,
            self.width, self.height
        )
        
        # Encoder l'état oculomoteur en spikes pour le réseau
        oculomotor_spikes = self.oculomotor.to_spike_encoding(resolution=8)
        
        # Traitement par NeuronStack pour genèse dynamique
        # Combiner les activations gauche/droite en pattern d'entrée
        combined_act = (left_act + right_act) / 2.0
        stack_outputs = self.neuron_stack.process(combined_act, learn=True)
        stack_stats = self.neuron_stack.get_stats()
        
        # Mettre à jour le zoom (transition lisse)
        self.attention.zoom.update()
        
        return {
            'gaze_x': gaze_x,
            'gaze_x_right': gaze_x_right,  # Position indépendante œil droit
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
            'direct_vergence': self._direct_vergence,  # True = vergence par détection directe
            'vergence_correlation': getattr(self, '_last_vergence_corr', 0.0),
            # Proprioception oculomoteur
            'oculomotor_spikes': oculomotor_spikes,
            'oculomotor_state': self.oculomotor,
            # Mémoire saccadique
            'saccade_memory_stats': self.saccade_memory.get_stats(),
            'motor_effort': self.oculomotor.effort,
            # NeuronStack - genèse dynamique
            'stack_neurons': stack_stats['total_neurons'],
            'stack_patterns': sum(stack_stats.get('patterns_per_layer', [])),
            'stack_stable': sum(stack_stats.get('stable_patterns_per_layer', [])),
            'stack_stats': stack_stats,
            # Saillance bio-inspirée POLAIRE (via ColorFovea)
            'polar_saliency': self._last_polar_saliency,
            # Voies rétiniennes (pour visualisation seulement)
            'retinal_result': getattr(self, '_last_retinal_result', None),
            'use_retinal_saliency': self.use_retinal_saliency,
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


@dataclass
class SaccadeMemoryItem:
    """Un élément de mémoire saccadique.
    
    Lie un pattern visuel pré-saccade avec un pattern post-saccade
    via le vecteur de commande musculaire qui les relie.
    """
    # Patterns visuels (activations fovéales)
    pre_pattern: np.ndarray      # Pattern avant la saccade
    post_pattern: np.ndarray     # Pattern après la saccade
    
    # Commande musculaire qui lie les deux
    motor_command: np.ndarray    # [dx, dy, d_vergence, rotation]
    
    # Métadonnées
    timestamp: float = 0.0
    confidence: float = 1.0      # Décroît si non confirmé
    uses: int = 0                # Nombre d'utilisations pour prédiction
    
    def similarity(self, pattern: np.ndarray) -> float:
        """Calcule la similarité avec un pattern donné."""
        if pattern.shape != self.pre_pattern.shape:
            return 0.0
        # Corrélation normalisée
        norm_pre = np.linalg.norm(self.pre_pattern)
        norm_pat = np.linalg.norm(pattern)
        if norm_pre < 1e-6 or norm_pat < 1e-6:
            return 0.0
        return np.dot(self.pre_pattern.flatten(), pattern.flatten()) / (norm_pre * norm_pat)


class SaccadeMemory:
    """Mémoire associative des saccades.
    
    Permet de:
    1. Prédire ce qu'on va voir après une saccade (forward model)
    2. Estimer le mouvement nécessaire pour atteindre un pattern cible
    3. Construire une représentation spatiale cohérente de l'environnement
    
    Bio-inspiration: Colliculus supérieur + cortex pariétal postérieur
    """
    
    def __init__(self, max_items: int = 100, similarity_threshold: float = 0.8):
        self.memories: List[SaccadeMemoryItem] = []
        self.max_items = max_items
        self.similarity_threshold = similarity_threshold
        
        # Buffer pour enregistrer les saccades en cours
        self._pre_saccade_pattern: Optional[np.ndarray] = None
        self._pre_saccade_motor: Optional[np.ndarray] = None
        self._saccade_start_time: float = 0.0
        
        # Statistiques
        self.predictions_made = 0
        self.predictions_correct = 0
    
    def start_saccade(
        self, 
        current_pattern: np.ndarray,
        motor_command: np.ndarray
    ):
        """Appelé au début d'une saccade.
        
        Args:
            current_pattern: Pattern visuel actuel (pré-saccade)
            motor_command: Commande musculaire [dx, dy, d_verg, rotation]
        """
        self._pre_saccade_pattern = current_pattern.copy()
        self._pre_saccade_motor = motor_command.copy()
        self._saccade_start_time = time.time()
    
    def end_saccade(
        self,
        post_pattern: np.ndarray,
        motor_actual: np.ndarray = None
    ) -> Optional[SaccadeMemoryItem]:
        """Appelé à la fin d'une saccade pour enregistrer l'association.
        
        Args:
            post_pattern: Pattern visuel après la saccade
            motor_actual: Commande motrice réelle (si différente de planifiée)
            
        Returns:
            L'item de mémoire créé, ou None si pas de saccade en cours
        """
        if self._pre_saccade_pattern is None:
            return None
        
        # Utiliser la commande réelle si fournie
        motor = motor_actual if motor_actual is not None else self._pre_saccade_motor
        
        # Créer l'item de mémoire
        item = SaccadeMemoryItem(
            pre_pattern=self._pre_saccade_pattern,
            post_pattern=post_pattern.copy(),
            motor_command=motor.copy() if motor is not None else np.zeros(4),
            timestamp=time.time()
        )
        
        # Vérifier si un item similaire existe déjà
        best_match = self._find_similar(item.pre_pattern, motor)
        if best_match is not None and best_match.similarity(item.pre_pattern) > self.similarity_threshold:
            # Fusionner avec l'existant (moyennage)
            alpha = 0.3  # Poids du nouvel item
            best_match.post_pattern = (1 - alpha) * best_match.post_pattern + alpha * item.post_pattern
            best_match.confidence = min(1.0, best_match.confidence + 0.1)
            best_match.uses += 1
            item = best_match
        else:
            # Ajouter le nouvel item
            self.memories.append(item)
            if len(self.memories) > self.max_items:
                # Supprimer le moins utilisé/confiant
                self.memories.sort(key=lambda m: m.confidence * (1 + m.uses * 0.1))
                self.memories.pop(0)
        
        # Reset du buffer
        self._pre_saccade_pattern = None
        self._pre_saccade_motor = None
        
        return item
    
    def predict_post_saccade(
        self,
        current_pattern: np.ndarray,
        motor_command: np.ndarray
    ) -> Optional[np.ndarray]:
        """Prédit le pattern visuel après une saccade.
        
        Forward model: étant donné où je suis et où je vais,
        qu'est-ce que je vais voir?
        
        Args:
            current_pattern: Pattern visuel actuel
            motor_command: Commande musculaire planifiée
            
        Returns:
            Pattern prédit, ou None si pas de prédiction possible
        """
        best_match = self._find_similar(current_pattern, motor_command)
        if best_match is None:
            return None
        
        self.predictions_made += 1
        return best_match.post_pattern.copy()
    
    def estimate_motor_command(
        self,
        current_pattern: np.ndarray,
        target_pattern: np.ndarray
    ) -> Optional[np.ndarray]:
        """Estime la commande musculaire pour atteindre un pattern cible.
        
        Inverse model: étant donné où je suis et où je veux aller,
        quel mouvement dois-je faire?
        
        Args:
            current_pattern: Pattern visuel actuel
            target_pattern: Pattern visuel cible
            
        Returns:
            Commande motrice estimée, ou None si pas d'estimation possible
        """
        best_item = None
        best_score = 0.0
        
        for item in self.memories:
            # Score = similarité pré × similarité post
            sim_pre = item.similarity(current_pattern)
            sim_post = self._pattern_similarity(item.post_pattern, target_pattern)
            score = sim_pre * sim_post
            
            if score > best_score:
                best_score = score
                best_item = item
        
        if best_item is not None and best_score > 0.3:
            return best_item.motor_command.copy()
        return None
    
    def validate_prediction(self, actual_pattern: np.ndarray, predicted_pattern: np.ndarray):
        """Valide une prédiction pour améliorer les stats."""
        sim = self._pattern_similarity(actual_pattern, predicted_pattern)
        if sim > 0.7:
            self.predictions_correct += 1
    
    def _find_similar(
        self, 
        pattern: np.ndarray, 
        motor: np.ndarray
    ) -> Optional[SaccadeMemoryItem]:
        """Trouve l'item le plus similaire dans la mémoire."""
        best_item = None
        best_score = 0.0
        
        for item in self.memories:
            # Similarité pattern + similarité motrice
            sim_pattern = item.similarity(pattern)
            if motor is not None and item.motor_command is not None:
                motor_diff = np.linalg.norm(item.motor_command - motor)
                sim_motor = np.exp(-motor_diff * 2)
            else:
                sim_motor = 0.5
            
            score = sim_pattern * 0.7 + sim_motor * 0.3
            if score > best_score:
                best_score = score
                best_item = item
        
        return best_item if best_score > 0.3 else None
    
    def _pattern_similarity(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Calcule la similarité entre deux patterns."""
        if p1.shape != p2.shape:
            return 0.0
        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        return np.dot(p1.flatten(), p2.flatten()) / (norm1 * norm2)
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de la mémoire."""
        accuracy = self.predictions_correct / max(1, self.predictions_made)
        return {
            'num_memories': len(self.memories),
            'predictions_made': self.predictions_made,
            'predictions_correct': self.predictions_correct,
            'accuracy': accuracy,
            'avg_confidence': np.mean([m.confidence for m in self.memories]) if self.memories else 0.0,
        }


@dataclass
class OculomotorState:
    """État proprioceptif des muscles oculaires virtuels.
    
    Encode les positions et mouvements du regard comme des impulsions
    pour le réseau de neurones. Fournit une rétroaction sensorielle
    bio-inspirée.
    
    6 muscles par œil (comme chez l'humain):
    - Droit médial/latéral: mouvement horizontal
    - Droit supérieur/inférieur: mouvement vertical
    - Oblique supérieur/inférieur: torsion/rotation
    
    Nous encodons:
    - Position X, Y en coordonnées normalisées [-1, 1]
    - Vélocité (dérivée de position)
    - Vergence (différence entre les deux yeux)
    - Rotation/Torsion (cyclorotation)
    - Effort musculaire (magnitude du mouvement)
    """
    # Positions normalisées [-1, 1]
    left_x: float = 0.0
    left_y: float = 0.0
    right_x: float = 0.0
    right_y: float = 0.0
    
    # Rotation/torsion (cyclotorsion) - normalisée [-1, 1]
    left_rotation: float = 0.0
    right_rotation: float = 0.0
    
    # Vélocités (dérivées)
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_vergence: float = 0.0
    vel_rotation: float = 0.0
    
    # Vergence (positif = convergent, négatif = divergent)
    vergence: float = 0.0
    
    # Effort musculaire (magnitude, 0-1)
    effort: float = 0.0
    
    # Historique pour calcul de vélocité
    _prev_x: float = field(default=0.0, repr=False)
    _prev_y: float = field(default=0.0, repr=False)
    _prev_vergence: float = field(default=0.0, repr=False)
    _prev_rotation: float = field(default=0.0, repr=False)
    
    def update(
        self,
        gaze_x: float,
        gaze_y: float,
        vergence_offset: float,
        frame_width: float,
        frame_height: float,
        rotation: float = 0.0
    ):
        """Met à jour l'état proprioceptif.
        
        Args:
            gaze_x, gaze_y: Position du regard en pixels
            vergence_offset: Offset de vergence en pixels
            frame_width, frame_height: Dimensions de l'image
            rotation: Rotation/torsion en radians
        """
        # Normaliser les positions [-1, 1]
        center_x = frame_width / 2
        center_y = frame_height / 2
        
        norm_x = (gaze_x - center_x) / center_x
        norm_y = (gaze_y - center_y) / center_y
        
        # Vergence normalisée (typiquement [-0.1, 0.1])
        norm_verg = vergence_offset / center_x
        
        # Rotation normalisée (typiquement ±30°)
        norm_rot = np.clip(rotation / (np.pi / 6), -1, 1)
        
        # Calculer les vélocités
        self.vel_x = norm_x - self._prev_x
        self.vel_y = norm_y - self._prev_y
        self.vel_vergence = norm_verg - self._prev_vergence
        self.vel_rotation = norm_rot - self._prev_rotation
        
        # Calculer l'effort musculaire (magnitude du mouvement)
        self.effort = np.sqrt(
            self.vel_x**2 + self.vel_y**2 + 
            self.vel_vergence**2 + self.vel_rotation**2
        )
        self.effort = np.clip(self.effort * 5, 0, 1)  # Normaliser
        
        # Mettre à jour les positions
        self._prev_x = norm_x
        self._prev_y = norm_y
        self._prev_vergence = norm_verg
        self._prev_rotation = norm_rot
        
        # Positions des deux yeux
        self.left_x = norm_x + norm_verg / 2
        self.left_y = norm_y
        self.right_x = norm_x - norm_verg / 2
        self.right_y = norm_y
        self.vergence = norm_verg
        
        # Rotation (cyclotorsion) - les yeux peuvent tourner en opposition
        self.left_rotation = norm_rot
        self.right_rotation = -norm_rot  # Opposition pour stabilisation
    
    def get_motor_vector(self) -> np.ndarray:
        """Retourne le vecteur de commande motrice actuel.
        
        Utilisé pour la mémoire saccadique.
        
        Returns:
            Array [dx, dy, d_vergence, d_rotation]
        """
        return np.array([
            self.vel_x,
            self.vel_y,
            self.vel_vergence,
            self.vel_rotation
        ], dtype=np.float32)
    
    def to_spike_encoding(self, resolution: int = 8) -> np.ndarray:
        """Encode l'état en pattern de spikes.
        
        Utilise un encodage de population où chaque "muscle" est
        représenté par plusieurs neurones avec des seuils différents.
        
        Args:
            resolution: Nombre de neurones par dimension
            
        Returns:
            Array de spikes (8 × resolution) pour:
            [pos_x, pos_y, vel_x, vel_y, vergence, vel_vergence, rotation, effort]
        """
        # 8 canaux × resolution neurones
        spikes = np.zeros((8, resolution), dtype=np.float32)
        
        # Encodage de population: chaque neurone a un "seuil préféré"
        thresholds = np.linspace(-1, 1, resolution)
        sigma = 2.0 / resolution  # Largeur du champ récepteur gaussien
        
        values = [
            (self.left_x + self.right_x) / 2,  # Position X moyenne
            (self.left_y + self.right_y) / 2,  # Position Y moyenne
            self.vel_x * 10,                    # Vélocité X (amplifié)
            self.vel_y * 10,                    # Vélocité Y (amplifié)
            self.vergence * 5,                  # Vergence (amplifié)
            self.vel_vergence * 20,             # Vélocité vergence (amplifié)
            self.left_rotation,                 # Rotation (torsion)
            self.effort * 2 - 1,                # Effort musculaire (recentré -1 à 1)
        ]
        
        for i, val in enumerate(values):
            # Activation gaussienne centrée sur la valeur
            val_clipped = np.clip(val, -1, 1)
            spikes[i] = np.exp(-0.5 * ((thresholds - val_clipped) / sigma) ** 2)
        
        return spikes
    
    def get_motor_command_from_spikes(
        self, 
        motor_spikes: np.ndarray,
        resolution: int = 8
    ) -> Tuple[float, float, float, float]:
        """Décode des spikes moteurs en commande de mouvement.
        
        Args:
            motor_spikes: Array (4 × resolution) pour [dx, dy, d_vergence, d_rotation]
            resolution: Nombre de neurones par dimension
            
        Returns:
            (delta_x, delta_y, delta_vergence, delta_rotation) normalisés [-1, 1]
        """
        thresholds = np.linspace(-1, 1, resolution)
        
        deltas = []
        for i in range(min(4, motor_spikes.shape[0])):
            # Décodage par moyenne pondérée
            weights = motor_spikes[i]
            if np.sum(weights) > 0.01:
                delta = np.sum(weights * thresholds) / np.sum(weights)
            else:
                delta = 0.0
            deltas.append(delta)
        
        while len(deltas) < 4:
            deltas.append(0.0)
            
        return tuple(deltas)


# Persistance visuelle des états FIRING (au moins 2 frames écran)
_neuron_display_state: dict = {}  # neuron_id -> (état affiché, compteur frames)

def draw_neuron_stack_2d(stack: NeuronStack, size: int = 180) -> np.ndarray:
    """Visualise le NeuronStack en 2D.
    
    Affiche chaque couche comme une grille avec:
    - Bleu = neurone en activation (firing) - priorité haute, persiste 2 frames
    - Rouge = neurone en inhibition (réfractaire)
    - Vert = neurone normal (dormant)
    
    Args:
        stack: Le NeuronStack à visualiser
        size: Taille de l'image de sortie
        
    Returns:
        Image BGR de la visualisation
    """
    global _neuron_display_state
    from neuronspikes.genesis import NeuronState
    
    num_layers = stack.num_layers
    stats = stack.get_stats()
    
    # Créer l'image (fond sombre)
    viz = np.zeros((size, size, 3), dtype=np.uint8)
    viz[:] = (20, 20, 20)  # Fond gris foncé
    
    # Diviser verticalement par couches
    layer_height = size // num_layers
    
    # Taille fixe des points - très petit
    point_radius = 1
    
    for layer_idx, layer in enumerate(stack.layers):
        y_start = layer_idx * layer_height
        y_end = (layer_idx + 1) * layer_height
        
        # Ligne de séparation entre couches
        cv2.line(viz, (0, y_start), (size, y_start), (40, 40, 40), 1)
        
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
                
                nid = neuron.neuron_id
                
                # Gérer la persistance visuelle (priorité: FIRING > REFRACTORY > autres)
                current_state = neuron.state
                
                # Si FIRING maintenant, enregistrer avec persistance de 2 frames
                if current_state == NeuronState.FIRING:
                    _neuron_display_state[nid] = ('FIRING', 2)
                
                # Récupérer l'état affiché (peut être persisté)
                color = None
                if nid in _neuron_display_state:
                    display_state, remaining = _neuron_display_state[nid]
                    if remaining > 0:
                        # Utiliser l'état persisté (priorité bleu)
                        if display_state == 'FIRING':
                            color = (255, 50, 50)    # Bleu pur vif = activation
                            _neuron_display_state[nid] = (display_state, remaining - 1)
                    else:
                        del _neuron_display_state[nid]
                
                # Si pas de couleur persistée, utiliser l'état réel
                if color is None:
                    if current_state == NeuronState.FIRING:
                        color = (255, 50, 50)    # Bleu pur vif = activation
                    elif current_state == NeuronState.REFRACTORY:
                        color = (50, 50, 255)    # Rouge vif = inhibition
                    elif current_state == NeuronState.CHARGING:
                        # Dégradé cyan selon le potentiel
                        intensity = int(150 + 105 * neuron.potential)
                        color = (intensity, intensity, 50)
                    else:  # DORMANT
                        color = (50, 180, 50)    # Vert clair = normal
                
                # Point de taille fixe, sans contour
                cv2.circle(viz, (px, py), point_radius, color, -1)
        
        # Label de la couche
        label = f"L{layer_idx}:{len(neurons)}"
        cv2.putText(viz, label, (3, y_end - 3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    # Stats globales en haut
    total_n = stats['total_neurons']
    patterns = sum(stats.get('patterns_per_layer', []))
    cv2.putText(viz, f"N:{total_n} P:{patterns}", (size - 65, 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    
    return viz


def draw_retinal_pathways(retinal_result: dict, size: int = 180) -> np.ndarray:
    """Visualise les voies rétiniennes bio-inspirées.
    
    Affiche 4 panneaux:
    - Mouvement (Magno) en haut gauche
    - Couleur R-G (Parvo) en haut droite
    - Orientations (V1) en bas gauche
    - Saillance combinée en bas droite
    
    Args:
        retinal_result: Dictionnaire retourné par RetinalProcessor.process()
        size: Taille de la visualisation en pixels
        
    Returns:
        Image BGR de la visualisation
    """
    if retinal_result is None:
        # Retourner une image vide avec message
        viz = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(viz, "Retinal OFF", (size//4, size//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        return viz
    
    half = size // 2
    viz = np.zeros((size, size, 3), dtype=np.uint8)
    
    def normalize_and_resize(arr, target_size):
        """Normalise [0,1] et redimensionne."""
        if arr is None:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        arr = np.clip(arr, 0, None)
        max_v = arr.max()
        if max_v > 0:
            arr = arr / max_v
        return cv2.resize((arr * 255).astype(np.uint8), (target_size, target_size))
    
    # 1. Mouvement (Magno) - en cyan/magenta
    motion = retinal_result.get('magno', {}).get('motion_inhibited', None)
    if motion is not None:
        motion_img = normalize_and_resize(motion, half)
        # Colormap: cyan pour le mouvement
        viz[0:half, 0:half, 0] = motion_img  # B
        viz[0:half, 0:half, 1] = motion_img // 2  # G
        viz[0:half, 0:half, 2] = 0  # R
    cv2.putText(viz, "M", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 100), 1)
    
    # 2. Couleur R-G (Parvo) - en rouge/vert
    rg = retinal_result.get('parvo', {}).get('red_green', None)
    if rg is not None:
        # R-G: positif = rouge, négatif = vert
        rg_pos = np.clip(rg, 0, 1)
        rg_neg = np.clip(-rg, 0, 1)
        rg_pos_img = normalize_and_resize(rg_pos, half)
        rg_neg_img = normalize_and_resize(rg_neg, half)
        viz[0:half, half:size, 0] = 0  # B
        viz[0:half, half:size, 1] = rg_neg_img  # G
        viz[0:half, half:size, 2] = rg_pos_img  # R
    cv2.putText(viz, "P", (half + 3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 200), 1)
    
    # 3. Orientations (V1) - coloré par angle
    energy = retinal_result.get('v1', {}).get('energy', None)
    orientation = retinal_result.get('v1', {}).get('orientation', None)
    if energy is not None and orientation is not None:
        energy_img = normalize_and_resize(energy, half)
        ori_img = cv2.resize(orientation, (half, half))
        # Colormap HSV par orientation
        hue = ((ori_img / np.pi) * 180).astype(np.uint8)
        sat = np.ones_like(hue) * 255
        val = energy_img
        hsv = np.stack([hue, sat, val], axis=-1)
        ori_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        viz[half:size, 0:half] = ori_bgr
    cv2.putText(viz, "V1", (3, half + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 100, 200), 1)
    
    # 4. Saillance combinée - en jaune/orange
    saliency = retinal_result.get('saliency', None)
    if saliency is not None:
        sal_img = normalize_and_resize(saliency, half)
        # Jaune chaud
        viz[half:size, half:size, 0] = sal_img // 4  # B
        viz[half:size, half:size, 1] = sal_img * 3 // 4  # G
        viz[half:size, half:size, 2] = sal_img  # R
    cv2.putText(viz, "S", (half + 3, half + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 255), 1)
    
    # Bordures entre les panneaux
    viz[half-1:half+1, :] = 50
    viz[:, half-1:half+1] = 50
    
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
    is_left_eye: bool = True,
    direct_vergence: bool = False,
    vergence_correlation: float = 0.0
) -> np.ndarray:
    """Dessine l'overlay de l'agent sur l'image."""
    img = image.copy()
    
    # LABEL BIEN VISIBLE pour identifier l'œil
    eye_label = "GAUCHE" if is_left_eye else "DROITE"
    label_color = (255, 100, 100) if is_left_eye else (100, 100, 255)  # Bleu gauche, Rouge droite
    cv2.putText(img, eye_label, (img.shape[1] // 2 - 50, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
    
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
        # Utiliser x_right pour l'œil droit si disponible
        if not is_left_eye and peak.x_right is not None:
            px = int(peak.x_right)
        else:
            px = int(peak.x)
        py = int(peak.y)
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
    verg_y = bar_y + bar_height + 15
    verg_center = bar_x + bar_width // 2
    verg_scale = 2.0  # pixels par unité de vergence
    
    # Indicateur de type de vergence (directe = mesure binoculaire, corrélation = template matching)
    verg_type = "DIRECT" if direct_vergence else f"NCC:{vergence_correlation:.2f}"
    verg_type_color = (0, 255, 100) if direct_vergence else (180, 180, 180)
    
    # Afficher la valeur numérique avec type
    cv2.putText(img, f"Verg: {vergence_offset:+.1f}px [{verg_type}]", (bar_x, verg_y + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, verg_type_color, 1)
    
    if abs(vergence_offset) > 0.5:
        verg_end = int(verg_center + vergence_offset * verg_scale * (1 if is_left_eye else -1))
        
        # Ligne de base
        cv2.line(img, (bar_x, verg_y), (bar_x + bar_width, verg_y), (80, 80, 80), 1)
        # Marqueur central
        cv2.line(img, (verg_center, verg_y - 3), (verg_center, verg_y + 3), (150, 150, 150), 1)
        # Flèche de vergence - verte si directe, orange/bleu si corrélation
        if direct_vergence:
            verg_color = (0, 255, 100)  # Vert = vergence directe fiable
        else:
            verg_color = (255, 200, 100) if vergence_offset > 0 else (100, 200, 255)
        cv2.arrowedLine(img, (verg_center, verg_y), (verg_end, verg_y), verg_color, 2, tipLength=0.3)
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Agent stéréo avec attention')
    parser.add_argument('-c', '--camera', type=int, default=1,
                        help='Index de la caméra stéréo')
    parser.add_argument('-r', '--rings', type=int, default=48,
                        help='Nombre d\'anneaux (défaut: 48 pour haute résolution)')
    parser.add_argument('-s', '--sectors', type=int, default=32,
                        help='Nombre de secteurs (défaut: 32 pour haute résolution)')
    parser.add_argument('--color', action='store_true',
                        help='Mode couleur avec détection de mouvement (ColorFovea)')
    args = parser.parse_args()
    
    # Initialiser la caméra
    try:
        camera = StereoCamera(args.camera)
    except RuntimeError as e:
        print(f"Erreur: {e}")
        return 1
    
    # Configuration des fovéas - haute résolution bio-inspirée
    # Utilise 80% de la hauteur pour une couverture optimale
    # Plus de cellules = meilleure corrélation stéréo et presque pixel-perfect au centre
    
    # Calculer le rayon max pour couvrir 80% de la hauteur
    target_coverage = 0.80
    max_rad = int(camera.height * target_coverage / 2)  # 288px pour 720p
    
    # Zone fovéale centrale (haute résolution) ~20% du rayon total
    fovea_rad = max(32, max_rad // 4)  # ~72px pour max_rad=288
    
    config = FoveaConfig(
        num_rings=args.rings,
        num_sectors=args.sectors,
        fovea_radius=fovea_rad,   # Zone centrale haute résolution: ~72px
        max_radius=max_rad,       # Rayon total: 80% de hauteur/2
    )
    
    print(f"Fovéa: {max_rad*2}px diamètre ({target_coverage*100:.0f}% de {camera.height}px)")
    print(f"  {args.rings} anneaux × {args.sectors} secteurs = {args.rings * args.sectors} cellules")
    print(f"  Résolution centrale: ~{fovea_rad / (args.rings // 4):.1f}px/anneau")
    
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
            direct_verg = result.get('direct_vergence', False)
            verg_corr = result.get('vergence_correlation', 0.0)
            
            left_viz = draw_agent_overlay(
                left, 
                result['gaze_x'],  # Position œil gauche (directe)
                result['gaze_y'],
                config,
                result['in_saccade'],
                result['fixating'],
                result['saliency_peaks'],
                result['correlation'],
                vergence_offset=vergence,
                is_left_eye=True,
                direct_vergence=direct_verg,
                vergence_correlation=verg_corr
            )
            right_viz = draw_agent_overlay(
                right,
                result['gaze_x_right'],  # Position œil droit (indépendante!)
                result['gaze_y'],
                config,
                result['in_saccade'],
                result['fixating'],
                result['saliency_peaks'],
                result['correlation'],
                vergence_offset=vergence,
                is_left_eye=False,
                direct_vergence=direct_verg,
                vergence_correlation=verg_corr
            )
            
            # Visualisation des fovéas
            fovea_size = 180
            left_fovea_viz = visualize_fovea(agent.left_fovea, size=fovea_size)
            right_fovea_viz = visualize_fovea(agent.right_fovea, size=fovea_size)
            
            # Visualisation du NeuronStack
            stack_viz = draw_neuron_stack_2d(agent.neuron_stack, size=fovea_size)
            
            # Visualisation des voies rétiniennes bio-inspirées
            retinal_viz = draw_retinal_pathways(result.get('retinal_result'), size=fovea_size)
            
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
            
            # Ligne 2: fovéas, analyse, neurones et voies rétiniennes
            fovea_row = np.hstack([left_fovea_viz, analysis_viz, stack_viz, retinal_viz, right_fovea_viz])
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
            
            # Affichage mémoire saccadique et effort musculaire
            saccade_stats = result.get('saccade_memory_stats', {})
            mem_text = f"SacMem: {saccade_stats.get('num_memories', 0)} Effort: {result.get('motor_effort', 0):.2f}"
            cv2.putText(display, mem_text, (display_w + 10, 105),
                       font, 0.4, (200, 150, 255), 1)
            
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
