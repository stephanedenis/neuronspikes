#!/usr/bin/env python3
"""
Live Stereo Fovea - Vision stéréoscopique avec rétines polaires.

Utilise une caméra stéréo side-by-side (ex: Gearway SPCA2100)
pour alimenter deux fovéas et calculer la disparité/profondeur.

Corrélation des objets communs entre les deux rétines.

Usage:
    python examples/live_stereo.py [options]
    
Options:
    -c, --camera INDEX    Index de la caméra stéréo (défaut: 1)
    -r, --rings N         Nombre d'anneaux de la fovéa (défaut: 16)
    -s, --sectors N       Nombre de secteurs angulaires (défaut: 16)
    --baseline PIXELS     Distance inter-oculaire simulée (défaut: 60)

Touches:
    q, ESC  Quitter
    SPACE   Pause/Reprendre
    r       Reset des fovéas
    +/-     Ajuster le seuil de disparité
    g       Afficher/Masquer la grille polaire
    d       Mode disparité / mode normal
"""

import argparse
import time
import sys
import cv2
import numpy as np

# Ajouter le chemin du projet
sys.path.insert(0, str(__file__).rsplit('/', 2)[0] + '/src')

from neuronspikes import (
    Fovea,
    FoveaConfig,
    StereoFovea,
    visualize_fovea,
)


class StereoCamera:
    """Capture stéréo side-by-side."""
    
    def __init__(self, camera_index: int = 1, width: int = 1280, height: int = 480):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra {camera_index}")
        
        # Configurer pour mode stéréo side-by-side
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Lire une frame pour obtenir les dimensions réelles
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Impossible de lire une frame")
        
        self.height, self.width = frame.shape[:2]
        self.half_width = self.width // 2
        
        print(f"Caméra stéréo: {self.width}x{self.height} (2x {self.half_width}x{self.height})")
    
    def read(self):
        """Lit une paire stéréo.
        
        Returns:
            Tuple (success, left_frame, right_frame)
        """
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None
        
        # Séparer gauche et droite
        left = frame[:, :self.half_width]
        right = frame[:, self.half_width:]
        
        return True, left, right
    
    def release(self):
        self.cap.release()


def create_disparity_visualization(
    left_act: np.ndarray,
    right_act: np.ndarray,
    size: int = 256
) -> np.ndarray:
    """Visualise la disparité entre les deux fovéas.
    
    Args:
        left_act: Activations de la fovéa gauche
        right_act: Activations de la fovéa droite
        size: Taille de l'image de sortie
        
    Returns:
        Image RGB de la disparité
    """
    # Calculer la disparité (différence)
    disparity = left_act - right_act
    
    # Normaliser entre -1 et 1
    max_disp = max(np.abs(disparity).max(), 0.01)
    disparity_norm = disparity / max_disp
    
    # Créer image colorée
    # Rouge = plus proche (disparité positive forte)
    # Bleu = plus loin (disparité négative forte)
    # Vert = même distance
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    h, w = disparity_norm.shape
    cell_h = size // h
    cell_w = size // w
    
    for i in range(h):
        for j in range(w):
            d = disparity_norm[i, j]
            
            if d > 0:
                # Proche - rouge
                color = (0, int((1-d) * 128), int(d * 255))
            else:
                # Loin - bleu
                color = (int(-d * 255), int((1+d) * 128), 0)
            
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            img[y1:y2, x1:x2] = color
    
    return img


def create_correlation_map(
    left_act: np.ndarray,
    right_act: np.ndarray,
    size: int = 256
) -> np.ndarray:
    """Visualise la corrélation entre les deux rétines.
    
    Les zones avec forte corrélation (objets communs) sont brillantes.
    
    Args:
        left_act: Activations de la fovéa gauche
        right_act: Activations de la fovéa droite
        size: Taille de l'image de sortie
        
    Returns:
        Image RGB de corrélation
    """
    # Corrélation = produit des activations normalisées
    # Les zones actives dans les deux yeux sont corrélées
    correlation = left_act * right_act
    
    # Normaliser
    max_corr = max(correlation.max(), 0.01)
    correlation_norm = correlation / max_corr
    
    # Créer image
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    h, w = correlation_norm.shape
    cell_h = size // h
    cell_w = size // w
    
    for i in range(h):
        for j in range(w):
            c = correlation_norm[i, j]
            
            # Jaune-blanc pour forte corrélation
            brightness = int(c * 255)
            color = (brightness // 2, brightness, brightness)
            
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            img[y1:y2, x1:x2] = color
    
    return img


def main():
    parser = argparse.ArgumentParser(description='Vision stéréo avec fovéas polaires')
    parser.add_argument('-c', '--camera', type=int, default=1,
                        help='Index de la caméra stéréo')
    parser.add_argument('-r', '--rings', type=int, default=16,
                        help='Nombre d\'anneaux')
    parser.add_argument('-s', '--sectors', type=int, default=16,
                        help='Nombre de secteurs')
    parser.add_argument('--baseline', type=float, default=60.0,
                        help='Distance inter-oculaire (pixels)')
    args = parser.parse_args()
    
    # Initialiser la caméra stéréo
    try:
        camera = StereoCamera(args.camera, width=1280, height=480)
    except RuntimeError as e:
        print(f"Erreur: {e}")
        print("Essayez avec -c 0 ou -c 2")
        return 1
    
    # Configuration des fovéas
    config = FoveaConfig(
        num_rings=args.rings,
        num_sectors=args.sectors,
        fovea_radius=camera.half_width // 8,
        max_radius=min(camera.half_width, camera.height) // 2,
    )
    
    # Créer le système stéréo
    stereo = StereoFovea(config, baseline=args.baseline)
    
    # Position initiale (centre de chaque image)
    center_x = camera.half_width // 2
    center_y = camera.height // 2
    stereo.set_target(center_x, center_y, depth=100)
    
    # État
    paused = False
    show_grid = True
    disparity_mode = False
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    print("\n=== Vision Stéréo avec Fovéas Polaires ===")
    print("Touches: q=quitter, SPACE=pause, r=reset, g=grille, d=disparité")
    print("Clic souris: déplacer le point de fixation")
    print("")
    
    # Fenêtre avec callback souris
    window_name = "Stereo Fovea"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal stereo, center_x, center_y
        if event == cv2.EVENT_LBUTTONDOWN:
            # Calculer la position relative dans l'une des vues
            panel_width = camera.half_width
            if x < panel_width:
                # Clic dans la vue gauche
                center_x = x
                center_y = y
            elif x < panel_width * 2:
                # Clic dans la vue droite
                center_x = x - panel_width
                center_y = y
            stereo.set_target(center_x, center_y, depth=100)
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    try:
        while True:
            if not paused:
                ret, left, right = camera.read()
                if not ret:
                    print("Erreur de lecture caméra")
                    break
                
                # Convertir en grayscale
                left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                
                # Échantillonner avec les fovéas
                left_act, right_act = stereo.sample(left_gray, right_gray)
                
                frame_count += 1
                
                # Calculer FPS
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()
            
            # Construire la visualisation
            viz_size = 200
            
            # Visualisation des fovéas
            left_fovea_viz = visualize_fovea(stereo.left, size=viz_size, show_grid=show_grid)
            right_fovea_viz = visualize_fovea(stereo.right, size=viz_size, show_grid=show_grid)
            
            # Disparité ou corrélation
            if disparity_mode:
                analysis_viz = create_disparity_visualization(left_act, right_act, viz_size)
                analysis_label = "Disparite"
            else:
                analysis_viz = create_correlation_map(left_act, right_act, viz_size)
                analysis_label = "Correlation"
            
            # Dessiner les points de fixation sur les images originales
            left_display = left.copy()
            right_display = right.copy()
            
            # Croix de fixation
            cx_l, cy_l = int(stereo.left.gaze.x), int(stereo.left.gaze.y)
            cx_r, cy_r = int(stereo.right.gaze.x), int(stereo.right.gaze.y)
            
            cv2.drawMarker(left_display, (cx_l, cy_l), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(right_display, (cx_r, cy_r), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Cercle de la zone fovéale
            fovea_r = config.fovea_radius
            cv2.circle(left_display, (cx_l, cy_l), fovea_r, (0, 255, 255), 1)
            cv2.circle(right_display, (cx_r, cy_r), fovea_r, (0, 255, 255), 1)
            
            # Cercle de la zone périphérique
            max_r = config.max_radius
            cv2.circle(left_display, (cx_l, cy_l), max_r, (255, 255, 0), 1)
            cv2.circle(right_display, (cx_r, cy_r), max_r, (255, 255, 0), 1)
            
            # Redimensionner pour l'affichage
            display_h = viz_size
            scale = display_h / camera.height
            display_w = int(camera.half_width * scale)
            
            left_small = cv2.resize(left_display, (display_w, display_h))
            right_small = cv2.resize(right_display, (display_w, display_h))
            
            # Assembler le panneau
            # Ligne 1: images caméra gauche + droite
            # Ligne 2: fovéa gauche + analyse + fovéa droite
            
            row1 = np.hstack([left_small, right_small])
            
            # Ajuster la largeur de la ligne 2
            row2_width = row1.shape[1]
            fovea_row = np.hstack([left_fovea_viz, analysis_viz, right_fovea_viz])
            
            # Redimensionner row2 pour correspondre à row1
            if fovea_row.shape[1] != row2_width:
                fovea_row = cv2.resize(fovea_row, (row2_width, viz_size))
            
            # Combiner
            display = np.vstack([row1, fovea_row])
            
            # Ajouter les labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display, "Gauche", (10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "Droite", (display_w + 10, 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, display_h - 10), font, 0.5, (0, 255, 0), 1)
            
            cv2.putText(display, "Fovea G", (10, display_h + 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(display, analysis_label, (viz_size + 10, display_h + 20), font, 0.5, (255, 255, 255), 1)
            cv2.putText(display, "Fovea D", (viz_size * 2 + 10, display_h + 20), font, 0.5, (255, 255, 255), 1)
            
            # Stats
            stats = stereo.get_stats()
            vergence_deg = np.degrees(stats['vergence'])
            cv2.putText(display, f"Vergence: {vergence_deg:.1f} deg", 
                       (10, display.shape[0] - 10), font, 0.4, (200, 200, 200), 1)
            
            cv2.imshow(window_name, display)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            
            if key in (ord('q'), 27):  # q ou ESC
                break
            elif key == ord(' '):
                paused = not paused
                print("Pause" if paused else "Reprise")
            elif key == ord('r'):
                stereo.left.reset()
                stereo.right.reset()
                stereo.set_target(center_x, center_y, depth=100)
                print("Reset")
            elif key == ord('g'):
                show_grid = not show_grid
            elif key == ord('d'):
                disparity_mode = not disparity_mode
                print("Mode:", "Disparité" if disparity_mode else "Corrélation")
            elif key == ord('+') or key == ord('='):
                stereo.set_target(center_x, center_y, depth=stats['target_depth'] * 0.9)
                print(f"Profondeur: {stats['target_depth']:.1f}")
            elif key == ord('-'):
                stereo.set_target(center_x, center_y, depth=stats['target_depth'] * 1.1)
                print(f"Profondeur: {stats['target_depth']:.1f}")
    
    except KeyboardInterrupt:
        print("\nInterruption...")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
