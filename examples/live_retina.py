#!/usr/bin/env python3
"""
Visualisation en temps réel de la rétine artificielle NeuronSpikes.

Ce script capture le flux vidéo d'une caméra et affiche:
- À gauche: l'image source (monochrome)
- À droite: la rétine (intensité = cumul d'activations par frame)

Contrôles:
- 'q' ou ESC: Quitter
- 's': Sauvegarder une capture
- 'r': Reset des statistiques
- '+'/'-': Ajuster la résolution de la rétine
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

# Ajouter le chemin src pour l'import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuronspikes import create_retina, RetinaLayer


@dataclass
class VisualizerConfig:
    """Configuration du visualiseur."""
    camera_id: int = 0
    retina_width: int = 128
    retina_height: int = 128
    window_name: str = "NeuronSpikes - Rétine Artificielle"
    target_fps: int = 60
    show_stats: bool = True


class RetinaVisualizer:
    """Visualiseur temps réel de la rétine artificielle."""
    
    def __init__(self, config: VisualizerConfig | None = None):
        """Initialise le visualiseur.
        
        Args:
            config: Configuration du visualiseur
        """
        self.config = config or VisualizerConfig()
        
        # Créer la rétine
        self.retina = create_retina(
            width=self.config.retina_width,
            height=self.config.retina_height,
            fps=self.config.target_fps
        )
        
        # Capture vidéo
        self.cap: cv2.VideoCapture | None = None
        
        # Statistiques
        self.fps_history: list[float] = []
        self.frame_count = 0
        self.start_time = time.time()
        
    def start_capture(self) -> bool:
        """Démarre la capture vidéo.
        
        Returns:
            True si la capture a démarré avec succès
        """
        self.cap = cv2.VideoCapture(self.config.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Impossible d'ouvrir la caméra {self.config.camera_id}")
            return False
        
        # Configurer la caméra pour la meilleure performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Caméra ouverte: {actual_width}x{actual_height} @ {actual_fps:.1f} fps")
        return True
    
    def process_frame(self, frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Traite une frame et retourne l'entrée et la sortie rétine.
        
        Args:
            frame: Image BGR de la caméra
            
        Returns:
            Tuple (image_mono_resized, retina_output)
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Redimensionner pour la rétine
        gray_resized = cv2.resize(
            gray, 
            (self.config.retina_width, self.config.retina_height),
            interpolation=cv2.INTER_AREA
        )
        
        # Traiter avec la rétine
        self.retina.process_frame(gray_resized)
        
        # Obtenir le pattern d'activation (équivalent à l'intensité)
        # C'est le cumul des activations pour la frame
        retina_output = self.retina.get_activation_pattern()
        
        return gray_resized, retina_output
    
    def create_display(
        self, 
        input_img: NDArray[np.uint8], 
        retina_img: NDArray[np.uint8],
        fps: float
    ) -> NDArray[np.uint8]:
        """Crée l'image d'affichage combinée.
        
        Args:
            input_img: Image d'entrée monochrome
            retina_img: Sortie de la rétine
            fps: FPS actuel
            
        Returns:
            Image combinée BGR pour affichage
        """
        # Taille d'affichage (upscale pour visibilité)
        display_size = 384
        
        # Upscale les images
        input_display = cv2.resize(
            input_img, 
            (display_size, display_size), 
            interpolation=cv2.INTER_NEAREST
        )
        retina_display = cv2.resize(
            retina_img, 
            (display_size, display_size), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convertir en BGR pour l'affichage
        input_bgr = cv2.cvtColor(input_display, cv2.COLOR_GRAY2BGR)
        
        # Colormap pour la rétine (plus visuel)
        retina_colored = cv2.applyColorMap(retina_display, cv2.COLORMAP_INFERNO)
        
        # Combiner horizontalement
        combined = np.hstack([input_bgr, retina_colored])
        
        # Ajouter les labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "ENTREE (Camera)", (10, 25), font, 0.6, (255, 255, 255), 1)
        cv2.putText(combined, "RETINE (Activations)", (display_size + 10, 25), font, 0.6, (255, 255, 255), 1)
        
        if self.config.show_stats:
            # Stats en bas
            stats_y = display_size - 10
            cv2.putText(
                combined, 
                f"FPS: {fps:.1f} | Frames: {self.frame_count} | Res: {self.config.retina_width}x{self.config.retina_height}",
                (10, stats_y), 
                font, 0.5, (0, 255, 0), 1
            )
            
            # Afficher les stats de la rétine
            total_spikes = self.retina.stats['total_spikes']
            cv2.putText(
                combined,
                f"Spikes: {total_spikes:,}",
                (display_size + 10, stats_y),
                font, 0.5, (0, 255, 255), 1
            )
        
        # Ligne de séparation
        cv2.line(combined, (display_size, 0), (display_size, display_size), (128, 128, 128), 1)
        
        return combined
    
    def run(self):
        """Boucle principale du visualiseur."""
        if not self.start_capture():
            return
        
        print()
        print("╔════════════════════════════════════════════╗")
        print("║    NeuronSpikes - Rétine en temps réel     ║")
        print("╠════════════════════════════════════════════╣")
        print("║  q/ESC: Quitter                            ║")
        print("║  s: Sauvegarder capture                    ║")
        print("║  r: Reset statistiques                     ║")
        print("║  +/-: Ajuster résolution rétine            ║")
        print("╚════════════════════════════════════════════╝")
        print()
        
        cv2.namedWindow(self.config.window_name, cv2.WINDOW_AUTOSIZE)
        
        last_time = time.time()
        fps = 0.0
        
        try:
            while True:
                # Capturer une frame
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Erreur de lecture caméra")
                    break
                
                # Traiter
                input_img, retina_img = self.process_frame(frame)
                self.frame_count += 1
                
                # Calculer FPS
                current_time = time.time()
                dt = current_time - last_time
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self.fps_history.append(instant_fps)
                    if len(self.fps_history) > 30:
                        self.fps_history.pop(0)
                    fps = sum(self.fps_history) / len(self.fps_history)
                last_time = current_time
                
                # Créer l'affichage
                display = self.create_display(input_img, retina_img, fps)
                
                # Afficher
                cv2.imshow(self.config.window_name, display)
                
                # Gérer les touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q ou ESC
                    break
                elif key == ord('s'):
                    # Sauvegarder
                    filename = f"capture_{int(time.time())}.png"
                    cv2.imwrite(filename, display)
                    print(f"📸 Capture sauvegardée: {filename}")
                elif key == ord('r'):
                    # Reset
                    self.retina.reset()
                    self.frame_count = 0
                    self.fps_history.clear()
                    print("🔄 Statistiques réinitialisées")
                elif key == ord('+') or key == ord('='):
                    # Augmenter résolution
                    self.resize_retina(2.0)
                elif key == ord('-'):
                    # Diminuer résolution
                    self.resize_retina(0.5)
                    
        except KeyboardInterrupt:
            print("\n⏹️  Arrêt demandé")
        finally:
            self.cleanup()
    
    def resize_retina(self, factor: float):
        """Redimensionne la rétine.
        
        Args:
            factor: Facteur de redimensionnement
        """
        new_width = max(32, min(512, int(self.config.retina_width * factor)))
        new_height = max(32, min(512, int(self.config.retina_height * factor)))
        
        if new_width != self.config.retina_width:
            self.config.retina_width = new_width
            self.config.retina_height = new_height
            self.retina = create_retina(
                width=new_width,
                height=new_height,
                fps=self.config.target_fps
            )
            print(f"📐 Nouvelle résolution rétine: {new_width}x{new_height}")
    
    def cleanup(self):
        """Nettoie les ressources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Afficher les stats finales
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print()
        print("═" * 50)
        print("Statistiques finales:")
        print(f"  • Durée: {elapsed:.1f} secondes")
        print(f"  • Frames traitées: {self.frame_count}")
        print(f"  • FPS moyen: {avg_fps:.1f}")
        print(f"  • Impulsions totales: {self.retina.stats['total_spikes']:,}")
        print("═" * 50)


def main():
    """Point d'entrée principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualiseur de rétine artificielle NeuronSpikes"
    )
    parser.add_argument(
        "-c", "--camera", 
        type=int, 
        default=0,
        help="ID de la caméra (défaut: 0)"
    )
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        default=128,
        help="Résolution de la rétine (défaut: 128)"
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Masquer les statistiques"
    )
    
    args = parser.parse_args()
    
    config = VisualizerConfig(
        camera_id=args.camera,
        retina_width=args.resolution,
        retina_height=args.resolution,
        show_stats=not args.no_stats
    )
    
    visualizer = RetinaVisualizer(config)
    visualizer.run()


if __name__ == "__main__":
    main()
