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

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Forcer le backend GTK au lieu de Qt (évite le clignotement)
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

import cv2
import numpy as np
from numpy.typing import NDArray

# Ajouter le chemin src pour l'import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuronspikes import (
    create_retina, RetinaLayer, 
    GroupDetector, GroupDetectorConfig, visualize_groups,
    NeuronLayer, GenesisConfig, NeuronConfig, visualize_neurons,
)
from neuronspikes.temporal import TemporalCorrelator, CorrelationConfig, visualize_patterns


@dataclass
class VisualizerConfig:
    """Configuration du visualiseur."""
    camera_id: int = 0
    retina_width: int = 128
    retina_height: int = 128
    window_name: str = "NeuronSpikes - Rétine Artificielle"
    target_fps: int = 60
    show_stats: bool = True
    show_groups: bool = True  # Afficher les groupes d'activation
    show_patterns: bool = True  # Afficher les patterns temporels
    intensity_threshold: int = 200  # Seuil pour détecter les activations "fortes"


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
        
        # Détecteur de groupes d'activation
        self.group_detector = GroupDetector(GroupDetectorConfig(
            min_group_size=3,  # Au moins 3 pixels pour former un groupe
            connectivity=8,    # 8-connexité pour les diagonales
            track_history=10   # Garder 10 frames d'historique
        ))
        
        # Corrélateur temporel pour détecter les patterns récurrents
        self.correlator = TemporalCorrelator(
            shape=(self.config.retina_height, self.config.retina_width),
            config=CorrelationConfig(
                history_size=30,       # 0.5 seconde à 60 fps
                min_overlap=0.5,       # 50% de chevauchement minimum
                min_occurrences=5,     # Vu au moins 5 fois pour être stable
                confidence_threshold=0.4,
                decay_rate=0.95
            )
        )
        self._stable_pattern_count: int = 0
        
        # Couche de neurones créés à partir des patterns stables
        self.neuron_layer = NeuronLayer(
            layer_id=1,
            shape=(self.config.retina_height, self.config.retina_width),
            config=GenesisConfig(
                min_pattern_confidence=0.5,
                min_pattern_occurrences=10,
                max_neurons_per_layer=100,
                prune_inactive_after=300  # 5 secondes à 60 fps
            ),
            neuron_config=NeuronConfig(
                threshold=0.6,
                decay_rate=0.15,
                refractory_period=5
            )
        )
        
        # Capture vidéo
        self.cap: cv2.VideoCapture | None = None
        
        # Buffer pour éviter le clignotement (garde la dernière frame valide)
        self._last_valid_frame: NDArray[np.uint8] | None = None
        self._last_display: NDArray[np.uint8] | None = None
        self._last_groups_count: int = 0
        
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
        # Réduire le buffer de la caméra pour moins de latence
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Caméra ouverte: {actual_width}x{actual_height} @ {actual_fps:.1f} fps")
        return True
    
    def process_frame(self, frame: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
        """Traite une frame et retourne l'entrée et les sorties.
        
        Args:
            frame: Image BGR de la caméra
            
        Returns:
            Tuple (image_mono_resized, retina_output, groups_image, patterns_image, neurons_image)
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
        retina_output = self.retina.get_activation_pattern()
        
        # Détecter les groupes d'activation sur les pixels "brillants"
        # On applique un seuil pour ne garder que les activations fortes
        activations = gray_resized > self.config.intensity_threshold
        groups = self.group_detector.detect_groups(
            activations, 
            slot=0, 
            frame=self.frame_count
        )
        self._last_groups_count = len(groups)
        
        # Créer l'image des groupes
        groups_img = visualize_groups(activations, groups)
        
        # Traiter les corrélations temporelles
        self.correlator.process_groups(groups)
        
        # Créer des neurones à partir des nouveaux patterns stables
        for pattern in self.correlator.stable_patterns:
            self.neuron_layer.create_neuron_from_pattern(pattern)
        
        # Créer l'image des patterns stables
        pattern_map = self.correlator.get_pattern_map()
        confidence_map = self.correlator.get_confidence_map()
        patterns_img = visualize_patterns(pattern_map, confidence_map)
        
        # Traiter avec la couche de neurones
        neuron_output = self.neuron_layer.process(activations)
        neurons_img = visualize_neurons(self.neuron_layer, show_potentials=True)
        
        # Stats patterns et neurones
        self._stable_pattern_count = len(self.correlator.stable_patterns)
        
        return gray_resized, retina_output, groups_img, patterns_img, neurons_img
    
    def create_display(
        self, 
        input_img: NDArray[np.uint8], 
        retina_img: NDArray[np.uint8],
        groups_img: NDArray[np.uint8],
        patterns_img: NDArray[np.uint8],
        neurons_img: NDArray[np.uint8],
        fps: float
    ) -> NDArray[np.uint8]:
        """Crée l'image d'affichage combinée.
        
        Args:
            input_img: Image d'entrée monochrome
            retina_img: Sortie de la rétine
            groups_img: Image des groupes d'activation (RGB)
            patterns_img: Image des patterns temporels (RGB)
            neurons_img: Image des neurones (RGB)
            fps: FPS actuel
            
        Returns:
            Image combinée BGR pour affichage
        """
        # Taille d'affichage (upscale pour visibilité)
        display_size = 200  # 5 panneaux de 200px = 1000px total
        
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
        groups_display = cv2.resize(
            groups_img, 
            (display_size, display_size), 
            interpolation=cv2.INTER_NEAREST
        )
        patterns_display = cv2.resize(
            patterns_img, 
            (display_size, display_size), 
            interpolation=cv2.INTER_NEAREST
        )
        neurons_display = cv2.resize(
            neurons_img, 
            (display_size, display_size), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Convertir en BGR pour l'affichage
        input_bgr = cv2.cvtColor(input_display, cv2.COLOR_GRAY2BGR)
        
        # Colormap pour la rétine (plus visuel)
        retina_colored = cv2.applyColorMap(retina_display, cv2.COLORMAP_INFERNO)
        
        # Groupes, patterns et neurones: RGB -> BGR pour OpenCV
        groups_bgr = cv2.cvtColor(groups_display, cv2.COLOR_RGB2BGR)
        patterns_bgr = cv2.cvtColor(patterns_display, cv2.COLOR_RGB2BGR)
        neurons_bgr = cv2.cvtColor(neurons_display, cv2.COLOR_RGB2BGR)
        
        # Combiner horizontalement (5 panneaux)
        combined = np.hstack([input_bgr, retina_colored, groups_bgr, patterns_bgr, neurons_bgr])
        
        # Ajouter les labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        neuron_count = self.neuron_layer.neuron_count
        cv2.putText(combined, "ENTREE", (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "RETINE", (display_size + 10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, f"GROUPES ({self._last_groups_count})", (display_size * 2 + 5, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(combined, f"PATTERNS ({self._stable_pattern_count})", (display_size * 3 + 5, 20), font, 0.4, (255, 255, 255), 1)
        cv2.putText(combined, f"NEURONES ({neuron_count})", (display_size * 4 + 5, 20), font, 0.4, (255, 255, 255), 1)
        
        if self.config.show_stats:
            # Stats en bas
            stats_y = display_size - 10
            correlator_stats = self.correlator.get_stats()
            layer_stats = self.neuron_layer.get_stats()
            cv2.putText(
                combined, 
                f"FPS: {fps:.1f} | F: {self.frame_count} | P: {correlator_stats['active_patterns']} | N: {neuron_count} | Spikes: {layer_stats['total_spikes']}",
                (10, stats_y), 
                font, 0.4, (0, 255, 0), 1
            )
            
            # Afficher les stats de la rétine
            total_spikes = self.retina.stats['total_spikes']
            cv2.putText(
                combined,
                f"Spikes: {total_spikes:,}",
                (display_size + 10, stats_y),
                font, 0.5, (0, 255, 255), 1
            )
            
            # Afficher les stats des groupes
            total_groups = self.group_detector.stats['total_groups']
            cv2.putText(
                combined,
                f"Total: {total_groups:,}",
                (display_size * 2 + 10, stats_y),
                font, 0.5, (0, 255, 255), 1
            )
        
        # Lignes de séparation
        cv2.line(combined, (display_size, 0), (display_size, display_size), (128, 128, 128), 1)
        cv2.line(combined, (display_size * 2, 0), (display_size * 2, display_size), (128, 128, 128), 1)
        
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
                
                # Gérer les frames corrompues ou manquantes
                if not ret or frame is None or frame.size == 0:
                    # Utiliser la dernière frame valide si disponible
                    if self._last_display is not None:
                        cv2.imshow(self.config.window_name, self._last_display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    continue
                
                # Vérifier que la frame est valide (pas de pixels aberrants)
                if np.mean(frame) < 1 or np.mean(frame) > 254:
                    if self._last_display is not None:
                        cv2.imshow(self.config.window_name, self._last_display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    continue
                
                # Sauvegarder la frame valide
                self._last_valid_frame = frame.copy()
                
                # Traiter (retourne maintenant 5 images)
                input_img, retina_img, groups_img, patterns_img, neurons_img = self.process_frame(frame)
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
                
                # Créer l'affichage (5 panneaux)
                display = self.create_display(input_img, retina_img, groups_img, patterns_img, neurons_img, fps)
                self._last_display = display  # Buffer pour éviter clignotement
                
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
            # Recréer le corrélateur avec les nouvelles dimensions
            self.correlator = TemporalCorrelator(
                shape=(new_height, new_width),
                config=self.correlator.config
            )
            # Recréer la couche de neurones
            self.neuron_layer = NeuronLayer(
                layer_id=1,
                shape=(new_height, new_width),
                config=self.neuron_layer.config,
                neuron_config=self.neuron_layer.neuron_config
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
        correlator_stats = self.correlator.get_stats()
        layer_stats = self.neuron_layer.get_stats()
        
        print()
        print("═" * 60)
        print("Statistiques finales:")
        print(f"  • Durée: {elapsed:.1f} secondes")
        print(f"  • Frames traitées: {self.frame_count}")
        print(f"  • FPS moyen: {avg_fps:.1f}")
        print(f"  • Impulsions rétine: {self.retina.stats['total_spikes']:,}")
        print(f"  • Patterns créés: {correlator_stats['total_created']}")
        print(f"  • Patterns stables: {correlator_stats['stable_patterns']}")
        print(f"  • Neurones créés: {layer_stats['total_created']}")
        print(f"  • Neurones actifs: {layer_stats['neuron_count']}")
        print(f"  • Spikes neurones: {layer_stats['total_spikes']}")
        print("═" * 60)


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
