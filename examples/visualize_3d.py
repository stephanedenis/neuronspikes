#!/usr/bin/env python3
"""
Visualiseur 3D de la pile de neurones.

Affiche la hiérarchie de neurones en temps réel avec:
- Chaque couche comme un plan horizontal
- Les neurones comme des sphères colorées
- Animation de l'activité

Usage:
    python examples/visualize_3d.py [--layers N] [--demo]
    
Contrôles:
    Souris gauche + drag: Rotation
    Souris droite + drag: Zoom
    Molette: Zoom
    Souris milieu + drag: Pan
    Space: Pause/Resume
    R: Reset vue
    A: Toggle auto-rotation
    +/-: Vitesse rotation
    Q/ESC: Quitter
"""

import argparse
import numpy as np
import time

from neuronspikes import (
    NeuronStack,
    NeuronVisualizer3D,
    VisualizerConfig,
    TemporalPattern,
    GenesisConfig,
    NeuronConfig,
)


def create_demo_stack(num_layers: int = 4) -> NeuronStack:
    """Crée une pile de démonstration avec des neurones."""
    stack = NeuronStack(
        base_shape=(64, 64),
        num_layers=num_layers,
        reduction_factor=0.5,
        config=GenesisConfig(
            min_pattern_confidence=0.5,
            min_pattern_occurrences=5,
        )
    )
    
    # Ajouter des neurones de démo dans chaque couche
    for layer_idx, layer in enumerate(stack.layers):
        h, w = layer.shape
        # Plus de neurones dans les couches basses
        num_neurons = max(5, 30 - layer_idx * 7)
        
        for i in range(num_neurons):
            # Position pseudo-aléatoire mais déterministe
            x = (i * 7 + layer_idx * 13) % (w - 5)
            y = (i * 11 + layer_idx * 17) % (h - 5)
            size = 3 + layer_idx  # Plus gros dans les couches hautes
            
            signature = np.zeros((h, w), dtype=bool)
            y1, y2 = max(0, y), min(h, y + size)
            x1, x2 = max(0, x), min(w, x + size)
            signature[y1:y2, x1:x2] = True
            
            pattern = TemporalPattern(
                pattern_id=layer_idx * 1000 + i,
                signature=signature,
                confidence=0.9,
                occurrences=20,
                first_seen=0,
                last_seen=10,
            )
            layer.create_neuron_from_pattern(pattern)
    
    return stack


def main():
    # Parser manuel car argparse a des conflits avec -h
    import sys
    
    # Valeurs par défaut
    num_layers = 4
    demo_mode = True
    width = 1280
    height = 720
    
    args_list = sys.argv[1:]
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg in ['-l', '--layers'] and i + 1 < len(args_list):
            num_layers = int(args_list[i + 1])
            i += 2
        elif arg == '--demo':
            demo_mode = True
            i += 1
        elif arg in ['-w', '--width'] and i + 1 < len(args_list):
            width = int(args_list[i + 1])
            i += 2
        elif arg == '--height' and i + 1 < len(args_list):
            height = int(args_list[i + 1])
            i += 2
        elif arg in ['--help']:
            print(__doc__)
            return 0
        else:
            i += 1
    
    print(f"Création de la pile avec {num_layers} couches...")
    stack = create_demo_stack(num_layers)
    
    stats = stack.get_stats()
    print(f"Stack créé:")
    print(f"  Couches: {stats['num_layers']}")
    print(f"  Neurones total: {stats['total_neurons']}")
    for i, count in enumerate(stats['neurons_per_layer']):
        shape = stack.layers[i].shape
        print(f"    Couche {i}: {count} neurones ({shape[0]}x{shape[1]})")
    
    # Configuration du visualiseur
    config = VisualizerConfig(
        width=width,
        height=height,
        title=f"NeuronSpikes 3D - {num_layers} couches, {stats['total_neurons']} neurones"
    )
    
    # Callback de mise à jour (simule l'activité)
    frame_count = [0]
    last_time = [time.time()]
    
    def update():
        frame_count[0] += 1
        
        # Limiter le taux de mise à jour
        current = time.time()
        if current - last_time[0] < 0.05:  # 20 Hz max pour l'activité
            return
        last_time[0] = current
        
        if demo_mode:
            # Simuler des entrées qui déclenchent l'activité
            h, w = stack.base_shape
            
            # Pattern qui bouge
            cx = int((w / 2) + (w / 4) * np.sin(frame_count[0] * 0.1))
            cy = int((h / 2) + (h / 4) * np.cos(frame_count[0] * 0.07))
            
            input_pattern = np.zeros((h, w), dtype=bool)
            # Zone active autour du centre
            r = 8
            y1, y2 = max(0, cy - r), min(h, cy + r)
            x1, x2 = max(0, cx - r), min(w, cx + r)
            input_pattern[y1:y2, x1:x2] = True
            
            # Ajouter du bruit sporadique
            if frame_count[0] % 10 == 0:
                noise = np.random.random((h, w)) > 0.98
                input_pattern |= noise
            
            stack.process(input_pattern)
    
    # Créer et lancer le visualiseur
    print("\nLancement du visualiseur 3D...")
    viz = NeuronVisualizer3D(
        stack=stack,
        config=config,
        update_callback=update if demo_mode else None
    )
    
    viz.run()
    
    return 0


if __name__ == "__main__":
    exit(main())
