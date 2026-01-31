#!/usr/bin/env python3
"""
Démonstration de la rétine artificielle NeuronSpikes.

Ce script montre le fonctionnement de la couche rétine:
1. Génération d'une image test (gradient)
2. Conversion en trains d'impulsions via LUT bit-reversal
3. Visualisation des activations dans le temps
"""

import numpy as np

from neuronspikes import create_retina, BIT_REVERSAL_LUT


def demo_lut():
    """Démonstration de la LUT d'inversion de bits."""
    print("=" * 60)
    print("LUT BIT-REVERSAL - Distribution temporelle homogène")
    print("=" * 60)
    print()
    
    examples = [0, 1, 2, 3, 127, 128, 254, 255]
    
    print(f"{'Intensité':>10} | {'Binaire':>10} | {'Inversé':>10} | {'Slot':>5}")
    print("-" * 45)
    
    for i in examples:
        binary = f"{i:08b}"
        reversed_val = BIT_REVERSAL_LUT[i]
        reversed_binary = f"{reversed_val:08b}"
        print(f"{i:>10} | {binary:>10} | {reversed_binary:>10} | {reversed_val:>5}")
    
    print()
    print("→ La valeur 1 (00000001) active le slot 128 (milieu de frame)")
    print("→ La valeur 128 (10000000) active le slot 1 (début de frame)")
    print("→ Cette distribution garantit une répartition temporelle uniforme")
    print()


def demo_retina():
    """Démonstration de la couche rétine."""
    print("=" * 60)
    print("COUCHE RÉTINE - Conversion intensité → impulsions")
    print("=" * 60)
    print()
    
    # Créer une petite rétine 8x8
    retina = create_retina(8, 8, 60)
    
    print(f"Configuration:")
    print(f"  • Résolution: {retina.shape[1]}×{retina.shape[0]} pixels")
    print(f"  • Fréquence max: {retina.max_frequency} Hz")
    print(f"  • Durée d'un slot: {retina.slot_duration_us:.1f} µs")
    print()
    
    # Créer une image test avec gradient
    frame = np.array([
        [0,   32,  64,  96,  128, 160, 192, 224],
        [32,  64,  96,  128, 160, 192, 224, 255],
        [64,  96,  128, 160, 192, 224, 255, 224],
        [96,  128, 160, 192, 224, 255, 224, 192],
        [128, 160, 192, 224, 255, 224, 192, 160],
        [160, 192, 224, 255, 224, 192, 160, 128],
        [192, 224, 255, 224, 192, 160, 128, 96],
        [224, 255, 224, 192, 160, 128, 96,  64],
    ], dtype=np.uint8)
    
    print("Image test (gradient diagonal):")
    for row in frame:
        print("  " + " ".join(f"{p:3d}" for p in row))
    print()
    
    # Traiter la frame
    retina.process_frame(frame)
    
    # Exécuter quelques slots et montrer les activations
    print("Activations par slot temporel (• = spike):")
    print()
    
    interesting_slots = [1, 64, 128, 192, 255]
    for slot in interesting_slots:
        activations = retina.get_activations(slot)
        spike_count = activations.sum()
        
        print(f"Slot {slot:3d}:")
        for row in activations:
            print("  " + " ".join("•" if a else "·" for a in row))
        print(f"  → {spike_count} activations")
        print()
    
    # Exécuter toute la frame
    retina.reset()
    retina.process_frame(frame)
    retina.run_frame()
    
    print("Statistiques après 1 frame complète:")
    print(f"  • Impulsions totales: {retina.stats['total_spikes']}")
    print(f"  • Groupes d'activation: {retina.stats['activation_groups']}")
    print(f"  • Intensité moyenne: {frame.mean():.1f}")
    print()


def demo_temporal_distribution():
    """Démonstration de la distribution temporelle."""
    print("=" * 60)
    print("DISTRIBUTION TEMPORELLE - Répartition uniforme des spikes")
    print("=" * 60)
    print()
    
    retina = create_retina(16, 16, 60)
    
    # Image uniforme à intensité 128
    frame = np.full((16, 16), 128, dtype=np.uint8)
    retina.process_frame(frame)
    
    # Compter les activations par quartile temporel
    quartiles = [0, 0, 0, 0]
    
    for slot in range(256):
        activations = retina.get_activations(slot)
        q = slot // 64
        quartiles[q] += activations.sum()
    
    print("Répartition des impulsions par quartile temporel:")
    print(f"  Q1 (slots 0-63):    {quartiles[0]:5d} impulsions")
    print(f"  Q2 (slots 64-127):  {quartiles[1]:5d} impulsions")
    print(f"  Q3 (slots 128-191): {quartiles[2]:5d} impulsions")
    print(f"  Q4 (slots 192-255): {quartiles[3]:5d} impulsions")
    print()
    
    total = sum(quartiles)
    print(f"Total: {total} impulsions (attendu: {16*16*128} = {16*16*128})")
    print()
    print("→ La distribution est relativement uniforme grâce au bit-reversal")
    print()


def main():
    """Point d'entrée principal."""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           NEURONSPIKES - Démonstration Rétine              ║")
    print("║                                                            ║")
    print("║  Système de neurones à impulsions déterministe             ║")
    print("║  Conversion intensité → trains d'impulsions temporels      ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    
    demo_lut()
    demo_retina()
    demo_temporal_distribution()
    
    print("=" * 60)
    print("Démonstration terminée.")
    print()
    print("Prochaines étapes:")
    print("  1. Connecter une source caméra réelle")
    print("  2. Implémenter la détection de groupes d'activation")
    print("  3. Créer les couches supérieures (genèse de neurones)")
    print("  4. Accélérer avec OpenCL sur le RX 480")
    print("=" * 60)


if __name__ == "__main__":
    main()
