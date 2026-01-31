# Architecture Conceptuelle - NeuronSpikes

## Vue d'Ensemble

NeuronSpikes est un système de réseau de neurones à impulsions (SNN - Spiking Neural Network)
qui se distingue par son approche **déterministe** et **dynamique**.

## Principes Fondamentaux

### 1. Déterminisme Absolu

> *"Il n'y a aucun hasard dans le système, même pas pour l'ensemencement. Tout est factuel."*

- Aucun générateur de nombres aléatoires
- Reproductibilité parfaite: même entrée → même sortie
- États initiaux explicites et traçables

### 2. Traitement par Frames Temporelles

Le système opère sur des **chaînes d'événements neuronaux** organisées en frames:

```
Frame 0 ──→ Frame 1 ──→ Frame 2 ──→ ... ──→ Frame N
   │           │           │                   │
   ▼           ▼           ▼                   ▼
[Activations] [Activations] [Activations]  [Activations]
```

### 3. Évolution Dynamique des Neurones

Les neurones ne sont pas statiques - ils suivent un cycle de vie:

```
        ┌──────────────────────────────────────────────┐
        │                                              ▼
    [Genèse] ──→ [Croissance] ──→ [Maturité] ──→ [Atrophie] ──→ [Mort]
        ▲                             │
        │                             │
        └─────── Corrélations ────────┘
                 candidates
```

### 4. Corrélations Candidates

Les **frames d'activation** sont l'essence du système:

- Patterns de co-activation détectés
- Candidats à la formation de nouveaux neurones
- Renforcement par répétition
- Atrophie par absence

## Architecture en Couches

```
┌─────────────────────────────────────────────────────────────────┐
│                     COUCHES COGNITIVES                          │
│            (Émergent par corrélations répétées)                 │
│                                                                 │
│    ┌─────┐ ┌─────┐ ┌─────┐                                     │
│    │ N₁  │ │ N₂  │ │ N₃  │  ← Neurones dynamiques              │
│    └──┬──┘ └──┬──┘ └──┬──┘                                     │
│       │       │       │                                         │
├───────┼───────┼───────┼─────────────────────────────────────────┤
│       │       │       │     RÉTROACTION                         │
│       ▼       ▼       ▼                                         │
│    ┌─────────────────────┐                                      │
│    │  Modulation top-down │ ← Affine le discernement            │
│    └──────────┬──────────┘                                      │
│               │                                                 │
├───────────────┼─────────────────────────────────────────────────┤
│               ▼                                                 │
│    ┌─────────────────────────────────────────────────────────┐  │
│    │                   COUCHE RÉTINE                         │  │
│    │                                                         │  │
│    │  ┌───┬───┬───┬───┬───┬───┬───┬───┐                     │  │
│    │  │ P │ P │ P │ P │ P │ P │ P │ P │  ← Pixels/Neurones  │  │
│    │  └───┴───┴───┴───┴───┴───┴───┴───┘                     │  │
│    │                                                         │  │
│    │  Caractéristiques:                                      │  │
│    │  • Monochrome 8 bits                                    │  │
│    │  • Seuil d'activation bas                               │  │
│    │  • Fréquence max: 15360 Hz                              │  │
│    │                                                         │  │
│    └─────────────────────────────────────────────────────────┘  │
│                              ▲                                  │
├──────────────────────────────┼──────────────────────────────────┤
│                              │                                  │
│    ┌─────────────────────────┴─────────────────────────────────┐│
│    │                    ENTRÉE SENSORIELLE                     ││
│    │                                                           ││
│    │  Source: Caméra @ 60 fps                                  ││
│    │  Format: Monochrome 8-bit                                 ││
│    │  Résolution: Configurable (64x64 → 1920x1080)            ││
│    │                                                           ││
│    └───────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Conversion Intensité → Train d'Impulsions

### Le Problème

Comment convertir une intensité lumineuse statique (0-255) en un train d'impulsions
temporellement distribué de façon homogène?

### La Solution: LUT Bit-Reversal

Pour éviter que les impulsions soient concentrées en début ou fin de frame,
on utilise une **permutation par inversion de bits**.

#### Principe

Pour chaque valeur d'intensité `i` (0-255):
1. Convertir en binaire 8 bits
2. Inverser l'ordre des bits
3. Utiliser comme index temporel

```python
def bit_reverse_lut():
    """Génère la LUT de permutation par inversion de bits."""
    lut = []
    for i in range(256):
        reversed_bits = int('{:08b}'.format(i)[::-1], 2)
        lut.append(reversed_bits)
    return lut
```

#### Visualisation

```
Intensité 1:   00000001 → 10000000 → Slot 128/256 (milieu)
Intensité 2:   00000010 → 01000000 → Slot 64/256 (quart)
Intensité 3:   00000011 → 11000000 → Slot 192/256 (3/4)
Intensité 128: 10000000 → 00000001 → Slot 1/256 (début)
Intensité 255: 11111111 → 11111111 → Slot 255/256 (fin)
```

Cette distribution garantit une **répartition temporelle maximalement uniforme**.

### Timing

```
1 frame vidéo = 1/60 seconde ≈ 16.67 ms
256 slots d'impulsion par frame
1 slot = 16.67 ms / 256 ≈ 65.1 µs
Fréquence maximale = 60 × 256 = 15360 Hz
```

## Mécanisme d'Activation

### Neurone de Rétine

```
┌─────────────────────────────────────┐
│         NEURONE DE RÉTINE           │
│                                     │
│  Entrée: Intensité pixel (0-255)    │
│                                     │
│  ┌───────────────────────────────┐  │
│  │     Charge interne: Q         │  │
│  │                               │  │
│  │  Q += intensité × Δt          │  │
│  │                               │  │
│  │  if Q >= seuil:               │  │
│  │      SPIKE! (impulsion)       │  │
│  │      Q = 0 (reset)            │  │
│  │                               │  │
│  └───────────────────────────────┘  │
│                                     │
│  Seuil: Très bas (sensibilité max)  │
│                                     │
└─────────────────────────────────────┘
```

### Groupes d'Activation

Les neurones qui s'activent **simultanément** forment des groupes:

```
Frame t:
┌───┬───┬───┬───┐      Groupe A: {(0,0), (1,0), (0,1)}
│ * │ * │   │   │      Groupe B: {(2,2), (3,2), (2,3), (3,3)}
├───┼───┼───┼───┤
│ * │   │   │   │      Ces patterns de co-activation sont
├───┼───┼───┼───┤      des CORRÉLATIONS CANDIDATES pour
│   │   │ * │ * │      la formation de nouveaux neurones
├───┼───┼───┼───┤      dans les couches supérieures.
│   │   │ * │ * │
└───┴───┴───┴───┘
```

## Accélération GPU

### Stratégie

Le **AMD Radeon RX 480** sera utilisé pour:

1. **Conversion intensité → impulsions** (kernel OpenCL)
2. **Accumulation de charge** (parallel reduction)
3. **Détection de groupes d'activation** (clustering)
4. **Mise à jour des corrélations** (matrix operations)

### Ressources Disponibles

| Caractéristique | Valeur |
|-----------------|--------|
| Compute Units | 36 |
| VRAM | 8 Go |
| Max Work Group | 1024 |
| Bande passante | 256 GB/s |

### Estimation de Capacité

Pour une résolution 640×480 @ 60fps:
- Pixels: 307,200
- Impulsions/frame: 307,200 × 256 = 78,643,200 max
- Impulsions/seconde: 4.7 milliards max

Le RX 480 avec 36 CU peut gérer cette charge.

## Prochaines Étapes

1. **Prototype CPU** - Valider la logique sans GPU
2. **Kernels OpenCL** - Porter les opérations critiques
3. **Capture caméra** - Intégrer une source vidéo réelle
4. **Couche suivante** - Implémenter la genèse de neurones par corrélation

---

*Document évolutif - Mis à jour au fil du développement*
