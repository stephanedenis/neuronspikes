# Journal de Bord - NeuronSpikes

## Vision du Projet

Système de neurones à impulsions (Spiking Neural Network) ultra-léger, déterministe et accéléré par GPU.
Concept originel revisité après ~10 ans.

### Principes Fondamentaux

1. **Aucun hasard** - Tout est factuel, même l'ensemencement
2. **Frames temporelles** - Traitement par chaînes d'événements neuronaux
3. **Évolution dynamique** - Neurones qui apparaissent, évoluent, s'atrophient et meurent
4. **Corrélations candidates** - Les frames d'activation forment des corrélations pour la genèse de nouveaux neurones
5. **Rétroaction** - Couches de rétroaction pour raffiner le discernement

---

## 2026-01-31 - Session Initiale

### Architecture Matérielle Disponible

#### CPU
- **Modèle**: AMD FX-8350 Eight-Core Processor
- **Cœurs**: 8 (1 thread/cœur)
- **Fréquence**: 1400-4000 MHz (boost activé)
- **Architecture**: x86_64, Famille 15h (Piledriver)

#### RAM
- **Total**: 32 Go DDR3
- **Disponible**: ~26 Go

#### GPUs OpenCL

| GPU | VRAM | Compute Units | Max Work Group | Plateforme |
|-----|------|---------------|----------------|------------|
| AMD Radeon RX 480 | 8 Go | 36 CU | 1024 | Rusticl (Mesa) |
| NVIDIA GTX 750 Ti | 2 Go | 5 SM | 1024 | NVIDIA CUDA |

**Note**: Le RX 480 sera le GPU principal pour le calcul SNN (8 Go VRAM, 36 CU).
Le GTX 750 Ti peut servir de GPU secondaire mais sa VRAM limitée (2 Go) et son architecture
Maxwell ancienne (sm_50) ne sont pas supportés par PyTorch CUDA moderne.

### Concept: Rétine Artificielle (Première Couche)

#### Spécifications
- **Source**: Feed caméra temps réel
- **Résolution**: À définir (commencer petit: 64x64 ou 128x128)
- **Profondeur**: Monochrome 8 bits par pixel
- **Fréquence source**: 60 fps

#### Conversion Intensité → Train d'Impulsions

**Principe**: Étaler l'intensité lumineuse dans le temps de façon homogène.

```
60 fps × 8 bits = 15360 Hz d'impulsions neuronales maximum
256 impulsions par pixel/frame (2^8 niveaux)
```

**LUT de Distribution Temporelle**:
Pour une distribution homogène des impulsions dans le temps, on utilise une LUT 8 bits
contenant les valeurs en ordre inversé des bits (bit-reversal permutation).

| Intensité (décimal) | Binaire | Bits inversés | Ordre temporel |
|---------------------|---------|---------------|----------------|
| 0 | 00000000 | 00000000 | 0 |
| 1 | 00000001 | 10000000 | 128 |
| 2 | 00000010 | 01000000 | 64 |
| 3 | 00000011 | 11000000 | 192 |
| ... | ... | ... | ... |

Cette distribution garantit que les impulsions sont réparties uniformément
dans la fenêtre temporelle plutôt que concentrées au début ou à la fin.

#### Comportement des Neurones de Rétine

1. **Accumulation de charge**: La charge interne augmente avec l'intensité des pixels
2. **Seuil bas**: Activation rapide pour capturer les variations fines
3. **Groupes d'activation**: Jusqu'à 15360 groupes/seconde
4. **Corrélations candidates**: Les patterns d'activation simultanée forment
   des candidats pour la création de nouveaux neurones dans les couches supérieures

### Architecture Multi-Couches (Vision)

```
┌─────────────────────────────────────────────────────────────┐
│                    COUCHES SUPÉRIEURES                       │
│         (Neurones émergents par corrélation)                 │
├─────────────────────────────────────────────────────────────┤
│                  ↑↓ RÉTROACTION ↑↓                          │
├─────────────────────────────────────────────────────────────┤
│                    COUCHE RÉTINE                             │
│    Neurones monochrome 8-bit, seuil bas, 15360 Hz max       │
├─────────────────────────────────────────────────────────────┤
│                    ENTRÉE CAMÉRA                             │
│                   60 fps, monochrome                         │
└─────────────────────────────────────────────────────────────┘
```

### Prochaines Étapes

1. [x] Implémenter la LUT d'inversion de bits
2. [x] Créer la structure de données pour les neurones de rétine
3. [x] Implémenter le mécanisme d'accumulation de charge
4. [x] Créer le système de frames d'activation
5. [ ] Intégrer OpenCL pour l'accélération GPU (RX 480)
6. [x] Connecter une source caméra

---

## 2026-01-31 - Visualiseur Rétine Temps Réel

### Implémentation `live_retina.py`

Création d'un visualiseur temps réel qui affiche côte à côte:
- **Gauche**: Image source de la caméra (monochrome)
- **Droite**: Sortie de la rétine (colormap INFERNO)

L'intensité affichée à droite correspond au cumul d'activations par pixel
pour la frame courante (équivalent direct à l'intensité du pixel source,
car le pattern d'activation encode exactement cette intensité).

#### Fonctionnalités
- Capture caméra en temps réel (640x480 → rétine 128x128)
- Contrôles interactifs:
  - `q`/`ESC`: Quitter
  - `s`: Sauvegarder capture
  - `r`: Reset statistiques
  - `+`/`-`: Ajuster résolution (32-512)
- Affichage stats: FPS, frames, résolution, impulsions totales

#### Observation

La rétine reproduit fidèlement l'image source. C'est attendu puisque
`get_activation_pattern()` retourne le nombre d'impulsions par pixel,
qui correspond exactement à l'intensité.

**La vraie valeur viendra avec les couches suivantes** qui détecteront
les corrélations entre les patterns d'activation dans le temps.

---

## Notes de Discussion

> *"Il n'y a aucun hasard dans le système, même pas pour l'ensemencement. Tout est factuel."*

> *"Ces frames d'activations sont l'essence du système. Ils constituent des corrélations
> candidates à la formation de nouveaux neurones."*

> *"Le système évolue ainsi faisant apparaître, évoluer et atrophier/mourir des neurones
> dynamiquement sur plusieurs couches."*

---

*Journal maintenu automatiquement - Chaque commit contient une entrée détaillée*

