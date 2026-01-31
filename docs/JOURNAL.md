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

## 2026-01-31 - Module Fovéa Bio-inspirée

### Concept: Fovéa Polaire

Implémentation d'une rétine fovéale inspirée de l'anatomie de l'œil humain:
- **Zone fovéale** (centre): Haute résolution, cellules petites et denses
- **Zone périphérique**: Basse résolution, cellules grandes et éparses
- **Grille polaire**: Organisation en anneaux concentriques et secteurs angulaires

#### Architecture `fovea.py`

| Classe | Description |
|--------|-------------|
| `FoveaConfig` | Configuration (num_rings, num_sectors, fovea_radius, max_radius) |
| `GazePoint` | Point de regard avec contraintes de mouvement |
| `PolarCell` | Cellule polaire (ring, sector, inner/outer radius, angles) |
| `Fovea` | Rétine fovéale avec échantillonnage polaire |
| `StereoFovea` | Paire de fovéas pour vision stéréoscopique |
| `visualize_fovea()` | Visualisation de la grille polaire |

#### Formule de Distribution des Rayons

Les cellules suivent une distribution log-polaire:
- **Zone fovéale** (25% internes): Distribution linéaire dense
- **Zone périphérique** (75% externes): Distribution exponentielle (t^1.5)

```python
if ring_idx < num_rings // 4:
    r = fovea_radius * (ring_idx + 1) / (num_rings // 4)
else:
    t = (ring_idx - num_rings // 4) / (num_rings * 0.75)
    r = fovea_radius + (max_radius - fovea_radius) * (t ** 1.5)
```

#### Tests
- 40 tests unitaires dans `tests/test_fovea.py`
- Couverture: construction, échantillonnage, stéréo, visualisation

---

## 2026-01-31 - Caméra Stéréoscopique

### Détection Hardware

Caméra stéréo **Gearway SPCA2100** détectée sur `/dev/video1`:

| Paramètre | Valeur |
|-----------|--------|
| Bus USB | 001 |
| Mode | Side-by-side (une image double largeur) |
| Résolution max | 2560×720 (1280×720 par œil) |
| Résolution standard | 1280×480 (640×480 par œil) |
| Format | MJPG, YUYV |

**Note**: Microsoft LifeCam VX-3000 sur `/dev/video0` (mono, 640×480).

### Implémentation `live_stereo.py`

Premier visualiseur stéréo simple:
- Séparation automatique gauche/droite
- Affichage des deux fovéas
- Calcul de corrélation entre les activations

---

## 2026-01-31 - Agent Stéréo Autonome

### Concept: "Yeux mobiles qui cherchent les détails communs"

Création de `live_stereo_agent.py` - un agent d'attention visuelle binoculaire
qui déplace activement son regard vers les zones de forte saillance.

#### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AttentionAgent                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Left Fovea  │  │ Right Fovea │  │  GazeController     │ │
│  │ (polaire)   │  │ (polaire)   │  │ - Saccades          │ │
│  │             │  │             │  │ - Poursuite         │ │
│  │ vergence +5 │  │ vergence -5 │  │ - Fixation          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Saliency Map (gradient + mouvement) → SaliencyPeaks        │
│  Stereo Correlation → Score de confiance (0-1)              │
├─────────────────────────────────────────────────────────────┤
│  Decision: Si corrélation < 0.3 → Saccade vers pic saillant │
│            Si fixation > 15 frames → Explorer                │
└─────────────────────────────────────────────────────────────┘
```

#### États du Regard (GazeController)

| État | Vitesse | Déclencheur |
|------|---------|-------------|
| **Saccade** | Rapide (0.3) | Nouvelle cible, faible corrélation |
| **Poursuite** | Moyenne (0.15) | Suivi d'objet en mouvement |
| **Fixation** | Stable | Distance < 2px pendant 5+ frames |

#### Configuration Fovéa (Optimisée pour Performance)

```python
FoveaConfig(
    num_rings=8,       # Anneaux (réduit pour rapidité)
    num_sectors=8,     # Secteurs (réduit pour rapidité)
    fovea_radius=16,   # Zone centrale: 16px
    max_radius=64,     # Rayon total: 64px (diamètre 128px)
)
```

#### Contrôles Interactifs

| Touche | Action |
|--------|--------|
| `a` | Toggle autonome/manuel |
| `s` | Saccade aléatoire |
| `r` | Reset au centre |
| `d` | Toggle affichage disparité |
| `q`/`ESC` | Quitter |
| **Clic souris** | Saccade vers position |

---

## 2026-01-31 - Backend OpenCL (GPU)

### Motivation

> *"On commence à sentir le besoin d'OpenCL"*

Le traitement temps réel (2×1280×720 à 30+ FPS) nécessite l'accélération GPU.

### Implémentation `opencl_backend.py`

Backend OpenCL avec détection automatique du meilleur GPU:
- **Préférence AMD** (rusticl/Mesa plus stable que NVIDIA sous Linux)
- **Fallback NVIDIA** si AMD non disponible

#### Kernels OpenCL Implémentés

| Kernel | Description | Performance |
|--------|-------------|-------------|
| `polar_sample` | Échantillonnage polaire accéléré | ~2.4 ms |
| `compute_saliency` | Gradient Sobel (contraste) | ~4.9 ms |
| `abs_diff` | Différence absolue (mouvement) | ~0.8 ms |
| `stereo_correlation` | Corrélation + disparité stéréo | ~0.76 ms |
| `detect_rotation` | Détection rotation (VOR) | ~0.5 ms |

#### Benchmark (AMD RX 480, 160×120)

```
polar_sample:      2.38 ± 0.02 ms
compute_saliency:  4.93 ± 0.09 ms
abs_diff:          0.78 ± 0.01 ms
stereo_correlation: 0.76 ± 0.01 ms
detect_rotation:   0.49 ± 0.01 ms
```

#### Tests
- 16 tests unitaires dans `tests/test_opencl.py`
- Validation numérique CPU vs GPU

### Intégration dans l'Agent Stéréo

Modifications de `live_stereo_agent.py`:

1. **Initialisation OpenCL** dans `AttentionAgent.__init__`:
   ```python
   if use_opencl and is_opencl_available():
       self.opencl = get_opencl_backend(prefer_amd=True, verbose=True)
       self._build_cell_params()  # Pré-calcul pour GPU
   ```

2. **compute_saliency_map()** → GPU:
   ```python
   saliency = self.opencl.compute_saliency(gray_small)
   if prev_gray is not None:
       motion = self.opencl.abs_diff(gray_small, prev_small)
       saliency = saliency * 0.6 + motion * 0.4
   ```

3. **compute_stereo_correlation()** → GPU:
   ```python
   corr_arr, _ = self.opencl.stereo_correlation(left_act, right_act)
   return float(np.mean(np.clip(corr_arr, 0, 1)))
   ```

4. **Échantillonnage polaire** → GPU:
   ```python
   left_act = self.opencl.polar_sample(left_gray, gaze_x+5, gaze_y, ...)
   right_act = self.opencl.polar_sample(right_gray, gaze_x-5, gaze_y, ...)
   ```

5. **Affichage enrichi**:
   - Indicateur `[GPU]` ou `[CPU]` à côté du FPS
   - Temps de calcul saliency en millisecondes

### Gain de Performance Estimé

| Opération | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Saliency | ~15 ms | ~5 ms | 3× |
| Polar sample | ~8 ms | ~2.5 ms | 3× |
| Correlation | ~2 ms | ~0.8 ms | 2.5× |
| **Total frame** | ~50 ms | ~15 ms | **3.3×** |

---

## 2026-01-31 - Module ColorFovea (Vision Couleur et Mouvement)

### Concept: Fovéa avec Couleur, Alpha et Mouvement

Extension de la fovéa polaire avec des capacités de vision couleur et détection de mouvement:

- **Espace YUV**: Luma (Y) + Chroma (U/V) pour analyse couleur séparée
- **Canal Alpha**: Masque de validité pour pixels hors image (regard libre)
- **Détection de mouvement**: Différence temporelle + flux optique simplifié
- **Suivi d'objets**: ObjectTracker pour discrimination des objets en mouvement

#### Architecture `color_fovea.py`

| Classe | Description |
|--------|-------------|
| `ColorChannel` | Enum (LUMA, CHROMA_U, CHROMA_V, ALPHA, MOTION, MOTION_DIR) |
| `ColorFoveaConfig` | Config étendue (use_color, motion_history, alpha_padding) |
| `MotionVector` | Vecteur mouvement (dx, dy, magnitude, direction) |
| `TrackedObject` | Objet suivi (position, vitesse, signature couleur, confiance) |
| `ColorFovea` | Fovéa étendue avec sample_color() |
| `ObjectTracker` | Suivi d'objets basé sur mouvement et couleur |

#### Canaux de Sortie de `sample_color()`

```python
result = fovea.sample_color(image_bgr)
# Retourne:
#   'luma': Luminosité Y (num_rings × num_sectors)
#   'chroma_u': Chrominance bleu-jaune U
#   'chroma_v': Chrominance rouge-vert V
#   'alpha': Validité (1=dans image, 0=hors limites)
#   'motion_mag': Magnitude mouvement
#   'motion_dir': Direction mouvement (radians)
```

#### Alpha pour Regard Libre

Le canal alpha permet de positionner l'attention n'importe où, même partiellement
hors de l'image source. Les cellules hors limites ont alpha=0, ce qui permet:
- Exploration jusqu'aux bords extrêmes
- Pondération des calculs par alpha (ignorer les pixels invalides)
- Support du zoom virtuel aux limites

#### Détection de Mouvement

1. **Différence temporelle**: Comparaison de luma entre frames
2. **Flux optique simplifié**: Corrélation de phase entre secteurs adjacents
3. **Vecteurs de mouvement**: Direction et magnitude par cellule polaire
4. **Mouvement dominant**: Agrégation pour détection de mouvement global

#### Intégration dans l'Agent

Nouvelle option `--color` dans `live_stereo_agent.py`:

```bash
# Mode grayscale standard (rapide)
python examples/live_stereo_agent.py

# Mode couleur avec détection de mouvement
python examples/live_stereo_agent.py --color
```

Affichage enrichi en mode couleur:
- Indicateur "COLOR" en bas à droite
- Vecteur de mouvement dominant (magnitude et direction)

#### Tests
- 36 tests unitaires dans `tests/test_color_fovea.py`
- Couverture: ColorChannel, config, MotionVector, TrackedObject, ColorFovea, ObjectTracker

---

## Statistiques du Projet

### Tests Unitaires

| Module | Tests |
|--------|-------|
| lut.py | 12 |
| retina.py | 28 |
| groups.py | 28 |
| synapses.py | 24 |
| temporal.py | 22 |
| genesis.py | 30 |
| fovea.py | 40 |
| attention.py | 42 |
| opencl_backend.py | 16 |
| color_fovea.py | 36 |
| **Total** | **288** |

### Structure du Code

```
src/neuronspikes/
├── __init__.py          # Exports publics
├── lut.py               # Tables de lookup (bit-reversal)
├── retina.py            # Rétine cartésienne
├── groups.py            # Détection de groupes d'activation
├── synapses.py          # Connexions synaptiques Hebbiennes
├── temporal.py          # Corrélation temporelle
├── genesis.py           # Genèse de nouveaux neurones
├── fovea.py             # Rétine fovéale polaire
├── attention.py         # Système d'attention (zoom, IOR, mémoire)
├── color_fovea.py       # Fovéa couleur avec mouvement
└── opencl_backend.py    # Accélération GPU

examples/
├── live_retina.py       # Visualiseur rétine mono
├── live_stereo.py       # Visualiseur stéréo simple
└── live_stereo_agent.py # Agent stéréo autonome (--color)
```

### Dépôt GitHub

- **URL**: https://github.com/stephanedenis/neuronspikes
- **Branche principale**: main
- **Licence**: MIT

---

*Journal maintenu automatiquement - Chaque commit contient une entrée détaillée*

