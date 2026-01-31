# NeuronSpikes 🧠⚡

**Réseau de neurones à impulsions (SNN) déterministe et ultra-léger**

Un système SNN original où tout est factuel — aucun hasard, même pas pour l'ensemencement.

## 🎯 Principes Fondamentaux

1. **Déterminisme absolu** — Même entrée = même sortie, toujours
2. **Frames temporelles** — Traitement par chaînes d'événements neuronaux (jusqu'à 15360 Hz)
3. **Évolution dynamique** — Neurones qui naissent, évoluent et meurent par corrélation
4. **Accélération GPU** — Conçu pour OpenCL (AMD RX 480)

## 🔬 Concept: Rétine Artificielle

La première couche est une rétine qui convertit des images monochromes en trains d'impulsions:

```
60 fps × 8 bits = 15360 Hz d'impulsions maximum
256 impulsions par pixel/frame
Distribution temporelle uniforme via LUT bit-reversal
```

## 🚀 Démarrage rapide

```bash
cd ~/GitHub/neuronspikes
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Exécuter la démo
python examples/demo_retina.py

# Lancer les tests
pytest tests/ -v
```

## 📁 Structure

```
neuronspikes/
├── src/neuronspikes/
│   ├── __init__.py
│   ├── model.py       # Modèle SNN de base
│   ├── lut.py         # LUT bit-reversal pour distribution temporelle
│   └── retina.py      # Couche rétine (entrée visuelle)
├── examples/
│   ├── minimal_run.py
│   └── demo_retina.py # Démonstration complète
├── tests/
│   ├── test_smoke.py
│   ├── test_lut.py    # Tests LUT (21 tests)
│   └── test_retina.py # Tests rétine (17 tests)
├── docs/
│   ├── JOURNAL.md     # Journal de bord du projet
│   └── ARCHITECTURE.md # Documentation technique
└── pyproject.toml
```

## 📊 Architecture Matérielle Cible

| Composant | Spécifications |
|-----------|----------------|
| CPU | AMD FX-8350 (8 cœurs @ 4 GHz) |
| RAM | 32 Go DDR3 |
| GPU Principal | AMD Radeon RX 480 (8 Go, 36 CU) |
| GPU Secondaire | NVIDIA GTX 750 Ti (2 Go, 5 SM) |

## 📝 Documentation

- [Journal de bord](docs/JOURNAL.md) — Historique des sessions de travail
- [Architecture](docs/ARCHITECTURE.md) — Documentation technique détaillée

## 🛠️ Développement

Les hooks git sont configurés pour:
- **prepare-commit-msg**: Ajoute automatiquement des métadonnées aux commits
- **post-commit**: Génère des logs détaillés dans `docs/commits/`

---

*Projet actif — Reboot d'un concept original datant de ~10 ans*


## Prochaines étapes
- Définir la dynamique neuronale exacte
- Ajouter un simulateur d'événements
- Ajouter des métriques et visualisations
