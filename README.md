# Handsketch3D 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green.svg)](https://google.github.io/mediapipe/)
[![Point-E](https://img.shields.io/badge/OpenAI-Point--E-orange.svg)](https://github.com/openai/point-e)

**Handsketch3D** est une application interactive qui transforme les dessins à main levée en modèles 3D. Développé dans le cadre du Master 2 Vision et Machine Intelligence de l'Université de Paris Cité, ce projet rend la modélisation 3D accessible sans expertise technique préalable.

##  Fonctionnalités

-  **Dessin gestuel en temps réel** avec reconnaissance des mouvements de la main
-  **Génération 3D automatique** à partir de croquis 2D
-  **Visualisation interactive** des nuages de points 3D

##  Architecture

Le projet repose sur une architecture modulaire intégrant plusieurs composants d'intelligence artificielle :

```
┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────────┐
│ Webcam  │───▶│  MediaPipe   │───▶│ Canvas 2D   │───▶│ Point-E     │───▶│ Open3D       │
│         │    │  Hands       │    │ Drawing     │    │ Pipeline    │    │ Viewer       │
└─────────┘    └──────────────┘    └─────────────┘    └─────────────┘    └──────────────┘
```

###  Composants principaux

1. **MediaPipe Hands** : Détection et tracking de 21 points de repère par main pour capturer les gestes de dessin en temps réel
2. **Canvas 2D** : Interface de dessin interactive
3. **Prétraitement** : Normalisation, binarisation et lissage morphologique pour optimiser les croquis
4. **Point-E (base40M)** : Pipeline de diffusion à deux passes (1024 → 4096 points) pour la génération de nuages de points 3D
5. **Open3D Viewer** : Visualisation 3D interactive du résultat

##  Installation

### Prérequis

- Python 3.8 ou supérieur
- Webcam fonctionnelle
- GPU recommandé pour de meilleures performances

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/votre-username/handsketch3d.git
cd handsketch3d

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
point-e @ git+https://github.com/openai/point-e.git
open3d>=0.17.0
pillow>=10.0.0
```

## Utilisation

### Lancement de l'application

```bash
python main_pipeline.py
```

### Contrôles

-  **Levez votre index** devant la webcam pour dessiner
-  **Fermez la main** pour arrêter de dessiner
-  **Appuyez sur 'R'** pour réinitialiser le canvas
-  **Appuyez sur 'G'** pour générer le modèle 3D
-  **Appuyez sur 'Q'** pour quitter l'application

### Conseils pour de meilleurs résultats

- Dessinez des formes **simples et convexes** (cube, chaise, lampe, table)
- Évitez les détails trop fins ou les formes très complexes
- Assurez un **éclairage uniforme** pour une meilleure détection de la main
- Gardez votre main bien visible face à la caméra

### Exemples de génération

```
Croquis 2D (chaise)  →  Nuage de points 3D (4096 points)
Croquis 2D (lampe)   →  Nuage de points 3D (4096 points)
```

### Limitations connues

-  **Domain gap** : Point-E est entraîné sur des rendus photoréalistes, pas sur des croquis
-  **Stochasticité** : Un même croquis peut produire des résultats variables
-  **Résolution limitée** : Maximum 4096 points, insuffisant pour les détails fins
- **Formes ambiguës** : Les dessins complexes ou abstraits produisent des résultats bruités

##  License

Ce projet a été développé dans un cadre académique à l'Université de Paris Cité.

##  Auteur

**Pugenger Riad**  
Master 2 Vision et Machine Intelligence  
Université de Paris Cité - 2025/2026

**Responsable UE** : Nizar Ouarti


##  Références

- **Point-E**: A System for Generating 3D Point Clouds from Complex Prompts ([Paper](https://arxiv.org/abs/2212.08751))
- **MediaPipe Hands**: On-device Real-time Hand Tracking
- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision
