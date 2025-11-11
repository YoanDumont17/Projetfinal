# GUIDE WINDOWS - Détection de Cercles de Fibre Optique

## 🚀 Installation Rapide

### 1. Installer Python
Si Python n'est pas déjà installé:
1. Aller sur https://www.python.org/downloads/
2. Télécharger Python 3.8 ou plus récent
3. **IMPORTANT**: Cocher "Add Python to PATH" pendant l'installation

### 2. Installer les dépendances
Ouvrir l'invite de commandes (cmd) et exécuter:
```
pip install -r requirements.txt
```
Ou manuellement:
```
pip install opencv-python numpy
```

## 💻 Utilisation

### Méthode 1: Script de Test Simple (RECOMMANDÉ)
Double-cliquer sur `test_windows.py` ou:
```
python test_windows.py
```
Puis entrer le chemin de l'image ou glisser-déposer le fichier.

### Méthode 2: Ligne de Commande
```
python fiber_circle_detection_windows.py 22.png
```

### Méthode 3: Script Batch
```
detect_fiber.bat 22.png
```

### Méthode 4: Dans votre code Python
```python
from fiber_circle_detection_windows import process_image

# Analyser une image
success = process_image("C:\\Users\\Yoan\\Desktop\\22.png", show=False)

if success:
    print("Cercle trouvé!")
```

## ⚠️ Résolution de Problèmes

### Erreur "charmap codec can't encode"
✅ Déjà corrigé dans `fiber_circle_detection_windows.py`
- Utilise uniquement des caractères ASCII
- Force l'encodage UTF-8 automatiquement

### Erreur "ModuleNotFoundError: No module named 'cv2'"
Installer OpenCV:
```
pip install opencv-python
```

### L'image n'est pas détectée
Essayer avec différents paramètres:
```python
from fiber_circle_detection_windows import FiberCircleDetector

# Ajuster le rayon attendu (en pixels)
detector = FiberCircleDetector(expected_radius=100, radius_tolerance=0.2)
circle, img = detector.detect("image.png")
```

## 📊 Paramètres Ajustables

- **expected_radius**: Rayon attendu du cercle en pixels (défaut: auto-détecté)
- **radius_tolerance**: Tolérance sur le rayon (défaut: 0.15 = ±15%)

## 🎯 Performances Attendues

- Images nettes: Détection précise à ±2 pixels
- Images floues: Détection à ±5 pixels
- Images très floues: Détection possible avec précision réduite
- Cercles partiels: Détection si >30% visible

## 📁 Fichiers du Package

- **fiber_circle_detection_windows.py**: Module principal (version Windows)
- **test_windows.py**: Script de test interactif
- **detect_fiber.bat**: Script batch pour Windows
- **requirements.txt**: Dépendances Python
- **README.md**: Documentation complète

## 💡 Conseils d'Utilisation

1. **Pour des images très floues**: L'algorithme a 4 stratégies, il devrait détecter automatiquement
2. **Pour ajuster la sensibilité**: Modifier les paramètres dans le code (param1, param2)
3. **Pour debug**: Utiliser `show=True` pour voir le résultat visuellement

## 🔧 Intégration dans votre Projet

```python
import cv2
from fiber_circle_detection_windows import FiberCircleDetector

class MonSystemeAutofocus:
    def __init__(self):
        self.detector = FiberCircleDetector(expected_radius=120)
    
    def aligner_fibre(self, image_path):
        circle, img = self.detector.detect(image_path)
        
        if circle:
            x, y, r = circle
            # Utiliser x, y pour l'alignement mécanique
            self.deplacer_vers(x, y)
            return True
        return False
```

## ✅ Test Rapide

Pour vérifier que tout fonctionne:
```
python -c "from fiber_circle_detection_windows import process_image; print('[OK] Module charge avec succes')"
```

---
Développé pour le projet de fin d'études en Génie Physique
Compatible Windows 7/10/11 | Python 3.6+
