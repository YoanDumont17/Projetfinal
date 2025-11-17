#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test simple pour Windows
Compatible avec les problemes d'encodage
"""

import os
import sys

# Force l'encodage UTF-8 pour Windows
if sys.platform == 'win32':
    import locale
    try:
        if sys.stdout.encoding != 'UTF-8':
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
    except:
        # Pour les anciennes versions de Python
        pass

try:
    import cv2
    import numpy as np
except ImportError:
    print("[ERREUR] OpenCV n'est pas installe!")
    print("Executez: pip install opencv-python numpy")
    sys.exit(1)

from fiber_circle_detection_windows import FiberCircleDetector, process_image


def test_simple():
    """Test simple sur une image."""
    
    print("=" * 50)
    print("  TEST SIMPLE DE DETECTION DE FIBRE")
    print("=" * 50)
    print()
    
    # Demander le fichier a l'utilisateur
    if len(sys.argv) > 1:
        image_file = sys.argv[1]
    else:
        print("Entrez le chemin de l'image a analyser:")
        print("(Exemple: C:\\Users\\Yoan\\Desktop\\22.png)")
        image_file = input("> ").strip('"')  # Enlever les guillemets si drag & drop
    
    # Verifier que le fichier existe
    if not os.path.exists(image_file):
        print(f"[ERREUR] Fichier non trouve: {image_file}")
        return False
    
    print()
    print(f"[INFO] Analyse de: {image_file}")
    print("-" * 50)
    
    # Traiter l'image
    success = process_image(image_file, show=False)
    
    print()
    if success:
        print("[SUCCES] Cercle detecte avec succes!")
    else:
        print("[INFO] Aucun cercle detecte.")
        print("Conseils:")
        print("  - Verifiez que l'image contient bien un cercle de fibre")
        print("  - Essayez d'ajuster le rayon attendu avec -r")
        print("  - L'image est peut-etre trop floue ou trop bruitee")
    
    return success


if __name__ == "__main__":
    print()
    print("DETECTEUR DE FIBRE OPTIQUE - Version Windows")
    print()
    
    try:
        test_simple()
    except Exception as e:
        print(f"[ERREUR] {e}")
    
    print()
    print("Appuyez sur Entree pour fermer...")
    input()
