#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fichier principal pour traiter une pile d'images : détecter la fibre, crop autour en carré avec masque circulaire configurable, et analyser le focus.
Importe les classes de detection_fibre.py et depth_focus.py.
"""

import os
import cv2
import numpy as np
from detection_fibre import FiberCircleDetector  # Import de la classe de détection de fibre
from depth_focus import AnalyseurNettete  # Import de la classe d'analyse de netteté

def crop_around_fiber(img, circle, padding=50):
    """
    Crop l'image en un carré autour du cercle détecté avec padding.
    Applique un masque circulaire centré, avec rayon ajusté par padding (positif: élargi, 0: exact, négatif: rétréci mais min r).
    Adoucit le masque avec blur pour éviter artefacts de bordure.
    Retourne l'image croppée/masquée ou l'originale si pas de cercle.
    """
    if circle is None:
        print("Aucun cercle détecté, utilisation de l'image complète.")
        return img
    
    orig_x, orig_y, r = map(int, circle)
    masked_r = max(1, r + padding)  # Rayon du masque ajusté (min 1 pour éviter zéro)
    side_length = 2 * masked_r  # Taille du carré basée sur le rayon masqué (sans padding extra pour le crop)
    
    # Bornes du crop centrées sur le cercle original
    left = max(0, orig_x - side_length // 2)
    right = min(img.shape[1], orig_x + side_length // 2)
    top = max(0, orig_y - side_length // 2)
    bottom = min(img.shape[0], orig_y + side_length // 2)
    
    cropped = img[top:bottom, left:right]
    
    # Ajuster les coordonnées du centre dans le crop
    crop_height, crop_width = cropped.shape
    center_x = orig_x - left
    center_y = orig_y - top
    
    # Créer un masque circulaire centré dans le crop, avec rayon ajusté
    mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), masked_r, 255, -1)
    
    # Adoucir le masque avec Gaussian blur pour réduire les artefacts de bordure
    mask = cv2.GaussianBlur(mask, (15, 15), 0)  # Blur fort pour transitions douces
    
    # Appliquer le masque : multiplier (pour gradients doux)
    masked_cropped = (cropped.astype(np.float32) * (mask / 255.0)).astype(np.uint8)
    
    print(f"Masque circulaire appliqué avec padding {padding} (rayon effectif: {masked_r}), adouci avec blur.")
    return masked_cropped

def process_stack(input_dir, cropped_dir="cropped_images", padding=50, show=False):
    """
    Traite la pile d'images du dossier input_dir : détecte fibre, crop en carré avec masque (padding configurable), et sauvegarde dans cropped_dir.
    """
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)
    
    detector = FiberCircleDetector()
    
    # Lister les images PNG
    image_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith('.png')]
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        print(f"Traitement de {input_path}...")
        
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Erreur de lecture: {input_path}")
            continue
        
        circle, _ = detector.detect(input_path)  # Obtient le cercle
        
        cropped_img = crop_around_fiber(img, circle, padding=padding)
        
        cropped_path = os.path.join(cropped_dir, filename)
        cv2.imwrite(cropped_path, cropped_img)
        print(f"Image croppée/masquée sauvegardée: {cropped_path}")
    
    return cropped_dir

if __name__ == "__main__":
    # Dossier d'entrée avec les images originales
    input_dir = r"Cercle\image_fibre_2"  # Remplace par ton dossier réel
    
    # Choisis manuellement le padding ici (ex: 0 pour cercle exact, 5 pour +5px, -10 pour rétréci de 10px)
    manual_padding = 6  # Modifie cette valeur pour tester
    
    # Traiter la pile et générer les crops masqués
    cropped_dir = process_stack(input_dir, padding=manual_padding)
    
    # Analyser les crops pour trouver la netteté maximale
    analyseur = AnalyseurNettete()
    resultats = analyseur.analyser_dossier(cropped_dir)
    analyseur.afficher_resultats_simple(resultats)