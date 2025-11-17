#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour tester différents paddings : détecter la fibre, crop avec masque circulaire, analyser le focus, et trouver le meilleur padding.
Importe les classes de detection_fibre.py et depth_focus.py.
"""
import matplotlib.pyplot as plt
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
    side_length = 2 * masked_r  # Taille du carré basée sur le rayon masqué
    
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

def test_paddings(input_dir, padding_range=np.linspace(-30, 30, 60), expected_peak_z=770):
    """
    Teste une plage de paddings : pour chaque, génère crops, analyse focus, collecte z_optimal et incertitude.
    Trouve le meilleur padding (celui minimisant |z_optimal - expected_peak_z| + incertitude).
    """
    results = []
    base_cropped_dir = "cropped_images_test"
    
    analyseur = AnalyseurNettete()
    detector = FiberCircleDetector()
    
    for padding in padding_range:
        padding_int = int(padding)  # Arrondir pour noms de dossiers
        cropped_dir = f"{base_cropped_dir}_pad{padding_int}"
        
        print(f"\n=== Test padding = {padding_int} ===")
        cropped_dir = process_stack(input_dir, cropped_dir, padding=padding_int)
        
        resultats = analyseur.analyser_dossier(cropped_dir)
        if not resultats['noms']:
            print("Aucune image analysée, skip.")
            continue
        
        z_optimal, incertitude, z_optimaux = analyseur.trouver_z_optimal_simple(resultats)
        print(f"z_optimal = {z_optimal:.1f}, incertitude = {incertitude:.1f}")
        
        # Score de "qualité" : proximité du peak attendu + faible incertitude
        error = abs(z_optimal - expected_peak_z)
        quality_score = error + incertitude
        
        results.append({
            'padding': padding_int,
            'z_optimal': z_optimal,
            'incertitude': incertitude,
            'error_to_expected': error,
            'quality_score': quality_score,
            'z_lapl8': z_optimaux.get('laplacien8', np.nan)
        })
    
    # Trouver le meilleur
    best = min(results, key=lambda x: x['quality_score'])
    print(f"\n=== MEILLEUR PADDING ===\nPadding optimal: {best['padding']} (z_optimal={best['z_optimal']:.1f}, incertitude={best['incertitude']:.1f}, error={best['error_to_expected']:.1f})")
    
    # Plot des résultats
    paddings = np.array([r['padding'] for r in results])
    z_opt = np.array([r['z_optimal'] for r in results])
    inc = np.array([r['incertitude'] for r in results])
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(paddings, z_opt, 'o-', label='z_optimal')
    plt.axhline(expected_peak_z, color='r', linestyle='--', label=f'Expected ({expected_peak_z})')
    plt.xlabel('Padding')
    plt.ylabel('z_optimal')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(paddings, inc, 's-', label='incertitude')
    plt.xlabel('Padding')
    plt.ylabel('Incertitude')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Test de Paddings : Impact sur Focus Optimal')
    plt.tight_layout()
    plt.show()
    
    return results, best

if __name__ == "__main__":
    # Dossier d'entrée avec les images originales
    input_dir = r"Cercle\image_fibre_2"  # Remplace par ton dossier réel
    
    # Plage de paddings à tester (ex. -30 à 30 en 60 steps)
    padding_range = [6]
    
    # z attendu (d'après tes plots, ~770)
    expected_peak_z = 920
    
    # Lancer les tests
    results, best_padding = test_paddings(input_dir, padding_range, expected_peak_z)