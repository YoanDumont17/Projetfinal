#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from detection_fibre import FiberCircleDetector
from depth_focus2 import AnalyseurNettete

def crop_around_fiber(img, circle,
                      crop_padding=140,      # taille du crop (positif = plus large
                      use_mask=False,        # ← True = activer le masque circulaire intérieur
                      mask_padding=-80,      # ← seulement utilisé si use_mask=True
                      use_blur=False,        # ← True = adoucir le masque (recommandé si masque activé)
                      blur_kernel=121):      # ← seulement utilisé si use_blur=True
    """
    Version ultra-modulaire :
    - Par défaut (use_mask=False) → simple crop carré centré, rien d'autre
    - Si use_mask=True → applique un masque circulaire plus petit (exclut le bord externe)
    - Si use_blur=True → rend le masque très doux (évite artefact de bord)
    """
    if circle is None:
        print("Aucun cercle détecté → image complète utilisée")
        return img

    x, y, r = map(int, circle)

    # === 1. Définition de la zone de crop (carré large) ===
    crop_radius = r + crop_padding                    # rayon du crop
    half_side = crop_radius                       # marge de sécurité

    left   = max(0, x - half_side)
    right  = min(img.shape[1], x + half_side)
    top    = max(0, y - half_side)
    bottom = min(img.shape[0], y + half_side)

    cropped = img[top:bottom, left:right]

    # centre du cercle dans l'image croppée
    center_x = x - left
    center_y = y - top

    # === 2. Si on veut un masque intérieur (padding négatif) ===
    if use_mask:
        mask_radius = max(1, r + mask_padding)  # rayon du masque (plus petit → exclut le bord)

        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), mask_radius, 255, -1)

        if use_blur:
            mask = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)

        masked_img = cropped.astype(np.float32) * (mask.astype(np.float32) / 255.0)
        result = np.uint8(np.clip(masked_img, 0, 255))

        print(f"→ Masque ACTIVÉ (rayon masque = {mask_radius}px, blur = {blur_kernel if use_blur else 'off'})")
    else:
        result = cropped
        print("→ Masque DÉSACTIVÉ → simple crop carré")

    print(f"Crop size: {result.shape[1]}x{result.shape[0]}, rayon fibre ≈ {r}px")
    return result


def process_stack(input_dir,
                  cropped_dir="cropped_images",
                  crop_padding=140,
                  use_mask=False,
                  mask_padding=-80,
                  use_blur=False,
                  blur_kernel=121):
    os.makedirs(cropped_dir, exist_ok=True)
    detector = FiberCircleDetector()

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])

    for filename in image_files:
        path = os.path.join(input_dir, filename)
        print(f"\nTraitement → {filename}")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        circle, _ = detector.detect(path)

        cropped_img = crop_around_fiber(img, circle,
                                        crop_padding=crop_padding,
                                        use_mask=use_mask,
                                        mask_padding=mask_padding,
                                        use_blur=use_blur,
                                        blur_kernel=blur_kernel)

        cv2.imwrite(os.path.join(cropped_dir, filename), cropped_img)

    return cropped_dir


if __name__ == "__main__":
    input_dir = r"Cercle\image_fibre_2"

    # ===================================================================
    # === RÉGLAGES ULTRA-SIMPLES (tu changes juste ici) ===
    # ===================================================================
    use_mask     = False      # ← mets True pour activer le masque intérieur (exclut le bord)
    mask_padding  = -80       # ← valeur négative = exclut le bord de la fibre (test -60, -80 -100)
    use_blur     = False      # ← mets True quand tu actives le masque (évite artefact)
    blur_kernel  = 121        # ← 81, 101, 121, 151 → plus gros = plus doux
    crop_padding  = 0     # ← taille du carré (positif = plus large, garde le bord réel)
    # ===================================================================

    cropped_dir = process_stack(input_dir,
                                crop_padding=crop_padding,
                                use_mask=use_mask,
                                mask_padding=mask_padding,
                                use_blur=use_blur,
                                blur_kernel=blur_kernel)

    # Analyse focus
    analyseur = AnalyseurNettete()
    resultats = analyseur.analyser_dossier(cropped_dir)
    analyseur.afficher_resultats_simple(resultats)