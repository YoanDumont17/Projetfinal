#!/usr/bin/env python3
"""
Script de test pour l'algorithme de détection de fibre optique
Teste sur les 4 images fournies avec différents niveaux de netteté
"""

import os
import sys
import cv2
import numpy as np
from fiber_circle_detection import FiberCircleDetector, process_image


def test_batch_images():
    """Test sur toutes les images fournies."""
    
    # Liste des images à tester
    test_images = [
        "Main_code\Cercle\files\22.png",  # Net
        "Main_code\Cercle\files\24.png",  # Flou
        "Main_code\Cercle\files\58.png",  # Net
        "Main_code\Cercle\files\61.png"   # Très flou
    ]
    
    print("=" * 60)
    print("TEST DE DÉTECTION DE CERCLES DE FIBRE OPTIQUE")
    print("=" * 60)
    
    results = []
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n📸 Traitement de: {os.path.basename(img_path)}")
            print("-" * 40)
            
            # Nom de sortie dans le dossier outputs
            output_name = os.path.basename(img_path).replace('.png', '_detected.png')
            output_path = f"/mnt/user-data/outputs/{output_name}"
            
            # Traitement sans affichage pour le batch
            success = process_image(
                img_path, 
                output_path,
                expected_radius=None,  # Auto-détection
                show=False
            )
            
            results.append((os.path.basename(img_path), success))
        else:
            print(f"⚠️  Image non trouvée: {img_path}")
    
    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES RÉSULTATS")
    print("=" * 60)
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    for name, success in results:
        status = "✓ Détecté" if success else "✗ Non détecté"
        print(f"  {name:15} : {status}")
    
    print(f"\nTaux de réussite: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    print("\n💾 Toutes les images annotées ont été sauvegardées dans /mnt/user-data/outputs/")


def analyze_image_properties():
    """Analyse les propriétés des images pour ajuster les paramètres."""
    
    test_images = [
        "/mnt/user-data/uploads/24.png",
        "/mnt/user-data/uploads/58.png",
        "/mnt/user-data/uploads/61.png",
        "/mnt/user-data/uploads/22.png"
    ]
    
    print("\n" + "=" * 60)
    print("ANALYSE DES PROPRIÉTÉS DES IMAGES")
    print("=" * 60)
    
    for img_path in test_images:
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            print(f"\n{os.path.basename(img_path)}:")
            print(f"  Dimensions: {img.shape[1]}x{img.shape[0]} pixels")
            print(f"  Intensité moyenne: {np.mean(img):.1f}")
            print(f"  Écart-type: {np.std(img):.1f}")
            
            # Estimation de la netteté (variance du Laplacien)
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            sharpness = laplacian.var()
            print(f"  Netteté (variance Laplacien): {sharpness:.1f}")
            
            # Classification de netteté
            if sharpness > 1000:
                quality = "Très net"
            elif sharpness > 500:
                quality = "Net"
            elif sharpness > 200:
                quality = "Moyennement net"
            else:
                quality = "Flou"
            print(f"  Qualité estimée: {quality}")


def test_partial_circle():
    """
    Test spécifique pour créer et détecter un cercle partiellement visible.
    """
    print("\n" + "=" * 60)
    print("TEST DE DÉTECTION DE CERCLE PARTIEL")
    print("=" * 60)
    
    # Création d'une image de test avec cercle partiel
    test_img = np.zeros((400, 400), dtype=np.uint8)
    test_img[:] = 128  # Fond gris
    
    # Dessiner un cercle partiellement hors cadre
    cv2.circle(test_img, (350, 200), 80, 255, -1)  # Cercle blanc partiellement visible
    
    # Ajouter du bruit
    noise = np.random.normal(0, 10, test_img.shape)
    test_img = np.clip(test_img + noise, 0, 255).astype(np.uint8)
    
    # Sauvegarder l'image de test
    test_path = "/mnt/user-data/outputs/test_partial_circle.png"
    cv2.imwrite(test_path, test_img)
    
    # Tester la détection
    detector = FiberCircleDetector(expected_radius=80, radius_tolerance=0.15)
    circle, img = detector.detect(test_path)
    
    if circle is not None:
        print("✓ Cercle partiel détecté avec succès!")
        print(f"  Position: ({int(circle[0])}, {int(circle[1])})")
        print(f"  Rayon détecté: {int(circle[2])} pixels")
        print(f"  Rayon attendu: 80 pixels")
        print(f"  Erreur: {abs(circle[2] - 80):.1f} pixels")
    else:
        print("✗ Échec de la détection du cercle partiel")
    
    # Sauvegarder le résultat
    result = detector.draw_result(img, circle)
    cv2.imwrite("/mnt/user-data/outputs/test_partial_circle_detected.png", result)


if __name__ == "__main__":
    print("\n🔬 TESTS DE L'ALGORITHME DE DÉTECTION DE FIBRE OPTIQUE\n")
    
    # Créer le dossier de sortie
    os.makedirs("/mnt/user-data/outputs", exist_ok=True)
    
    # 1. Analyser les propriétés des images
    analyze_image_properties()
    
    # 2. Tester sur toutes les images
    test_batch_images()
    
    # 3. Test spécifique pour cercle partiel
    test_partial_circle()
    
    print("\n✅ Tests terminés!")
    print("📁 Vérifiez les résultats dans /mnt/user-data/outputs/")
