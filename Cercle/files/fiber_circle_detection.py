#!/usr/bin/env python3
"""
Détection robuste de cercle de fibre optique dans des images PNG
Projet de fin d'études - Génie Physique
Auteur: Assistant pour Yoan
"""

import cv2
import numpy as np
import os
from typing import Tuple, Optional, List
import argparse


class FiberCircleDetector:
    """
    Classe pour détecter des cercles de fibre optique dans des images
    avec robustesse aux variations de netteté et position.
    """
    
    def __init__(self, expected_radius: int = 50, radius_tolerance: float = 0.15):
        """
        Initialise le détecteur.
        
        Args:
            expected_radius: Rayon attendu du cercle en pixels
            radius_tolerance: Tolérance sur le rayon (0.15 = ±15%)
        """
        self.expected_radius = expected_radius
        self.radius_tolerance = radius_tolerance
        self.min_radius = int(expected_radius * (1 - radius_tolerance))
        self.max_radius = int(expected_radius * (1 + radius_tolerance))
        
    def preprocess_image(self, img: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        """
        Prétraite l'image pour améliorer la détection.
        
        Args:
            img: Image en niveaux de gris
            enhance_contrast: Applique CLAHE pour améliorer le contraste
            
        Returns:
            Image prétraitée
        """
        # Débruitage avec filtre bilatéral (préserve les bords)
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        if enhance_contrast:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Améliore le contraste local, utile pour les images peu contrastées
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
        else:
            enhanced = denoised
            
        # Filtre médian pour enlever le bruit salt-and-pepper
        cleaned = cv2.medianBlur(enhanced, 5)
        
        return cleaned
    
    def detect_edges_adaptive(self, img: np.ndarray) -> np.ndarray:
        """
        Détection de contours adaptative basée sur les statistiques de l'image.
        
        Args:
            img: Image prétraitée
            
        Returns:
            Image des contours
        """
        # Calcul des seuils adaptatifs basés sur la médiane de l'image
        # Méthode inspirée de l'auto-Canny
        v = np.median(img)
        sigma = 0.33
        
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        # Application de Canny avec seuils adaptatifs
        edges = cv2.Canny(img, lower, upper)
        
        # Morphologie pour connecter les contours brisés
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def find_best_circle(self, circles: np.ndarray, img_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Sélectionne le meilleur cercle parmi les candidats.
        
        Args:
            circles: Array de cercles détectés (x, y, rayon)
            img_shape: Dimensions de l'image (hauteur, largeur)
            
        Returns:
            Meilleur cercle ou None
        """
        if circles is None or len(circles) == 0:
            return None
            
        height, width = img_shape
        best_circle = None
        best_score = -1
        
        for circle in circles:
            x, y, r = circle
            
            # Score basé sur plusieurs critères
            score = 0
            
            # 1. Proximité avec le rayon attendu (plus important)
            radius_diff = abs(r - self.expected_radius) / self.expected_radius
            radius_score = max(0, 1 - radius_diff) * 2.0  # Poids double
            score += radius_score
            
            # 2. Complétude du cercle (pénalité si trop près des bords)
            margin = r * 0.3  # Marge de 30% du rayon
            if x - r >= -margin and x + r <= width + margin and \
               y - r >= -margin and y + r <= height + margin:
                completeness_score = 1.0
            else:
                # Calcul du pourcentage visible approximatif
                visible_portion = self.estimate_visible_portion(x, y, r, width, height)
                completeness_score = visible_portion
            score += completeness_score
            
            # 3. Distance au centre (légère préférence pour les cercles centrés)
            center_x, center_y = width / 2, height / 2
            dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            center_score = 1 - (dist_to_center / max_dist) * 0.3  # Poids faible
            score += center_score
            
            if score > best_score:
                best_score = score
                best_circle = circle
                
        return best_circle
    
    def estimate_visible_portion(self, x: float, y: float, r: float, 
                                 width: int, height: int) -> float:
        """
        Estime la portion visible d'un cercle partiellement hors cadre.
        
        Returns:
            Proportion visible approximative (0 à 1)
        """
        # Calcul simplifié basé sur la distance du centre aux bords
        left_overflow = max(0, r - x) / r
        right_overflow = max(0, (x + r) - width) / r
        top_overflow = max(0, r - y) / r
        bottom_overflow = max(0, (y + r) - height) / r
        
        # Estimation approximative
        visible = 1.0 - (left_overflow + right_overflow + top_overflow + bottom_overflow) / 4
        return max(0.3, visible)  # Au minimum 30% visible
    
    def detect_multi_strategy(self, img: np.ndarray) -> Optional[np.ndarray]:
        """
        Détection avec plusieurs stratégies pour maximiser la robustesse.
        
        Args:
            img: Image en niveaux de gris
            
        Returns:
            Meilleur cercle détecté ou None
        """
        all_candidates = []
        
        # Stratégie 1: Image originale avec prétraitement standard
        preprocessed = self.preprocess_image(img, enhance_contrast=True)
        circles1 = cv2.HoughCircles(
            preprocessed,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.expected_radius,
            param1=50,
            param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        if circles1 is not None:
            all_candidates.extend(circles1[0])
        
        # Stratégie 2: Avec détection de contours Canny
        edges = self.detect_edges_adaptive(preprocessed)
        circles2 = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.expected_radius,
            param1=50,
            param2=20,  # Plus sensible
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        if circles2 is not None:
            all_candidates.extend(circles2[0])
        
        # Stratégie 3: Image floue (pour les cas très bruités)
        blurred = cv2.GaussianBlur(img, (5, 5), 1)
        circles3 = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.expected_radius,
            param1=60,
            param2=35,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        if circles3 is not None:
            all_candidates.extend(circles3[0])
        
        # Stratégie 4: Pour images TRÈS floues (paramètres très permissifs)
        heavily_blurred = cv2.GaussianBlur(img, (9, 9), 2)
        circles4 = cv2.HoughCircles(
            heavily_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=self.expected_radius,
            param1=40,  # Seuil Canny très bas
            param2=15,  # Seuil d'accumulation très bas
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        if circles4 is not None:
            all_candidates.extend(circles4[0])
        
        if len(all_candidates) == 0:
            return None
            
        # Fusionner les candidats proches
        merged_circles = self.merge_close_circles(np.array(all_candidates))
        
        # Sélectionner le meilleur
        return self.find_best_circle(merged_circles, img.shape)
    
    def merge_close_circles(self, circles: np.ndarray, threshold: float = 20) -> np.ndarray:
        """
        Fusionne les cercles proches (probablement le même cercle détecté plusieurs fois).
        
        Args:
            circles: Array de cercles
            threshold: Distance maximale entre centres pour fusionner
            
        Returns:
            Cercles fusionnés
        """
        if len(circles) == 0:
            return circles
            
        merged = []
        used = [False] * len(circles)
        
        for i in range(len(circles)):
            if used[i]:
                continue
                
            cluster = [circles[i]]
            used[i] = True
            
            for j in range(i + 1, len(circles)):
                if used[j]:
                    continue
                    
                dist = np.sqrt((circles[i][0] - circles[j][0])**2 + 
                              (circles[i][1] - circles[j][1])**2)
                
                if dist < threshold:
                    cluster.append(circles[j])
                    used[j] = True
            
            # Moyenne des cercles du cluster
            cluster = np.array(cluster)
            avg_circle = np.mean(cluster, axis=0)
            merged.append(avg_circle)
        
        return np.array(merged)
    
    def detect(self, image_path: str) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Détecte le cercle principal dans l'image.
        
        Args:
            image_path: Chemin vers l'image PNG
            
        Returns:
            Tuple (cercle détecté, image originale)
        """
        # Lecture de l'image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")
        
        # Auto-calibration du rayon attendu si nécessaire
        if self.expected_radius == 50:  # Valeur par défaut
            # Estimation basée sur la taille de l'image
            img_size = min(img.shape[0], img.shape[1])
            self.expected_radius = int(img_size * 0.08)  # ~8% de la taille min
            self.min_radius = int(self.expected_radius * (1 - self.radius_tolerance))
            self.max_radius = int(self.expected_radius * (1 + self.radius_tolerance))
        
        # Détection multi-stratégie
        best_circle = self.detect_multi_strategy(img)
        
        return best_circle, img
    
    def draw_result(self, img: np.ndarray, circle: Optional[np.ndarray], 
                   output_path: Optional[str] = None) -> np.ndarray:
        """
        Dessine le cercle détecté sur l'image.
        
        Args:
            img: Image originale (niveaux de gris)
            circle: Cercle détecté (x, y, rayon)
            output_path: Chemin de sauvegarde (optionnel)
            
        Returns:
            Image avec le cercle dessiné
        """
        # Conversion en couleur pour dessiner le cercle
        if len(img.shape) == 2:
            output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            output = img.copy()
        
        if circle is not None:
            x, y, r = circle.astype(int)
            
            # Dessiner le contour du cercle (vert, épais)
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
            
            # Dessiner le centre (rouge)
            cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
            
            # Ajouter les informations
            text = f"Centre: ({x}, {y}), Rayon: {r}px"
            cv2.putText(output, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Message d'échec
            cv2.putText(output, "Aucun cercle detecte", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, output)
            
        return output


def process_image(input_path: str, output_path: Optional[str] = None,
                 expected_radius: Optional[int] = None,
                 show: bool = True) -> bool:
    """
    Traite une image pour détecter le cercle de fibre optique.
    
    Args:
        input_path: Chemin de l'image d'entrée
        output_path: Chemin de sauvegarde (optionnel)
        expected_radius: Rayon attendu en pixels (optionnel)
        show: Afficher le résultat
        
    Returns:
        True si un cercle a été détecté
    """
    # Initialisation du détecteur
    radius = expected_radius if expected_radius else 50
    detector = FiberCircleDetector(expected_radius=radius, radius_tolerance=0.15)
    
    try:
        # Détection
        circle, img = detector.detect(input_path)
        
        # Génération du nom de sortie si non spécifié
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_detected.png"
        
        # Dessin du résultat
        result = detector.draw_result(img, circle, output_path)
        
        if circle is not None:
            print(f"✓ Cercle détecté dans {input_path}")
            print(f"  Position: ({int(circle[0])}, {int(circle[1])})")
            print(f"  Rayon: {int(circle[2])} pixels")
        else:
            print(f"✗ Aucun cercle détecté dans {input_path}")
        
        # Affichage si demandé
        if show:
            cv2.imshow('Detection de fibre optique', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"→ Résultat sauvegardé: {output_path}")
        
        return circle is not None
        
    except Exception as e:
        print(f"Erreur lors du traitement de {input_path}: {e}")
        return False


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Détection de cercles de fibre optique dans des images PNG"
    )
    parser.add_argument("input", help="Chemin vers l'image PNG d'entrée")
    parser.add_argument("-o", "--output", help="Chemin de sortie (optionnel)")
    parser.add_argument("-r", "--radius", type=int, 
                       help="Rayon attendu en pixels (auto-détecté si non spécifié)")
    parser.add_argument("--no-show", action="store_true",
                       help="Ne pas afficher le résultat")
    
    args = parser.parse_args()
    
    # Traitement
    success = process_image(
        args.input,
        args.output,
        args.radius,
        show=not args.no_show
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
