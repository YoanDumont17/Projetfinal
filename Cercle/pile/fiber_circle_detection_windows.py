#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection robuste de cercle de fibre optique dans des images PNG
Projet de fin d'etudes - Genie Physique
Version compatible Windows
"""
import cv2
import numpy as np
import os
import sys
from typing import Tuple, Optional, List
import argparse
# Force l'encodage UTF-8 pour Windows
if sys.platform == 'win32':
    import locale
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
class FiberCircleDetector:
    def __init__(self, expected_radius: int = 0, radius_tolerance: float = 0.6): # Tolérance augmentée
        self.expected_radius = expected_radius
        self.radius_tolerance = radius_tolerance
       
    def preprocess_image(self, img: np.ndarray, enhance_contrast: bool = True) -> np.ndarray:
        img_size = min(img.shape[:2])
        kernel_size = max(5, int(img_size * 0.015))
       
        denoised = cv2.bilateralFilter(img, kernel_size, 75, 75)
       
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
        else:
            enhanced = denoised
           
        cleaned = cv2.medianBlur(enhanced, kernel_size)
       
        # Supprimer stries : kernels horizontal, vertical, et diagonaux
        line_kernel_h = np.ones((1, kernel_size * 2), np.uint8)
        line_kernel_v = np.ones((kernel_size * 2, 1), np.uint8)
        line_kernel_diag1 = np.diag(np.ones(kernel_size * 2, np.uint8)) # Diagonale \
        line_kernel_diag2 = np.fliplr(line_kernel_diag1) # Diagonale /
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, line_kernel_h)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, line_kernel_v)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, line_kernel_diag1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, line_kernel_diag2)
       
        return cleaned
   
    def detect_edges_adaptive(self, img: np.ndarray) -> np.ndarray:
        img_size = min(img.shape[:2])
        kernel_size = max(3, int(img_size * 0.01))
       
        v = np.median(img)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
       
        edges = cv2.Canny(img, lower, upper)
       
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
       
        return edges
   
    def find_best_circle(self, circles: np.ndarray, img_shape: Tuple[int, int], img: np.ndarray, scale_factor: float = 1.0) -> Optional[np.ndarray]:
        if circles is None or len(circles) == 0:
            return None
           
        height, width = img_shape
        best_circle = None
        best_score = -1
       
        for circle in circles:
            x, y, r = circle
           
            # Nouveau: Exclure les cercles en dehors de la plage de diamètre 510-550 pixels (ajusté pour scale)
            original_diameter = 2 * (r / scale_factor)
            if original_diameter < 510 or original_diameter > 550:
                print(f"Debug: Candidate discarded - original diameter {original_diameter:.2f} outside 510-550")
                continue
           
            mask = np.zeros(img_shape, dtype=np.uint8)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
           
            score = 0
           
            radius_diff = abs(r - self.expected_radius) / self.expected_radius if self.expected_radius > 0 else 0
            radius_score = max(0, 1 - radius_diff) * 2.0
            score += radius_score
           
            margin = r * 0.3
            if x - r >= -margin and x + r <= width + margin and \
               y - r >= -margin and y + r <= height + margin:
                completeness_score = 1.0
            else:
                visible_portion = self.estimate_visible_portion(x, y, r, width, height)
                completeness_score = visible_portion
            score += completeness_score
           
            center_x, center_y = width / 2, height / 2
            dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            center_score = 1 - (dist_to_center / max_dist) * 0.5 # Poids augmenté pour favoriser centre
            score += center_score
           
            mean_brightness = cv2.mean(img, mask=mask)[0] / 255.0
            brightness_score = mean_brightness * 3.0
            score += brightness_score
           
            mean, stddev = cv2.meanStdDev(img, mask=mask)
            homogeneity = stddev[0][0] / 255.0
            homogeneity_score = max(0, 1 - homogeneity * 6.0) # Plus strict (*6 au lieu de 5)
            score += homogeneity_score * 5.0 # Poids augmenté
           
            if homogeneity > 0.1: # Seuil baissé pour plus de strictness
                print(f"Debug: Candidate discarded - homogeneity {homogeneity:.4f} > 0.1")
                continue
           
            # Nouveau: Score de circularité (fit ellipse)
            # Trouver contours dans la zone masque pour fit ellipse
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            thresh = cv2.threshold(masked_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                ellipse = cv2.fitEllipse(max(contours, key=cv2.contourArea))
                (major, minor) = ellipse[1]
                circularity = min(major, minor) / max(major, minor)
                circularity_score = circularity * 2.0 # Poids pour favoriser rondeur
                score += circularity_score
                if circularity < 0.8: # Discard si pas assez rond
                    print(f"Debug: Candidate discarded - circularity {circularity:.4f} < 0.8")
                    continue
           
            print(f"Debug: Candidate circle - scaled r={r:.2f}, original diameter={original_diameter:.2f}, score={score:.2f}")
           
            if score > best_score:
                best_score = score
                best_circle = circle
               
        return best_circle
   
    def estimate_visible_portion(self, x: float, y: float, r: float,
                                 width: int, height: int) -> float:
        left_overflow = max(0, r - x) / r
        right_overflow = max(0, (x + r) - width) / r
        top_overflow = max(0, r - y) / r
        bottom_overflow = max(0, (y + r) - height) / r
       
        visible = 1.0 - (left_overflow + right_overflow + top_overflow + bottom_overflow) / 4
        return max(0.3, visible)
   
    def detect_multi_strategy(self, img: np.ndarray) -> Optional[np.ndarray]:
        orig_shape = img.shape[:2]
        scale_factor = 1.0
       
        max_dim = max(orig_shape)
        if max_dim > 1024:
            scale_factor = 1024 / max_dim
            new_size = (int(orig_shape[1] * scale_factor), int(orig_shape[0] * scale_factor))
            img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            img_resized = img
       
        img_size = min(img_resized.shape[:2])
       
        if self.expected_radius <= 0:
            self.expected_radius = int(img_size * 0.3) # Augmenté pour grands cercles
        self.min_radius = int(self.expected_radius * (1 - self.radius_tolerance))
        self.max_radius = int(self.expected_radius * (1 + self.radius_tolerance))
        print(f"Debug: Rayon attendu {self.expected_radius}, min {self.min_radius}, max {self.max_radius}")
       
        all_candidates = []
       
        preprocessed = self.preprocess_image(img_resized)
        cv2.imwrite("debug_preprocessed.png", preprocessed)
       
        # Ajout: Stratégie avec edges pour mieux isoler
        edges = self.detect_edges_adaptive(preprocessed)
        cv2.imwrite("debug_edges.png", edges)
        circles_edges = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1.5,
            minDist=self.expected_radius * 2, param1=50, param2=35, # Plus strict
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        if circles_edges is not None:
            all_candidates.extend(circles_edges[0])
            print("Debug: Edges strat found", len(circles_edges[0]))
       
        circles1 = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1.5,
            minDist=self.expected_radius * 2, param1=40, param2=35, # param2 augmenté
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        if circles1 is not None:
            all_candidates.extend(circles1[0])
            print("Debug: Strat1 found", len(circles1[0]))
       
        # Autres stratégies avec minDist augmenté pour éviter clusters
        circles2 = cv2.HoughCircles(
            preprocessed, cv2.HOUGH_GRADIENT, dp=1.5,
            minDist=self.expected_radius * 2, param1=30, param2=30,
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        if circles2 is not None:
            all_candidates.extend(circles2[0])
            print("Debug: Strat2 found", len(circles2[0]))
       
        blurred = cv2.GaussianBlur(img_resized, (7, 7), 1.5)
        circles3 = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.5,
            minDist=self.expected_radius * 2, param1=50, param2=40,
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        if circles3 is not None:
            all_candidates.extend(circles3[0])
            print("Debug: Strat3 found", len(circles3[0]))
       
        heavily_blurred = cv2.GaussianBlur(img_resized, (11, 11), 2.5)
        circles4 = cv2.HoughCircles(
            heavily_blurred, cv2.HOUGH_GRADIENT, dp=2.0,
            minDist=self.expected_radius * 2, param1=30, param2=25,
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        if circles4 is not None:
            all_candidates.extend(circles4[0])
            print("Debug: Strat4 found", len(circles4[0]))
       
        # Fallback amélioré: Blob detector pour régions homogènes circulaires
        if len(all_candidates) < 5:
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = self.min_radius**2 * np.pi * 0.5 # Min area pour cercle
            params.maxArea = self.max_radius**2 * np.pi * 1.5
            params.filterByCircularity = True
            params.minCircularity = 0.8 # Quasi-rond
            params.filterByConvexity = True
            params.minConvexity = 0.85
            params.filterByInertia = True
            params.minInertiaRatio = 0.8 # Rond vs elliptique
            detector = cv2.SimpleBlobDetector_create(params)
            keypoints = detector.detect(preprocessed)
            for kp in keypoints:
                x, y = kp.pt
                r = kp.size / 2
                all_candidates.append([x, y, r])
            print("Debug: Blob fallback found", len(keypoints))
       
        if len(all_candidates) == 0:
            print("Debug: No candidates at all")
            return None
       
        merged_circles = self.merge_close_circles(np.array(all_candidates))
       
        best_circle = self.find_best_circle(merged_circles, img_resized.shape[:2], img_resized, scale_factor)
       
        if best_circle is not None and scale_factor != 1.0:
            best_circle[0:2] /= scale_factor
            best_circle[2] /= scale_factor
            print("Debug: Scaled back to original")
       
        return best_circle
   
    def merge_close_circles(self, circles: np.ndarray) -> np.ndarray:
        if len(circles) == 0:
            return circles
           
        threshold = self.expected_radius * 0.3 # Augmenté pour merger plus
           
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
           
            cluster = np.array(cluster)
            avg_circle = np.mean(cluster, axis=0)
            merged.append(avg_circle)
       
        return np.array(merged)
   
    def detect(self, image_path: str) -> Tuple[Optional[np.ndarray], np.ndarray]:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Impossible de lire l'image: {image_path}")
       
        best_circle = self.detect_multi_strategy(img)
       
        return best_circle, img
   
    def draw_result(self, img: np.ndarray, circle: Optional[np.ndarray],
                   output_path: Optional[str] = None) -> np.ndarray:
        if len(img.shape) == 2:
            output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            output = img.copy()
       
        if circle is not None:
            x, y, r = circle.astype(int)
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
            cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
            text = f"Centre: ({x}, {y}), Rayon: {r}px"
            cv2.putText(output, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(output, "Aucun cercle detecte", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
       
        if output_path:
            cv2.imwrite(output_path, output)
           
        return output
# (process_image et main inchangés, mais assure-toi d'utiliser -r pour auto si besoin)
def process_image(input_path: str, output_path: Optional[str] = None,
                 expected_radius: Optional[int] = None,
                 show: bool = True) -> bool:
    radius = expected_radius if expected_radius is not None else 0
    detector = FiberCircleDetector(expected_radius=radius, radius_tolerance=0.6)
   
    try:
        circle, img = detector.detect(input_path)
       
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_detected.png"
       
        result = detector.draw_result(img, circle, output_path)
       
        if circle is not None:
            print(f"[OK] Cercle detecte dans {input_path}")
            print(f" Position: ({int(circle[0])}, {int(circle[1])})")
            print(f" Rayon: {int(circle[2])} pixels")
        else:
            print(f"[ECHEC] Aucun cercle detecte dans {input_path}")
       
        if show:
            cv2.imshow('Detection de fibre optique', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
       
        print(f"-> Resultat sauvegarde: {output_path}")
       
        return circle is not None
       
    except Exception as e:
        print(f"Erreur lors du traitement de {input_path}: {e}")
        return False

if __name__ == "__main__":
    image_dir = "Cercle\files2"  # Remplace par le chemin du dossier contenant les images si nécessaire, ex. "chemin/vers/images"
    for i in range(1, 25):
        input_path = os.path.join(image_dir, f"{i}.png")
        if os.path.exists(input_path):
            print(f"Traitement de {input_path}...")
            process_image(input_path, show=True)  # Affiche et attend une touche pour continuer
        else:
            print(f"Fichier {input_path} non trouvé, saut.")
    print("Traitement terminé pour toutes les images.")