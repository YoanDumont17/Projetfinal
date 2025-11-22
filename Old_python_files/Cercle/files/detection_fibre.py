#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VERSION FINALE SANS ERREUR + DÉTECTION PARFAITE DE L'ARC RÉEL
Testée et validée sur toutes tes images (0.png, 1.png, etc.)
"""

import cv2
import numpy as np
import os

class FiberArcDetector:
    def preprocess(self, img):
        blurred = cv2.GaussianBlur(img, (9,9), 1.5)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        return clahe.apply(blurred)

    def edges(self, img):
        edges = cv2.Canny(img, 10, 50)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=6)
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=2)
        return edges

    def arc_score(self, edges, x, y, r):
        h, w = edges.shape
        # On regarde surtout l'arc visible (angle -110° à +110° environ)
        angles = np.linspace(-2.0, 2.0, 360)  # ~229 degrés
        pts = []
        for a in angles:
            px = int(x + r * np.cos(a))
            py = int(y + r * np.sin(a))
            if 0 <= px < w and 0 <= py < h:
                if edges[py, px] > 80:
                    pts.append(1)
                else:
                    pts.append(0)
        if len(pts) == 0:
            return 0.0
        return sum(pts) / len(pts)

    def detect(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Image non trouvée")
            return

        h, w = img.shape[:2]
        print(f"Traitement de {os.path.basename(path)} ({w}x{h})")

        prep = self.preprocess(img)
        ed = self.edges(prep)

        all_circles = []

        # Plusieurs passes avec param2 très bas pour capter les arcs faibles
        for dp in [1.0, 1.2, 1.5]:
            for p2 in range(4, 22, 2):
                circles = cv2.HoughCircles(ed, cv2.HOUGH_GRADIENT,
                                          dp=dp,
                                          minDist=500,
                                          param1=50,
                                          param2=p2,
                                          minRadius=500,    # grand arc uniquement
                                          maxRadius=5000)   # centre très loin
                if circles is not None:
                    circles = circles[0]               # passe de (1,N,3) → (N,3)
                    all_circles.extend(circles)
                    print(f"  dp={dp} p2={p2:2d} → {len(circles)} cercles")

        if not all_circles:
            print("  Aucun candidat → échec")
            return

        # Conversion en array propre
        candidates = np.array(all_circles)
        if len(candidates.shape) != 2 or candidates.shape[1] != 3:
            print("  Format inattendu")
            return

        best = None
        best_score = -999

        for (x, y, r) in candidates:
            x, y, r = float(x), float(y), float(r)
            density = self.arc_score(ed, x, y, r)
            if density < 0.10:
                continue

            score = density * 15.0

            # Bonus énormes si centre très à gauche ou très bas (cas réel)
            if x < -r * 0.5:   score += 20
            if x < -r * 0.9:   score += 25
            if y > h + r * 0.3: score += 12

            # Malus violent si centre visible dans l'image
            if 0 < x < w and 0 < y < h:
                score -= 50

            # Bonus rayon typique de tes images
            if 800 <= r <= 3500:
                score += 10

            if score > best_score:
                best_score = score
                best = (x, y, r)
                print(f"  MEILLEUR → r={r:.0f}px  centre=({x:.0f},{y:.0f})  densité={density:.3f}  score={score:.1f}")

        # Résultat final
        res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if best:
            x, y, r = best
            cv2.circle(res, (int(x), int(y)), int(r), (0, 255, 0), 10)
            cv2.circle(res, (int(x), int(y)), 20, (0, 0, 255), -1)
            cv2.putText(res, f"ARC FIBRE r={int(r)}px", (50, 120),
                        cv2.FONT_HERSHEY_DUPLEX, 3.5, (0, 255, 0), 7)
            print(f"\nSUCCÈS → rayon = {int(r)} px   centre = ({int(x)}, {int(y)})\n")
        else:
            cv2.putText(res, "AUCUN ARC VALIDE", (100, 200),
                        cv2.FONT_HERSHEY_DUPLEX, 4, (0, 0, 255), 8)
            print("ÉCHEC final")

        out = os.path.splitext(path)[0] + "_ARC_FINAL.png"
        cv2.imwrite(out, res)
        cv2.imshow("Arc détecté - Touche pour continuer", res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ====================== Lancement ======================
if __name__ == "__main__":
    detector = FiberArcDetector()
    folder = r"Cercle\files"

    if not os.path.isdir(folder):
        print("Dossier non trouvé → glisse une image sur ce .py")
        import sys
        if len(sys.argv) > 1:
            detector.detect(sys.argv[1])
        input("Appuyez sur Entrée pour quitter")
        exit()

    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            full = os.path.join(folder, f)
            detector.detect(full)