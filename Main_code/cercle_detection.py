import cv2
import numpy as np

# Charge l'image
image = cv2.imread('Main_code\24.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Prétraitement
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Détection de cercles
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=100, param2=30, minRadius=10, maxRadius=100)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)  # Dessine le cercle en vert

# Affiche le résultat
cv2.imshow('Cercles détectés', image)
cv2.waitKey(0)
cv2.destroyAllWindows()