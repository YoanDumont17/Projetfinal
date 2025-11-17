import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from scipy import ndimage
import glob
import os
import re
from PIL import Image


def entier_final(nom):
    """Extrait le dernier nombre du nom de fichier"""
    m = re.search(r'(-?\d+)(?!.*\d)', os.path.splitext(nom)[0])
    return int(m.group(1)) if m else np.inf

class AnalyseurNettete:
    
    def __init__(self):
        self.resultats = {}
    
    def metrique_haute_frequence(self, image, mask=None):
        if mask is not None:
            image = image * (mask / 255)  # Applique masque
        
        fft = fft2(image)
        fft_shift = fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        rows, cols = image.shape
        centre_y, centre_x = rows//2, cols//2
        
        Y, X = np.ogrid[:rows, :cols]
        distances = np.sqrt((X-centre_x)**2 + (Y-centre_y)**2)
        
        rayon_min = min(centre_x, centre_y) * 0.3
        masque_hautes_freq = distances > rayon_min
        
        energie_HF = np.sum(magnitude[masque_hautes_freq]**2)
        
        # Normaliser par area si masque
        if mask is not None:
            area = np.sum(mask > 0)
            return energie_HF / area if area > 0 else 0
        return energie_HF
    
    def metrique_laplacien(self, image, mask=None):
        laplacien = ndimage.laplace(image)
        if mask is not None:
            laplacien = laplacien * (mask / 255)
        return np.var(laplacien[laplacien != 0]) if np.any(laplacien != 0) else 0

    def metrique_laplacien8(self, image, mask=None):
        laplacien = ndimage.convolve(image, np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
        if mask is not None:
            laplacien = laplacien * (mask / 255)
        return np.var(laplacien[laplacien != 0]) if np.any(laplacien != 0) else 0
    
    def metrique_gradient(self, image, mask=None):
        grad_x = ndimage.sobel(image, axis=0)
        grad_y = ndimage.sobel(image, axis=1)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        if mask is not None:
            gradient_magnitude = gradient_magnitude * (mask / 255)
        sum_mag = np.sum(gradient_magnitude[gradient_magnitude != 0]) if np.any(gradient_magnitude != 0) else 0
        
        # Normaliser par area
        if mask is not None:
            area = np.sum(mask > 0)
            return sum_mag / area if area > 0 else 0
        return sum_mag
    
    def analyser_dossier(self, chemin_dossier):
        """Analyser toutes les images PNG d'un dossier"""
        chemins_images = glob.glob(os.path.join(chemin_dossier, "*.png"))
        
        resultats_complets = {
            'noms': [],
            'z_values': [],  
            'haute_freq': [],
            'laplacien': [],
            'laplacien8': [],
            'gradient': []
        }
        
        for chemin in chemins_images:
            nom = os.path.basename(chemin)
            z_val = entier_final(nom)
            print(f"Analyse de {nom} (z={z_val})...")
            
            # Charger l'image
            img = Image.open(chemin)
            if img.mode == 'RGB':
                img = img.convert('L')
            img_array = np.array(img, dtype=np.float64)
            
            # Créer masque automatique : non-zéro pixels (extérieur masque=0)
            mask = (img_array > 0).astype(np.uint8) * 255
            
            # Normaliser l'image (seulement sur région masquée)
            if np.any(mask):
                mean = np.mean(img_array[mask == 255])
                std = np.std(img_array[mask == 255])
                if std > 0:
                    img_array = (img_array - mean) / std
            
            # Calculer métriques avec masque
            resultats_complets['noms'].append(nom)
            resultats_complets['z_values'].append(z_val)
            resultats_complets['haute_freq'].append(self.metrique_haute_frequence(img_array, mask))
            resultats_complets['laplacien'].append(self.metrique_laplacien(img_array, mask))
            resultats_complets['laplacien8'].append(self.metrique_laplacien8(img_array, mask))
            resultats_complets['gradient'].append(self.metrique_gradient(img_array, mask))
        
        self.resultats = resultats_complets
        return resultats_complets
    
    def trouver_z_optimal_simple(self, resultats):
        """Trouve le z optimal de manière simple"""
        
        z_values = np.array(resultats['z_values'])
        
        # Dictionnaire pour stocker le z optimal de chaque métrique
        z_optimaux = {}
        
        # Normaliser et trouver le maximum pour chaque métrique
        for metrique in ['haute_freq', 'laplacien', 'laplacien8', 'gradient']:
            valeurs = np.array(resultats[metrique])
            # Trouver l'index du maximum
            idx_max = np.argmax(valeurs)
            z_optimaux[metrique] = z_values[idx_max]
        
        # Moyenne pondérée (on privilégie le Laplacien8)
        z_optimal = (0.35*z_optimaux['laplacien8']  +
                    0.1*z_optimaux['haute_freq'] + 
                    0.25*z_optimaux['laplacien'] + 
                    0.30*z_optimaux['gradient'])
        
        # Incertitude = écart entre min et max divisé par 2
        z_min = min(z_optimaux.values())
        z_max = max(z_optimaux.values())
        incertitude = (z_max - z_min) / 2
        
        return z_optimal, incertitude, z_optimaux
    
    def afficher_resultats_simple(self, resultats):      
        z_values = np.array(resultats['z_values'])
        z_optimal, incertitude, z_optimaux = self.trouver_z_optimal_simple(resultats)
        
        # Configuration du style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')
       
        couleurs = {
            'haute_freq': '#3498db',  
            'laplacien': '#e74c3c',   
            'laplacien8': '#2ecc71',  
            'gradient': '#f39c12'     
        }
        
        configs = [
            ('haute_freq', 'Haute Fréquence (FFT)', axes[0,0]),
            ('laplacien', 'Laplacien C4', axes[0,1]),
            ('laplacien8', 'Laplacien C8 ', axes[1,0]),
            ('gradient', 'Gradient (Sobel)', axes[1,1])
        ]
        
        for metrique, titre, ax in configs:
            valeurs = np.array(resultats[metrique])
            valeurs_norm = (valeurs - valeurs.min()) / (valeurs.max() - valeurs.min())
            
            ax.plot(z_values, valeurs_norm, 'o-', 
                   color=couleurs[metrique], 
                   markersize=6, 
                   linewidth=2,
                   alpha=0.7)
            
            # Marquer le maximum
            idx_max = np.argmax(valeurs_norm)
            ax.plot(z_values[idx_max], valeurs_norm[idx_max], 
                   'o', color=couleurs[metrique], 
                   markersize=12, 
                   markeredgecolor='white',
                   markeredgewidth=2)
            
            # Ligne verticale au z optimal de cette métrique
            ax.axvline(x=z_optimaux[metrique], 
                      color=couleurs[metrique], 
                      linestyle='--', 
                      alpha=0.5,
                      linewidth=1.5)
            
            # Annotations
            ax.text(z_optimaux[metrique], 0.05, 
                   f'z = {z_optimaux[metrique]:.0f}',
                   ha='center',
                   fontsize=10,
                   color=couleurs[metrique],
                   fontweight='bold',
                   transform=ax.get_xaxis_transform())
            
            ax.set_xlabel('Position z', fontsize=11)
            ax.set_ylabel('Score normalisé', fontsize=11)
            ax.set_title(titre, fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, 1.1)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Position Focale Optimale: z = {z_optimal:.0f} ± {incertitude:.0f}', 
                    fontsize=16, 
                    fontweight='bold',
                    y=1.02)   
        plt.tight_layout()
        plt.show()
        
  
        print(f"\n Z OPTIMAL: {z_optimal:.0f} ± {incertitude:.0f}\n")
        print("Détails par méthode:")
   
        for metrique, nom in [('laplacien8', 'Laplacien C8 (le plus fiable)'),
                              ('laplacien', 'Laplacien C4'),
                              ('gradient', 'Gradient'),
                              ('haute_freq', 'Haute Fréquence')]:
            print(f"  {nom:30s}: z = {z_optimaux[metrique]:.0f}")