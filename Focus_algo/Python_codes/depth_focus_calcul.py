#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter1d
import glob
import re


def entier_final(nom):
    m = re.search(r'(-?\d+)(?!.*\d)', os.path.splitext(nom)[0])
    return int(m.group(1)) if m else np.inf


class AnalyseurNettete:
    
    def __init__(self):
        self.resultats = {}
        self.preblur_kernel = (5, 5)
        self.preblur_sigma = 0.8
    
    def preprocess(self, img):
        return cv2.GaussianBlur(img, self.preblur_kernel, self.preblur_sigma)

    def metrique_haute_frequence(self, image):
        image = self.preprocess(image)
        fft = fft2(image)
        fft_shift = fftshift(fft)
        magnitude = np.abs(fft_shift)
        rows, cols = image.shape
        cy, cx = rows // 2, cols // 2
        Y, X = np.ogrid[:rows, :cols]
        distances = np.sqrt((X - cx)**2 + (Y - cy)**2)
        rayon_min = min(cx, cy) * 0.75
        return np.sum(magnitude[distances > rayon_min] ** 2)

    def metrique_brenner(self, image):
        image = self.preprocess(image)
        return np.sum((image[2:, :] - image[:-2, :]) ** 2)

    def metrique_tenengrad(self, image, ksize=5):
        image = self.preprocess(image)
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        return np.mean(gx*gx + gy*gy)

    def analyser_dossier(self, chemin_dossier, smooth_sigma=3.0):
        chemins_images = sorted(glob.glob(os.path.join(chemin_dossier, "*.png")))
        
        resultats = {
            'noms': [], 'z_values': [],
            'haute_freq': [], 'brenner': [], 'tenengrad': []
        }
        
        for chemin in chemins_images:
            nom = os.path.basename(chemin)
            z_val = entier_final(nom)
            print(f"Analyse de {nom} (z={z_val})...")
            
            img = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img_float = img.astype(np.float64)

            resultats['noms'].append(nom)
            resultats['z_values'].append(z_val)
            resultats['haute_freq'].append(self.metrique_haute_frequence(img_float))
            resultats['brenner'].append(self.metrique_brenner(img_float))
            resultats['tenengrad'].append(self.metrique_tenengrad(img_float))
        
        # Lissage des courbes (indispensable pour stacks 1 µm)
        for met in ['haute_freq', 'brenner', 'tenengrad']:
            resultats[met] = gaussian_filter1d(resultats[met], sigma=smooth_sigma)
        
        self.resultats = resultats
        return resultats

    def trouver_z_optimal_simple(self, resultats):
        z = np.array(resultats['z_values'])

        z_tenengrad = z[np.argmax(resultats['tenengrad'])]
        z_brenner   = z[np.argmax(resultats['brenner'])]
        z_hf        = z[np.argmax(resultats['haute_freq'])]

        z_final = round(0.55 * z_tenengrad + 0.40 * z_brenner + 0.05 * z_hf)
        incertitude = max(abs(z_final - z_tenengrad), abs(z_final - z_brenner))

        return z_final, incertitude, {
            'tenengrad': z_tenengrad,
            'brenner'  : z_brenner,
            'haute_freq': z_hf
        }

    def afficher_resultats_simple(self, resultats):      
        z_values = np.array(resultats['z_values'])
        z_optimal, incertitude, z_optimaux = self.trouver_z_optimal_simple(resultats)

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor('white')
       
        couleurs = {
            'haute_freq': '#3498db', 
            'brenner'   : '#e74c3c', 
            'tenengrad' : '#2ecc71'      # ← CORRIGÉ ICI !
        }
        configs = [
            ('haute_freq', 'Haute Fréquence (FFT)', axes[0]),
            ('brenner',    'Brenner',               axes[1]),
            ('tenengrad',  'Tenengrad (Sobel)',     axes[2]),
        ]
        
        for metrique, titre, ax in configs:
            valeurs = np.array(resultats[metrique])
            valeurs_norm = (valeurs - valeurs.min()) / (valeurs.max() - valeurs.min() + 1e-8)
            
            ax.plot(z_values, valeurs_norm, 'o-', color=couleurs[metrique], markersize=7, linewidth=2.5)
            idx_max = np.argmax(valeurs)
            ax.plot(z_values[idx_max], valeurs_norm[idx_max], 'o', color=couleurs[metrique], 
                   markersize=14, markeredgecolor='white', markeredgewidth=2.5)
            ax.axvline(x=z_optimaux[metrique], color=couleurs[metrique], linestyle='--', alpha=0.7, linewidth=2)
            ax.text(z_optimaux[metrique], 0.07, f'z = {z_optimaux[metrique]} µm', 
                   ha='center', fontsize=12, color=couleurs[metrique], fontweight='bold',
                   transform=ax.get_xaxis_transform())
            ax.set_title(titre, fontsize=14, fontweight='bold')
            ax.set_xlabel('Position z (µm)', fontsize=12)
            ax.set_ylabel('Score normalisé', fontsize=12)
            ax.set_ylim(-0.05, 1.15)
            ax.grid(True, alpha=0.4)

        fig.suptitle(f'POSITION FOCALE OPTIMALE: z = {z_optimal} µm ± {incertitude} µm', 
                    fontsize=18, fontweight='bold', y=1.05)   
        plt.tight_layout()
        plt.show()
        
        print(f"\n>>> Z OPTIMAL: {z_optimal} µm ± {incertitude} µm <<<\n")
        print("Détails :")
        print(f"  Tenengrad : z = {z_optimaux['tenengrad']} µm")
        print(f"  Brenner   : z = {z_optimaux['brenner']} µm")
        print(f"  Haute Fréq: z = {z_optimaux['haute_freq']} µm")