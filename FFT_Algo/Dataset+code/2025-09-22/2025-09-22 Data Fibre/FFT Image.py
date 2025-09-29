import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from numpy.fft import fft2 as np_fft2
import os
from PIL import Image
import glob


dossier = r"FFT_Algo\2025-09-22 Data Fibre"

chemin_images = glob.glob(os.path.join(dossier, "*.png"))

images = []
images_fft = []
noms_fichiers = []

for chemin in chemin_images:
    # Extraire le nom du fichier
    nom_fichier = os.path.basename(chemin)
    noms_fichiers.append(nom_fichier)
    
    print(f"Traitement de : {nom_fichier}")
    
    # Charger l'image
    img = Image.open(chemin)
    
    
    # Convertir en array numpy
    img_array = np.array(img)
    print(img_array.shape)
    images.append(img_array)
    
    # Appliquer la FFT2
    fft_result = np_fft2(img_array)
    images_fft.append(fft_result)
    
    # Calculer le spectre d'amplitude (optionnel)
    magnitude_spectrum = np.log(np.abs(fft_result) + 1)  # +1 pour éviter log(0)
    
    # Optionnel : Afficher l'image originale et son spectre FFT
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Image originale
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title(f'Image originale\n{nom_fichier}')
    axes[0].axis('off')
    
    # Spectre d'amplitude (centré)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum_centered = np.log(np.abs(fft_shifted) + 1)
    axes[1].imshow(magnitude_spectrum_centered, cmap='gray')
    axes[1].set_title('Spectre FFT (centré)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

print(f"\nNombre total d'images traitées : {len(images)}")