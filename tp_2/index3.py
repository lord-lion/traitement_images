import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Vérifions l'existence des fichiers image
image_paths = [
    '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ.png',
    '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png',
    '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ_sinus.png',
    '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_zd6ypc20QAIFMzrbCmJRMg.png'
]

for image_path in image_paths:
    if not os.path.exists(image_path):
        print(f"Le fichier {image_path} n'existe pas.")
        exit()

# Chargement des images en niveaux de gris
images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# Définition du filtre de renforcement
filter_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

# Appliquons la convolution avec le filtre à chaque image
filtered_images = [cv2.filter2D(image, -1, filter_kernel) for image in images]

# Affichons les images originales et les images après convolution
plt.figure(figsize=(12, 6))
for i, (image, filtered_image, image_path) in enumerate(zip(images, filtered_images, image_paths)):
    plt.subplot(2, len(image_paths), i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Image {i + 1}')
    plt.axis('off')

    plt.subplot(2, len(image_paths), len(image_paths) + i + 1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Image {i + 1} après convolution')
    plt.axis('off')

plt.tight_layout()
plt.show()
