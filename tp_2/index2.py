# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Charger l'image en niveaux de gris
# image = cv2.imread('images/boat.png', cv2.IMREAD_GRAYSCALE)


# # Méthode 1 : Seuillage sur la norme du gradient
# # Appliquer un filtre de Sobel pour calculer le gradient
# gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
# gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# # Calculer la norme du gradient
# gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

# # Choisir un seuil pour le seuillage
# threshold = 50

# # Appliquer le seuillage
# edges_gradient = (gradient_magnitude > threshold).astype(np.uint8) * 255

# # Méthode 2 : Filtre Canny
# # Appliquer le filtre Canny avec différents seuils
# low_threshold = 50
# high_threshold = 150
# edges_canny = cv2.Canny(image, low_threshold, high_threshold)

# # Afficher les résultats
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 3, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Image originale')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(edges_gradient, cmap='gray')
# plt.title('Contours (Seuillage sur le gradient)')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(edges_canny, cmap='gray')
# plt.title('Contours (Filtre Canny)')
# plt.axis('off')

# plt.tight_layout()
# plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Liste des noms de fichiers des images à tester
image_files = [
    '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ.png',
    '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png',
    '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ_sinus.png',
    '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_zd6ypc20QAIFMzrbCmJRMg.png'
]

# Seuil pour le seuillage de la norme du gradient
threshold = 50

# Seuils pour le filtre Canny
low_threshold = 50
high_threshold = 150

# Boucle sur les images
for image_file in image_files:
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Méthode 1 : Seuillage sur la norme du gradient
    # Appliquer un filtre de Sobel pour calculer le gradient
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculer la norme du gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Appliquer le seuillage
    edges_gradient = (gradient_magnitude > threshold).astype(np.uint8) * 255

    # Méthode 2 : Filtre Canny
    # Appliquer le filtre Canny
    edges_canny = cv2.Canny(image, low_threshold, high_threshold)

    # Afficher les résultats
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image originale')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(edges_gradient, cmap='gray')
    plt.title('Contours (Seuillage sur le gradient)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(edges_canny, cmap='gray')
    plt.title('Contours (Filtre Canny)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
