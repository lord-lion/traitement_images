# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Chargement de l'image en niveaux de gris
# image = cv2.imread('images/boat.', cv2.IMREAD_GRAYSCALE)

# # Bruit gaussien ajouté à l'image (pour les tests)
# noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
# noisy_image = cv2.add(image, noise)

# # Appliquons le filtre moyenneur avec différentes tailles de noyau
# kernel_sizes = [3, 5, 7]
# for kernel_size in kernel_sizes:
#     filtered_image = cv2.blur(noisy_image, (kernel_size, kernel_size))
#     plt.subplot(2, 4, 1)
#     plt.imshow(filtered_image, cmap='gray')
#     plt.title(f'Filtre Moyenneur {kernel_size}x{kernel_size}')
#     plt.axis('off')

# # Appliquons le filtre gaussien avec différentes valeurs d'écart type (sigma)
# sigmas = [1, 3, 5]
# for sigma in sigmas:
#     filtered_image = cv2.GaussianBlur(noisy_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
#     plt.subplot(2, 4, 2)
#     plt.imshow(filtered_image, cmap='gray')
#     plt.title(f'Filtre Gaussien Sigma {sigma}')
#     plt.axis('off')

# # Appliquons le filtre médian avec différentes tailles de noyau
# kernel_sizes = [3, 5, 7]
# for kernel_size in kernel_sizes:
#     filtered_image = cv2.medianBlur(noisy_image, kernel_size)
#     plt.subplot(2, 4, 3)
#     plt.imshow(filtered_image, cmap='gray')
#     plt.title(f'Filtre Médian {kernel_size}x{kernel_size}')
#     plt.axis('off')

# # Appliquons le filtre max (dilation) et min (érosion) avec différentes tailles de noyau
# kernel_sizes = [3, 5, 7]
# for kernel_size in kernel_sizes:
#     max_filtered = cv2.dilate(noisy_image, np.ones((kernel_size, kernel_size), np.uint8))
#     min_filtered = cv2.erode(noisy_image, np.ones((kernel_size, kernel_size), np.uint8))
    
#     plt.subplot(2, 4, 4)
#     plt.imshow(max_filtered, cmap='gray')
#     plt.title(f'Filtre Max {kernel_size}x{kernel_size}')
#     plt.axis('off')
    
#     plt.subplot(2, 4, 5)
#     plt.imshow(min_filtered, cmap='gray')
#     plt.title(f'Filtre Min {kernel_size}x{kernel_size}')
#     plt.axis('off')

# # Affichons l'image d'origine et l'image bruitée
# plt.subplot(2, 4, 6)
# plt.imshow(image, cmap='gray')
# plt.title('Image d\'origine')
# plt.axis('off')

# plt.subplot(2, 4, 7)
# plt.imshow(noisy_image, cmap='gray')
# plt.title('Image bruitée')
# plt.axis('off')

# plt.tight_layout()
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Liste des noms de fichiers des images à tester
# image_files = [
#     '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ.png',
#     '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ.png_Salt_Pepper_Noise1.png',
#     '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ_sinus.png',
#     '/home/oem/github.com/TRAITEMENT_IMAGES/tp_2/images_TP1-2/1_zd6ypc20QAIFMzrbCmJRMg.png'
# ]

# # Bruit gaussien standard
# noise_std = 25

# # Tailles de noyau pour les filtres
# kernel_sizes = [3, 5, 7]

# # Boucle sur les images
# for image_file in image_files:
#     # Charger l'image en niveaux de gris
#     image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

#     # Bruit gaussien ajouté à l'image (pour les tests)
#     noise = np.random.normal(0, noise_std, image.shape).astype(np.uint8)
#     noisy_image = cv2.add(image, noise)

#     # Créer une nouvelle figure pour chaque image
#     plt.figure(figsize=(12, 6))
#     plt.subplot(2, 4, 1)
#     plt.imshow(noisy_image, cmap='gray')
#     plt.title('Image bruitée')
#     plt.axis('off')

#     # Appliquer le filtre moyenneur avec différentes tailles de noyau
#     for i, kernel_size in enumerate(kernel_sizes):
#         filtered_image = cv2.blur(noisy_image, (kernel_size, kernel_size))
#         plt.subplot(2, 4, i + 2)
#         plt.imshow(filtered_image, cmap='gray')
#         plt.title(f'Filtre Moyenneur {kernel_size}x{kernel_size}')
#         plt.axis('off')

#     # Afficher l'image d'origine
#     plt.subplot(2, 4, 6)
#     plt.imshow(image, cmap='gray')
#     plt.title('Image d\'origine')
#     plt.axis('off')

#     # Ajustement de la disposition des sous-graphiques
#     plt.tight_layout()
#     plt.show()


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

# Bruit gaussien standard
noise_std = 25

# Tailles de noyau pour les filtres
kernel_sizes = [3, 5, 7]

# Boucle sur les images
for image_file in image_files:
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Bruit gaussien ajouté à l'image (pour les tests)
    noise = np.random.normal(0, noise_std, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    # Créer une nouvelle figure pour chaque image
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 8, 1)
    plt.imshow(noisy_image, cmap='gray')
    plt.title('Image bruitée')
    plt.axis('off')

    # Appliquer le filtre moyenneur avec différentes tailles de noyau
    for i, kernel_size in enumerate(kernel_sizes):
        filtered_image = cv2.blur(noisy_image, (kernel_size, kernel_size))
        plt.subplot(2, 8, i + 2)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Filtre Moyenneur {kernel_size}x{kernel_size}')
        plt.axis('off')

    # Appliquer le filtre gaussien avec différentes valeurs d'écart type (sigma)
    for i, sigma in enumerate([1, 3, 5]):
        filtered_image = cv2.GaussianBlur(noisy_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        plt.subplot(2, 8, i + 5)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Filtre Gaussien Sigma {sigma}')
        plt.axis('off')

    # Appliquer le filtre médian avec différentes tailles de noyau
    for i, kernel_size in enumerate(kernel_sizes):
        filtered_image = cv2.medianBlur(noisy_image, kernel_size)
        plt.subplot(2, 8, i + 8)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'Filtre Médian {kernel_size}x{kernel_size}')
        plt.axis('off')

    # Appliquer le filtre max (dilation) et min (érosion) avec différentes tailles de noyau
    for i, kernel_size in enumerate(kernel_sizes):
        max_filtered = cv2.dilate(noisy_image, np.ones((kernel_size, kernel_size), np.uint8))
        min_filtered = cv2.erode(noisy_image, np.ones((kernel_size, kernel_size), np.uint8))

        plt.subplot(2, 8, i + 11)
        plt.imshow(max_filtered, cmap='gray')
        plt.title(f'Filtre Max {kernel_size}x{kernel_size}')
        plt.axis('off')

        plt.subplot(2, 8, i + 14)
        plt.imshow(min_filtered, cmap='gray')
        plt.title(f'Filtre Min {kernel_size}x{kernel_size}')
        plt.axis('off')

    # Afficher l'image d'origine
    plt.subplot(2, 8, 16)
    plt.imshow(image, cmap='gray')
    plt.title('Image d\'origine')
    plt.axis('off')

    # Ajustement de la disposition des sous-graphiques
    plt.tight_layout()
    plt.show()
