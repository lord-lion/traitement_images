# import cv2
# import numpy as np

# Charger l'image en niveaux de gris
# image = cv2.imread('images/hair_remy.jpg', cv2.IMREAD_GRAYSCALE)

# # Paramètres d'ajustement linéaire
# alpha = 1.5  # Facteur d'échelle
# beta = 50   # Facteur de translation

# # Ajustement linéaire
# adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# # Afficher les images originale et ajustée
# cv2.imshow('Image originale', image)
# cv2.imshow('Image ajustée', adjusted_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#-------------------------------------------------------------------------------------------------------------

# Charger l'image en niveaux de gris
# image = cv2.imread('images/hair_remy.jpg', cv2.IMREAD_GRAYSCALE)

# # Appliquer l'égalisation de l'histogramme
# equalized_image = cv2.equalizeHist(image)

# # Afficher les images originale et égalisée
# cv2.imshow('Image originale', image)
# cv2.imshow('Image égalisée', equalized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------------------------


# Charger l'image en niveaux de gris
# image = cv2.imread('images/hair_remy.jpg', cv2.IMREAD_GRAYSCALE)

# # Paramètre de correction gamma
# gamma = 1.5

# # Appliquer la correction gamma
# gamma_corrected = np.power(image / 255.0, gamma)
# gamma_corrected = np.uint8(gamma_corrected * 255)

# # Afficher les images originale et corrigée
# cv2.imshow('Image originale', image)
# cv2.imshow('Image corrigée', gamma_corrected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#---------------------------------------------------------------------------------------------------------


# Charger l'image en niveaux de gris
# image = cv2.imread('images/hair_remy.jpg', cv2.IMREAD_GRAYSCALE)

# # Déterminer les valeurs minimales et maximales de l'image
# min_val = np.min(image)
# max_val = np.max(image)

# # Paramètres pour l'étirement de contraste
# new_min = 0
# new_max = 255

# # Appliquer l'étirement de contraste
# stretched_image = np.uint8((image - min_val) * ((new_max - new_min) / (max_val - min_val)) + new_min)

# # Afficher les images originale et étirée
# cv2.imshow('Image originale', image)
# cv2.imshow('Image étirée', stretched_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------------------------


# Charger l'image en niveaux de gris
# image = cv2.imread('images/hair_remy.jpg', cv2.IMREAD_GRAYSCALE)

# # Créer un objet CLAHE (Contrast Limited Adaptive Histogram Equalization)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# # Appliquer l'amélioration locale du contraste
# clahe_image = clahe.apply(image)

# # Afficher les images originale et améliorée
# cv2.imshow('Image originale', image)
# cv2.imshow('Image améliorée', clahe_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#---------------------------------------------------------------------------------------------------------

# Charger les images en niveaux de gris
# Charger les images en niveaux de gris
# image1 = cv2.imread('images/hair_remy.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('images/bateau_2.png', cv2.IMREAD_GRAYSCALE)

# # Redimensionner les images à une taille fixe
# image1 = cv2.resize(image1, (20, 40))
# image2 = cv2.resize(image2, (20, 40))

# # Appliquer l'opération logique "ET"
# result_and = cv2.bitwise_and(image1, image2)

# # Appliquer l'opération logique "OU"
# result_or = cv2.bitwise_or(image1, image2)

# # Afficher les images d'origine et les résultats
# cv2.imshow('Image 1', image1)
# cv2.imshow('Image 2', image2)
# cv2.imshow('Résultat "ET"', result_and)
# cv2.imshow('Résultat "OU"', result_or)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#--------------------------------------------------------------------------------------------

# 1. Utilisation de l'égalisation de l'histogramme avec OpenCV :



# Charger l'image en niveaux de gris
# image = cv2.imread('images/hair_remy.jpg', cv2.IMREAD_GRAYSCALE)

# # Appliquer l'égalisation de l'histogramme avec OpenCV
# equalized_image_opencv = cv2.equalizeHist(image)

# # Afficher l'image originale et l'image égalisée par OpenCV
# cv2.imshow('Image originale', image)
# cv2.imshow('Image égalisée (OpenCV)', equalized_image_opencv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 2. Implémentation de la transformation linéaire :


# Charger l'image en niveaux de gris
# image = cv2.imread('images/hair_remy.jpg', cv2.IMREAD_GRAYSCALE)

# # Ajustement linéaire
# alpha = 1.5
# beta = 30
# linear_transformed_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

# # Afficher l'image originale et l'image avec transformation linéaire
# cv2.imshow('Image originale', image)
# cv2.imshow('Image avec transformation linéaire', linear_transformed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 3. Implémentation de la correction gamma :



# Charger l'image en niveaux de gris
# image = cv2.imread('images/hair_remy.jpg', cv2.IMREAD_GRAYSCALE)

# # Paramètre de correction gamma
# gamma = 1.5

# # Appliquer la correction gamma
# gamma_corrected = np.power(image / 255.0, gamma)
# gamma_corrected = np.uint8(gamma_corrected * 255)

# # Afficher l'image originale et l'image avec correction gamma
# cv2.imshow('Image originale', image)
# cv2.imshow('Image avec correction gamma', gamma_corrected)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#--------------------------------------------------------------------------------------------

# Exo complet

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
image = cv2.imread('images/img_sombre.jpeg', cv2.IMREAD_GRAYSCALE)
#image = cv2.imread('../tp_2/images_TP1-2/1_wIXlvBeAFtNVgJd49VObgQ.png', cv2.IMREAD_GRAYSCALE)

# Paramètres pour la transformation linéaire
alpha_linear = 1.5   # Facteur d'échelle pour la transformation linéaire
beta_linear = 30     # Décalage pour la transformation linéaire

# Paramètre pour la correction gamma
gamma = 1.5   # Paramètre gamma pour la correction gamma

# Appliquer l'égalisation de l'histogramme avec OpenCV
equalized_image_opencv = cv2.equalizeHist(image)

# Appliquer la transformation linéaire
linear_transformed_image = np.clip(alpha_linear * image + beta_linear, 0, 255).astype(np.uint8)

# Appliquer la correction gamma
gamma_corrected = np.power(image / 255.0, gamma)
gamma_corrected = np.uint8(gamma_corrected * 255)

# Afficher les résultats et leurs histogrammes
images = [image, equalized_image_opencv, linear_transformed_image, gamma_corrected]
titles = ['Image originale', 'Égalisation de l\'histogramme (OpenCV)', 'Transformation linéaire', 'Correction gamma']
params = [None, None, (alpha_linear, beta_linear), gamma]

# Boucle pour afficher les images et les histogrammes
for i in range(len(images)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])   # Titre de l'image
    plt.axis('off')   # Désactiver les axes

    # Affichage de l'histogramme
    plt.subplot(2, 4, i + 5)
    plt.hist(images[i].ravel(), bins=256, range=[0, 256])
    plt.title('Histogramme ' + titles[i])   # Titre de l'histogramme
    plt.xlim(0, 256)   # Limites de l'axe x de l'histogramme

    # Affichage des paramètres si disponibles
    if params[i] is not None:
        plt.text(10, 100, f'Paramètres: {params[i]}', color='white')   # Afficher les paramètres

# Ajustement de la disposition des sous-graphiques
plt.tight_layout()
plt.show()   # Afficher les graphiques
