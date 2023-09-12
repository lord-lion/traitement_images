import cv2
import numpy as np
import matplotlib.pyplot as plt

# Définition des fonctions pour appliquer les transformations
def apply_linear_transformation(image, alpha, beta):
    """
    Applique une transformation linéaire à une image.

    Paramètres:
    image (numpy.ndarray): L'image d'entrée à transformer.
    alpha (float): Le facteur d'échelle pour la transformation.
    beta (float): Le facteur de décalage pour la transformation.

    Retour:
    numpy.ndarray: L'image transformée avec des valeurs limitées entre 0 et 255, et convertie en entier non signé sur 8 bits.
    """
    return np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

def apply_gamma_correction(image, gamma):
    """
    Applique une correction gamma à une image.
 
    Paramètres:
        image (numpy.ndarray): L'image d'entrée à corriger.
        gamma (float): La valeur gamma à appliquer pour la correction.

    Retour:
        numpy.ndarray: L'image corrigée gamma.

    """
    gamma_corrected = np.power(image / 255.0, gamma)
    return np.uint8(gamma_corrected * 255)

# Chemins vers différentes images en niveaux de gris
images = ['images/hair_remy.jpg', 'images/bateau_2.png', 'images/boat.png']

# Valeurs de paramètres pour la transformation linéaire
alpha_values = [0.5, 1.0, 1.5]
beta_values = [-50, 0, 50]

# Valeurs de paramètres pour la correction gamma
gamma_values = [0.5, 1.0, 1.5]

# Boucle pour traiter chaque image
for image_path in images:
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Créer une nouvelle figure pour chaque image
    plt.figure(figsize=(12, 8))
    
    # Afficher l'image originale
    plt.subplot(3, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image originale')
    plt.axis('off')
    
    # Afficher l'histogramme de l'image originale
    plt.subplot(3, 4, 2)
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title('Histogramme original')
    plt.xlim(0, 256)
    
    # Appliquer la transformation linéaire avec différentes valeurs de paramètres
    for i, (alpha, beta) in enumerate(zip(alpha_values, beta_values), start=3):
        linear_transformed = apply_linear_transformation(image, alpha, beta)
        
        # Afficher l'image transformée
        plt.subplot(3, 4, i)
        plt.imshow(linear_transformed, cmap='gray')
        plt.title(f'Transformation linéaire\nAlpha: {alpha}, Beta: {beta}')
        plt.axis('off')
        
        # Afficher l'histogramme de l'image transformée
        plt.subplot(3, 4, i+1)
        plt.hist(linear_transformed.ravel(), bins=256, range=[0, 256])
        plt.title(f'Histogramme transformation linéaire')
        plt.xlim(0, 256)
    
    # Appliquer la correction gamma avec différentes valeurs de paramètres
    for i, gamma in enumerate(gamma_values, start=9):
        gamma_corrected = apply_gamma_correction(image, gamma)
        
        # Afficher l'image corrigée gamma
        plt.subplot(3, 4, i)
        plt.imshow(gamma_corrected, cmap='gray')
        plt.title(f'Correction gamma\nGamma: {gamma}')
        plt.axis('off')
        
        # Afficher l'histogramme de l'image corrigée gamma
        plt.subplot(3, 4, i+1)
        plt.hist(gamma_corrected.ravel(), bins=256, range=[0, 256])
        plt.title(f'Histogramme correction gamma')
        plt.xlim(0, 256)
    
    # Ajustement de la disposition des sous-graphiques pour chaque image
    plt.tight_layout()
    
    # Afficher les graphiques
    plt.show()
