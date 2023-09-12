import cv2
import numpy as np

def process_image(image_path: str, channel: int) -> np.ndarray:
    """
    Process the image by loading it, creating a copy, zeroing out the irrelevant channels, and resizing it.

    Args:
        image_path: The path to the image file.
        channel: The channel to keep in the image (0 for Blue, 1 for Green, 2 for Red).

    Returns:
        The resized image with the irrelevant channels zeroed out.
    """

    # Load the image
    img = cv2.imread(image_path)

    # Create a copy of the image
    img_copy = img.copy()

    # Zero out the irrelevant channels for the given channel
    img_copy[:, :, (channel+1) % 3] = 0  # Zero out the irrelevant channel
    img_copy[:, :, (channel+2) % 3] = 0  # Zero out the irrelevant channel

    # Resize the image to a width of 40 pixels while maintaining the aspect ratio
    new_width = 40
    aspect_ratio = img_copy.shape[1] / img_copy.shape[0]
    new_height = int(new_width / aspect_ratio)

    img_resized = cv2.resize(img_copy, (new_width, new_height))

    return img_resized

# Charger l'image
image_path = "images/hair_remy.jpg"

# Processus pour chaque canal
imgBlue = process_image(image_path, 0)
imgGreen = process_image(image_path, 1)
imgRed = process_image(image_path, 2)

# Afficher les images
cv2.imshow("Image en bleu (40px)", imgBlue)
cv2.imshow("Image en vert (40px)", imgGreen)
cv2.imshow("Image en rouge (40px)", imgRed)

cv2.waitKey(0)
cv2.destroyAllWindows()
