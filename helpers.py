import os
from PIL import Image


def get_image_files(directory="data/pdf_resumes/CV_Images_EN"):
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith(image_extensions)
    ]

def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    # Calculate the scaling factor
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    scaled_image = image.resize((new_width, new_height))

    return scaled_image