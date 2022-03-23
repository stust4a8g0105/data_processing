import os

def imageExtNegotiate(image_path):
    image_path_png = f"{os.path.splitext(image_path)[0]}.png"
    if os.path.exists(image_path_png):
        return image_path_png
    else:
        return f"{os.path.splitext(image_path)[0]}.jpg"