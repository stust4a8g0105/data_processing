import os
import math
import cv2
from coco_data_augmentation import spinImage

def image_augmentation(image_path, save_path, styles):
    image_file_names = []
    for image_path_name in os.listdir(image_path):
        image_file_names.append(os.path.basename(image_path_name))

    for style in styles:
        for file_name in image_file_names:
            angle = style[0]
            flip = style[1]

            image = cv2.imread(os.path.join(image_path, file_name))
            augmented_image = spinImage(image, angle, keep_size=True)

            if flip:
                augmented_image = cv2.flip(augmented_image, 1)

            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)

            if angle < 0:
                angle_name = f"1{-angle}"
            else:
                angle_name = f"{angle}"

            img_file_name = f"1{int(flip)}{angle_name}{file_name}"
            cv2.imwrite(os.path.join(save_path, img_file_name), augmented_image)
            print(f'{img_file_name} saved')


if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), './unet_original/labels/ChestX_test')
    save_path = os.path.join(os.getcwd(), './unet_augmented/labels/ChestX_test')
    styles = [[0, False], [5, False], [10, False], [-5, False], [-10, False], [0, True], [5, True], [10, True],
              [-5, True], [-10, True]]
    image_augmentation(image_path, save_path, styles)