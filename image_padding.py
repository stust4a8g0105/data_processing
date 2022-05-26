import cv2
import numpy as np
import os

def image_padding(images_path, save_path, new_size=2688, color=(0, 0, 0)):
    image_filename_list = os.listdir(images_path)
    image_filename_list = map(lambda filename: os.path.join(images_path, filename), image_filename_list)
    for image_filename in image_filename_list:
        # read image
        img = cv2.imread(image_filename)
        old_image_height, old_image_width, channels = img.shape

        result = np.full((new_size, new_size, channels), color, dtype=np.uint8)

        # compute center offset
        x_center = (new_size - old_image_width) // 2
        y_center = (new_size - old_image_height) // 2

        # copy img image into center of result image
        result[y_center:y_center + old_image_height,
        x_center:x_center + old_image_width] = img

        image_save_path = os.path.join(save_path, os.path.basename(image_filename))
        # save result
        cv2.imwrite(image_save_path, result)
        print(f"added padding and saved to {image_save_path}")

if __name__ == '__main__':
    images_path = os.path.join(os.getcwd(), "../Datasets/chimei_all_91")
    save_path = os.path.join(os.getcwd(), "../Datasets/chimei_all_91_padding_2688")
    image_padding(images_path, save_path)