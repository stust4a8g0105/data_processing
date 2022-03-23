import cv2
import numpy as np
import os

def histo_equalization(image_path, save_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    H, W = image.shape
    total_pixels = H * W
    pixel_counts = np.zeros(256)
    transform_pixels = np.zeros(256)
    converted_image = image.copy()
    for x in range(W):
        for y in range(H):
            gray = image[y][x]
            pixel_counts[gray] += 1

    for p in range(256):
        if p == 0:
            transform_pixels[p] = 255 * (pixel_counts[p] / total_pixels)
        else:
            transform_pixels[p] = transform_pixels[p - 1] + 255 * (pixel_counts[p] / total_pixels)

    for x in range(W):
        for y in range(H):
            grayscale = image[y][x]
            converted_image[y][x] = transform_pixels[grayscale]

    converted_image = cv2.cvtColor(converted_image, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join(os.getcwd(), save_path), converted_image)
    print(f'saved result to {os.path.join(os.getcwd(), save_path)}')


def cv2_histogram_equalization(image_path, save_path):
    img = cv2.imread(image_path, 0)
    equ = cv2.equalizeHist(img)
    converted_image = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(save_path, converted_image)
    print(f"Processe and saved image {save_path}")

def cv2_histogram_equalization_comparison(image_path, save_path):
    img = cv2.imread(image_path, 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))  # stacking images side-by-side
    converted_image = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(save_path, converted_image)


def main():
    image_dir = os.path.join(os.getcwd(), '../Datasets/ChestX_test')
    save_dir = os.path.join(os.getcwd(), '../Datasets/2688_plus_ChestX_histo_without_augmentation/ChestX_test')

    image_paths = os.listdir(image_dir)
    for image_path in image_paths:
        image_filename = os.path.basename(image_path)
        cv2_histogram_equalization(os.path.join(os.getcwd(), image_dir, image_path), os.path.join(save_dir, image_filename))

    # image_paths = os.listdir(image_dir)
    # for image_path in image_paths:
    #     image_filename = os.path.basename(image_path)
    #     cv2_histogram_equalization(os.path.join(os.getcwd(), image_dir, image_path), os.path.join(save_dir, image_filename))

if __name__ == '__main__':
    main()