import os
import cv2


def separateImageUsingSize(image_dir, save_dir, separate_size=(608, 608), separate_rate=2):
    image_filenames = os.listdir(image_dir)
    image_paths = []
    size_before_divide = (separate_size[0] * separate_rate, separate_size[1] * separate_rate)
    for image_filename in image_filenames:
        image_paths.append(os.path.join(image_dir, image_filename))

    for image_path in image_paths:
        original_image = cv2.imread(image_path)
        resized_image = cv2.resize(original_image, size_before_divide, interpolation=cv2.INTER_AREA)
        for height_part in range(separate_rate):
            for width_part in range(separate_rate):
                top_border = separate_size[1] * height_part
                bottom_border = separate_size[1] * (height_part + 1)
                left_border = separate_size[0] * width_part
                right_border = separate_size[0] * (width_part + 1)
                separated_image = resized_image[top_border:bottom_border, left_border:right_border]
                separated_image_splitext = os.path.splitext(os.path.basename(image_path))
                separated_image_save_path = os.path.join(save_dir, f"{separated_image_splitext[0]}_{height_part}{width_part}{separated_image_splitext[1]}")
                cv2.imwrite(separated_image_save_path, separated_image)
                print(f"Saved separated image in {separated_image_save_path}")


def main():
    image_dir = os.path.join(os.getcwd(), '../Datasets/histo_equalization/2688/images/train')
    save_dir = os.path.join(os.getcwd(), '../Datasets/2688_plus_ChestX_relabling_histo_separated/images/train')
    separate_size = (608, 608)
    separate_rate = 4
    separateImageUsingSize(image_dir, save_dir, separate_size, separate_rate)



if __name__ == '__main__':
    main()