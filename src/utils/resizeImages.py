import os
import cv2

def resizeImages(original_image_dir, resized_image_dir, new_size=(1024, 1024)):
    image_filenames = os.listdir(original_image_dir)
    for image_filename in image_filenames:
        image_path = os.path.join(original_image_dir, image_filename)
        print(f"processing {image_path}")
        save_path = os.path.join(resized_image_dir, image_filename)
        original_image = cv2.imread(image_path)
        resized_image = cv2.resize(original_image, new_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path, resized_image)
        print(f"resized {image_filename} and saved to {save_path}")


def main():
    original_image_dir = os.path.join(os.getcwd(), '../Datasets/2688_plus_ChestX_relabling_histo/images/ChestX_test')
    resized_image_dir = os.path.join(os.getcwd(), '../fracture_darknet_yolov4/build/darknet/x64/data/fracture/2688_plus_ChestX_relabling_histo/ChestX_test_resized')
    new_size = (608, 608)
    resizeImages(original_image_dir, resized_image_dir, new_size)

if __name__ == "__main__":
    main()