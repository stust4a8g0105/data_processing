import os
import cv2

def separateDataFor_K_Fold(whole_image_path, k_fold_save_path, K=5):
    image_filenames = os.listdir(whole_image_path)
    image_file_paths = list(map(lambda image_filename: os.path.join(whole_image_path, image_filename) , image_filenames))
    image_file_len = len(image_file_paths)
    K_image_portion = []
    for i in range(K):
        K_image_portion.append(image_file_paths[i:image_file_len:K])

    for i, K_image_paths in enumerate(K_image_portion):
        k_fold_corresponding_save_dir = os.path.join(k_fold_save_path, f"{i}")
        if not os.path.exists(k_fold_corresponding_save_dir):
            os.mkdir(k_fold_corresponding_save_dir)

        for image_path in K_image_paths:
            image = cv2.imread(image_path)
            image_save_path = os.path.join(k_fold_corresponding_save_dir, os.path.basename(image_path))
            cv2.imwrite(image_save_path, image)
            print(f"saved image into {image_save_path}")

def main():
    whole_image_path = os.path.join(os.getcwd(), '../Datasets/2688_plus_ChestX_histo_total_without_augmentation')
    k_fold_save_path = os.path.join(os.getcwd(), '../Datasets/K_Fold/2688_plus_ChestX_histo')
    K = 5
    separateDataFor_K_Fold(whole_image_path, k_fold_save_path, K)

if __name__ == '__main__':
    main()