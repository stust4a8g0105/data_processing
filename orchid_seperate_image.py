import cv2
import os
import random

# train:val (8:2)
def orchid_separate_image(image_path, label_path, save_path, train_rate=8, val_rate=2):
    train_save_path = os.path.join(save_path, "./train")
    val_save_path = os.path.join(save_path, "./val")

    label_filename_cache = {}
    filename_label_pairs = []

    train_image_filenames = []
    val_image_filenames = []


    #讀取label資料
    #將filename_label_pairs變成 [[filename, label]...]
    #將filename_label_pairs隨機排序
    with open(label_path, "r") as label_file:
        labels = label_file.readlines()
        filename_label_pairs = list(map(lambda p: p.split("\t"), labels))
        filename_label_pairs = list(map(lambda p: [p[0], int(p[1])], filename_label_pairs))
        random.shuffle(filename_label_pairs)

    # 將label_filename_cache變成{"id1": [filename1, filename2, ...] ...}
    for filename_label_pair in filename_label_pairs:
        filename = filename_label_pair[0]
        label = filename_label_pair[1]
        if f"{label}" not in label_filename_cache:
            label_filename_cache[f"{label}"] = []
        label_filename_cache[f"{label}"].append(filename)

    # 將資料分配到train_image_filenames和val_image_filenames
    for _, filenames in label_filename_cache.items():
        train_image_filenames.extend(filenames[:train_rate])
        val_image_filenames.extend(filenames[train_rate:])

    for train_image_filename in train_image_filenames:
        train_image_file_path = os.path.join(image_path, train_image_filename)
        train_image_save_path = os.path.join(train_save_path, train_image_filename)
        image = cv2.imread(train_image_file_path)
        cv2.imwrite(train_image_save_path, image)
        print(f"save image {train_image_filename} to {train_image_save_path}")

    for val_image_filename in val_image_filenames:
        val_image_file_path = os.path.join(image_path, val_image_filename)
        val_image_save_path = os.path.join(val_save_path, val_image_filename)
        image = cv2.imread(val_image_file_path)
        cv2.imwrite(val_image_save_path, image)
        print(f"save image {val_image_filename} to {val_image_save_path}")



if __name__ == "__main__":
    image_path = os.path.join(os.getcwd(), "../Dataset/images/original_images")
    label_path = os.path.join(os.getcwd(), "../Dataset/label/filenames.txt")
    save_path = os.path.join(os.getcwd(), "../Dataset/images")
    orchid_separate_image(image_path, label_path, save_path)