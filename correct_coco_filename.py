import json
import os

def correct_coco_filename(json_path, save_path):
    with open(json_path, "r", encoding="utf-8") as json_f:
        json_dict = json.load(json_f)
        corrected_json = json_dict.copy()
        original_images = json_dict["images"]
        corrected_images = []
        for original_image in original_images:
            image_filename = original_image["file_name"]
            if image_filename.endswith(".jpg"):
                image_filename = f"{os.path.splitext(image_filename)[0]}.png"
                original_image["file_name"] = image_filename
            corrected_images.append(original_image)
        corrected_json["images"] = corrected_images
        with open(save_path, "w", encoding="utf-8") as save_f:
            json.dump(corrected_json, save_f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    json_path = os.path.join(os.getcwd(), '../Datasets/whole_without_augmentation.json')
    save_path = os.path.join(os.getcwd(), '../Datasets/whole_without_augmentation_corrected.json')
    correct_coco_filename(json_path, save_path)