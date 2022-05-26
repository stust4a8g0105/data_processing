import os
import cv2
import json
import numpy as np
from torchvision import transforms

def dataAugmentation_contrast_brightness(image_path, json_path, image_savePath, json_savePath):
    colorJitterTransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.3, saturation=100, hue=0.2, contrast=0.3)
    ])
    image_filename_list = os.listdir(image_path)
    image_filename_list = list(map(lambda image_filename: os.path.join(image_path, image_filename), image_filename_list))
    result_annotation_id_counter = 0


    for i in range(2):
        for image_filename in image_filename_list:
            image = cv2.imread(image_filename)
            transformed_image = colorJitterTransform(image).permute(1, 2, 0)
            transformed_image = (transformed_image.cpu().detach().numpy() * 255).astype(int)
            image_file_basename = os.path.basename(image_filename)
            cv2.imwrite(os.path.join(image_savePath, f'./{1}{i}{image_file_basename}'), transformed_image)
            print(f"saved image into {os.path.join(image_savePath, f'{1}{i}{image_file_basename}')}\n")
            transformed_image = (np.vectorize(lambda p: 255 - p))(transformed_image)
            cv2.imwrite(os.path.join(image_savePath, f"./{2}{i}{image_file_basename}"), transformed_image)
            print(f"saved image into {os.path.join(image_savePath, f'{2}{i}{image_file_basename}')}\n")


    with open(json_path, "r", encoding='utf-8') as json_f:
        original_json_dict = json.load(json_f)
        result_json_dict = original_json_dict.copy()
        original_images = original_json_dict["images"]
        result_images = []
        original_annotations = original_json_dict["annotations"]
        result_annotations = []

        #對images擴增
        for i in range(2):
            for j in range(1, 3):
                for original_image in original_images:
                    result_image = original_image.copy()
                    filename = original_image["file_name"]
                    result_filename = f"{j}{i}{filename}"
                    result_image["file_name"] = result_filename
                    result_image["id"] = int(os.path.splitext(result_filename)[0])
                    result_images.append(result_image)

        #對annotations擴增
        for i in range(2):
            for j in range(1, 3):
                for original_annotation in original_annotations:
                    result_annotation = original_annotation.copy()
                    result_annotation["id"] = result_annotation_id_counter
                    result_annotation_id_counter += 1
                    original_image_id = original_annotation["image_id"]
                    result_annotation["image_id"] = int(f"{j}{i}{original_image_id}")
                    result_annotations.append(result_annotation)
        result_json_dict["images"] = result_images
        result_json_dict["annotations"] = result_annotations

        with open(json_savePath, 'w', encoding='utf-8') as f:
            json.dump(result_json_dict, f, ensure_ascii=False, indent=4)
            print(f"saved portion {json_savePath}")




if __name__ == '__main__':
    # push
    image_path = os.path.join(os.getcwd(), "../Datasets/K_Fold/contrast_brightness_24times_before_augmentation/4")
    json_path = os.path.join(os.getcwd(), "../Datasets/K_Fold/contrast_brightness_24times_before_augmentation/4.json")
    image_savePath = os.path.join(os.getcwd(), "../Datasets/K_Fold/contrast_brightness_24times_augmentation/4")
    json_savePath = os.path.join(os.getcwd(), "../Datasets/K_Fold/contrast_brightness_24times_augmentation/4.json")
    dataAugmentation_contrast_brightness(image_path, json_path, image_savePath, json_savePath)