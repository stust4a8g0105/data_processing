from skimage.draw import polygon
import cv2
import json
import os
import numpy as np

def convertCocoSegmentToMask(coco_path, mask_save_path):
    with open(coco_path, 'r', encoding='utf-8') as coco_f:
        coco_dict = json.load(coco_f)
        coco_images = coco_dict['images']
        coco_image_cache = {}

        for coco_image in coco_images:
            coco_image_cache[f"{coco_image['id']}"] = {'width': coco_image['width'], 'height': coco_image['height'], 'file_name': coco_image['file_name']}

            mask_image = np.zeros((coco_image['height'], coco_image['width']))
            coco_mask_path = os.path.join(mask_save_path, coco_image["file_name"])
            cv2.imwrite(coco_mask_path, mask_image)

        coco_annotations = coco_dict['annotations']
        for coco_annotation in coco_annotations:
            coco_filename = coco_image_cache[f"{coco_annotation['image_id']}"]["file_name"]
            coco_mask_path = os.path.join(mask_save_path, coco_filename)
            coco_segmentation = coco_annotation['segmentation'][0]
            coco_image_shape = (coco_image_cache[f"{coco_annotation['image_id']}"]['height'], coco_image_cache[f"{coco_annotation['image_id']}"]['width'])
            coco_polygon = np.array(coco_segmentation).reshape((-1, 2))
            coco_polygon = coco_polygon.transpose() # shape: (2, n)
            print(f"processing {coco_filename}")
            row = coco_polygon[1]
            column = coco_polygon[0]
            rr, cc = polygon(row, column)

            # 若超出rr或cc超出範圍則設成 shape - 1
            rr[rr >= coco_image_shape[0]] = coco_image_shape[0] - 1
            cc[cc >= coco_image_shape[0]] = coco_image_shape[1] - 1
            mask_image = cv2.imread(coco_mask_path)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            mask_image[rr, cc] = 255
            cv2.imwrite(coco_mask_path, mask_image)

def main():
    coco_path = os.path.join(os.getcwd(), '../Datasets/K_Fold/2688_plus_ChestX_histo_augmentation/1.json')
    mask_save_path = os.path.join(os.getcwd(), '../Datasets/K_Fold/2688_plus_ChestX_histo_augmentation/mask/1')
    convertCocoSegmentToMask(coco_path, mask_save_path)

if __name__ == '__main__':
    main()

