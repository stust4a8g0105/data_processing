import os
import json
import numpy as np

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def resizeCocoAnnotations(original_coco_path, saved_coco_path, new_size):
    with open(original_coco_path, encoding='utf-8') as original_coco_f:
        original_coco_dict = json.load(original_coco_f)
        original_coco_images = original_coco_dict['images']
        original_coco_annotations = original_coco_dict['annotations']

        original_size_cache = {}

        resized_coco_dict = original_coco_dict.copy()
        resized_coco_images = []
        resized_coco_annotations = []

        for original_coco_image in original_coco_images:
            original_size_cache[f"{original_coco_image['id']}"] = original_coco_image['width']
            resized_coco_image = original_coco_image.copy()
            resized_coco_image['width'] = new_size
            resized_coco_image['height'] = new_size
            resized_coco_images.append(resized_coco_image)

        for original_coco_annotation in original_coco_annotations:
            original_width = original_size_cache[f"{original_coco_annotation['image_id']}"]
            # calc resize_scale
            resize_scale = new_size / original_width

            # resize segmentation
            resized_coco_annotation = original_coco_annotation.copy()
            resized_segmentation = resized_coco_annotation['segmentation'][0]
            resized_segmentation = list(map(lambda seg: int(seg * resize_scale), resized_segmentation))
            resized_coco_annotation['segmentation'][0] = resized_segmentation

            #resize area
            polygon = np.array(resized_segmentation).reshape((-1, 2))
            polygon_t = np.transpose(polygon)
            resized_area = int(PolyArea(polygon_t[0], polygon_t[1]))
            resized_coco_annotation['area'] = resized_area

            #resize bbox
            resized_coco_bbox = resized_coco_annotation['bbox']
            resized_coco_bbox = list(map(lambda bbox_point: int(bbox_point * resize_scale), resized_coco_bbox))
            resized_coco_annotation['bbox'] = resized_coco_bbox

            #append to resized_coco_annotations
            resized_coco_annotations.append(resized_coco_annotation)

        resized_coco_dict['images'] = resized_coco_images
        resized_coco_dict['annotations'] = resized_coco_annotations

        with open(saved_coco_path, 'w', encoding='utf-8') as resized_f:
            json.dump(resized_coco_dict, resized_f, ensure_ascii=False, indent=4)



def main():
    original_coco_path = os.path.join(os.getcwd(), '../fracture_Yet-Another-EfficientDet-Pytorch/datasets/2688plusChestX_histo/annotations/instances_val.json')
    saved_coco_path = os.path.join(os.getcwd(), '../fracture_Yet-Another-EfficientDet-Pytorch/datasets/2688plusChestX_histo/annotations/instances_val_resized.json')
    new_size = 896
    resizeCocoAnnotations(original_coco_path, saved_coco_path, new_size)

if __name__ == '__main__':
    main()
