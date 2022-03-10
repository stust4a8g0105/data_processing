import os
import json
import cv2

def convert_coco2yolo(image_path, json_path, save_path):
    image_path_names = os.listdir(image_path)
    image_file_names = []
    image_id_filename_mapper = {}
    image_shape_cache = {}  #id: {'width': W, 'height': H}
    for image_path_name in image_path_names:
        image_file_names.append(os.path.basename(image_path_name))

    with open(json_path, encoding='utf-8') as json_f:
        coco_json = json.load(json_f)
        coco_images = coco_json['images']
        coco_annotations = coco_json['annotations']
        for coco_image in coco_images:
            file_name = coco_image["file_name"]
            id = coco_image["id"]
            image_id_filename_mapper[f"{id}"] = file_name
            label_path = os.path.join(save_path, f'{os.path.splitext(os.path.basename(file_name))[0]}.txt')
            image = cv2.imread(os.path.join(image_path, file_name))
            H, W, _ = image.shape
            image_shape_cache[f"{id}"] = {
                'width': W,
                'height': H
            }
            with open(label_path, 'a') as label_file:
                print(f'Create {label_path} label file')


        for coco_annotation in coco_annotations:
            image_id = coco_annotation['image_id']
            category_id = coco_annotation['category_id']
            bbox = coco_annotation['bbox']
            W = image_shape_cache[f'{image_id}']['width']
            H = image_shape_cache[f'{image_id}']['height']

            lt_x = bbox[0]
            lt_y = bbox[1]
            bbox_w = bbox[2]
            bbox_h = bbox[3]

            yolo_cx = (lt_x + bbox_w / 2) / W
            yolo_cy = (lt_y + bbox_h / 2) / H
            yolo_bbox_width = bbox_w / W
            yolo_bbox_height = bbox_h / H

            image_file_name = image_id_filename_mapper[f'{image_id}']
            label_path = os.path.join(save_path, f'{os.path.splitext(os.path.basename(image_file_name))[0]}.txt')
            with open(label_path, 'a') as label_f:
                content = f"{category_id} {yolo_cx} {yolo_cy} {yolo_bbox_width} {yolo_bbox_height} \n"
                label_f.write(content)
                print(f"update {label_path}: ", content)

if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), './2688_plus_ChestX_relabling_augmented/images/ChestX_test')
    json_path = os.path.join(os.getcwd(), './2688_plus_ChestX_relabling_augmented/ChestX_test.json')
    save_path = os.path.join(os.getcwd(), './2688_plus_ChestX_relabling_augmented/labels/ChestX_test')
    convert_coco2yolo(image_path, json_path, save_path)