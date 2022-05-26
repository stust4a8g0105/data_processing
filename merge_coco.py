import os
import json

def merge_coco(coco_path1, coco_path2, save_path):
    with open(coco_path1, encoding="utf-8") as coco_f1, open(coco_path2, encoding="utf-8") as coco_f2:
        # load json
        result_json = {}
        coco_json1 = json.load(coco_f1)
        coco_json2 = json.load(coco_f2)

        result_json['info'] = coco_json1['info']
        result_json['licenses'] = coco_json1['licenses']
        categories = coco_json1['categories']
        categories[0]['id'] = 0
        result_json['categories'] = categories

        images = []
        annotations = []
        annotation_id = 0
        images.extend(coco_json1['images'])
        print('coco_json1_image_count: ', len(images))
        images.extend(coco_json2['images'])
        print('coco_json1+json2_image_count: ', len(images))
        for annotation in coco_json1["annotations"]:
            annotation["id"] = annotation_id
            annotation_id += 1
            annotations.append(annotation)

        for annotation in coco_json2["annotations"]:
            annotation["id"] = annotation_id
            annotation_id += 1
            annotations.append(annotation)

        result_json['images'] = images
        result_json['annotations'] = annotations

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)

def convertCategories(coco_path, save_path):
    with open(coco_path) as coco_f:

        coco_json = json.load(coco_f)
        result_json = coco_json.copy()
        categories = coco_json['categories']
        categories[0]['id'] = 0
        result_json['categories'] = categories

        annotations = coco_json["annotations"]
        for a in annotations:
            a['category_id'] = 0
        result_json['annotations'] = annotations

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    coco_json1_path = os.path.join(os.getcwd(), '../Datasets/K_Fold/24times_augmentation/01.json')
    coco_json2_path = os.path.join(os.getcwd(), '../Datasets/K_Fold/24times_augmentation/23.json')
    save_path = os.path.join(os.getcwd(), '../Datasets/K_Fold/24times_augmentation/0123.json')
    merge_coco(coco_json1_path, coco_json2_path, save_path)
    #convertCategories(coco_json2_path, coco_json2_path)