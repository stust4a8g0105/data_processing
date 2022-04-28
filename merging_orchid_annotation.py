import os
import json

def merge_coco(coco_path1, coco_path2, save_path):
    id_filename_mapper = {}
    filename_category_mapper = {}
    image_oldID_newID_mapper1 = {}
    image_oldID_newID_mapper2 = {}
    category_oldID_newID_mapper1 = {}
    category_oldID_newID_mapper2 = {}
    coco_image_id_counter = 1
    coco_annotation_id_counter = 1
    image_width = 480
    image_height = 640
    with open(coco_path1, encoding="utf-8") as coco_f1, open(coco_path2, encoding="utf-8") as coco_f2:
        # load json
        result_json = {}
        coco_json1 = json.load(coco_f1)
        coco_json2 = json.load(coco_f2)

        result_json['info'] = coco_json1['info']
        result_json['licenses'] = coco_json1['licenses']
        categories = coco_json1['categories']
        result_json['categories'] = categories

        coco_image1 = coco_json1["images"]
        coco_image2 = coco_json2["images"]

        images = []
        annotations = []

        # merging images part
        for image in coco_image1:
            image_oldID_newID_mapper1[f"{image['id']}"] = coco_image_id_counter
            image["id"] = coco_image_id_counter
            image["width"] = image_width
            image["height"] = image_height
            id_filename_mapper[f"{coco_image_id_counter}"] = image["file_name"]
            images.append(image)
            coco_image_id_counter += 1
        for image in coco_image2:
            image_oldID_newID_mapper2[f"{image['id']}"] = coco_image_id_counter
            image["id"] = coco_image_id_counter
            image["width"] = image_width
            image["height"] = image_height
            id_filename_mapper[f"{coco_image_id_counter}"] = image["file_name"]
            images.append(image)
            coco_image_id_counter += 1


        for annotation in coco_json1["annotations"]:
            annotation["id"] = coco_annotation_id_counter
            annotation["image_id"] = image_oldID_newID_mapper1[f"{annotation['image_id']}"]
            coco_annotation_id_counter += 1
            annotations.append(annotation)

        for annotation in coco_json2["annotations"]:
            annotation["id"] = coco_annotation_id_counter
            annotation["image_id"] = image_oldID_newID_mapper2[f"{annotation['image_id']}"]
            coco_annotation_id_counter += 1
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
    coco_json1_path = os.path.join(os.getcwd(), '../Dataset/label/mergin_processing/549+1097.json')
    coco_json2_path = os.path.join(os.getcwd(), '../Dataset/label/mergin_processing/1644+2191.json')
    save_path = os.path.join(os.getcwd(), '../Dataset/label/mergin_processing/whole.json')
    merge_coco(coco_json1_path, coco_json2_path, save_path)
    #convertCategories(coco_json2_path, coco_json2_path)
