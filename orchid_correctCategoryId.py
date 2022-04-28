import json
import os

def correctCategoryId(coco_dict, filename_label_pairs, category_name, id_from, id_to):
    id_filename_mapper = {}
    filename_category_mapper = {}
    corrected_coco_dict = coco_dict.copy()
    original_annotations = coco_dict['annotations']
    corrected_annotations = []
    corrected_categories = []

    
    #填滿id和file_name的映射
    for image in coco_dict["images"]:
        id = image['id']
        id_filename_mapper[f"{id}"] = image['file_name']

    #填滿file_name和category的映射
    for filename_label_pair in filename_label_pairs:
        filename_category_mapper[filename_label_pair[0]] = int(filename_label_pair[1])


    # 將annotation的categoryId修正
    for original_annotation in original_annotations:
        filename = id_filename_mapper[f"{original_annotation['image_id']}"]
        correct_category_id = filename_category_mapper[filename]
        original_annotation['category_id'] = correct_category_id
        corrected_annotations.append(original_annotation)
        print(f"修正{filename} {original_annotation['id']}的categories_id: {correct_category_id}")

    # 將categories部分修正
    for i in range(id_from, id_to + 1):
        corrected_categories.append({
            "supercategory": category_name,
            "name": "",
            "id": i
        })

    # 將修改後的結果填入corrected_coco_dict
    corrected_coco_dict["annotations"] = corrected_annotations
    corrected_coco_dict['categories'] = corrected_categories
    return corrected_coco_dict


def orchid_correctCategoryId(coco_annotation_path, label_txt_path, coco_save_path):
    with open(coco_annotation_path, 'r', encoding="utf-8") as coco_file, open(label_txt_path, 'r') as label_file:
        coco_dict = json.load(coco_file)
        label_list = label_file.readlines()
        filename_label_pairs = list(map(lambda l: l.split("\t"), label_list))
        corrected_coco_dict = correctCategoryId(coco_dict, filename_label_pairs, 'orchid', 0, 218)
        with open(coco_save_path, 'w', encoding='utf-8') as save_file:
            json.dump(corrected_coco_dict, save_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    coco_annotation_path = os.path.join(os.getcwd(), '../Dataset/label/mergin_processing/whole.json')
    label_path = os.path.join(os.getcwd(), '../Dataset/label/filenames.txt')
    coco_save_path = os.path.join(os.getcwd(), '../Dataset/label/mergin_processing/whole_correct.json')
    orchid_correctCategoryId(coco_annotation_path, label_path, coco_save_path)
