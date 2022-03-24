import os
import json

def changeCocoCategoryId(original_coco_path, save_path, new_id=1):
    with open(original_coco_path, 'r', encoding="utf-8") as coco_f:
        coco_dict = json.load(coco_f)
        processed_coco_dict = coco_dict.copy()

        original_categories = coco_dict['categories']
        processed_categories = original_categories.copy()
        processed_categories[0]['id'] = new_id

        original_annotations = coco_dict['annotations']
        processed_annotations = []
        for original_annotation in original_annotations:
            processed_annotation = original_annotation.copy()
            processed_annotation['category_id'] = new_id
            processed_annotations.append(processed_annotation)
            print(f"Change category_id to {new_id}")

        processed_coco_dict['categories'] = processed_categories
        processed_coco_dict['annotations'] = processed_annotations

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(processed_coco_dict, f, ensure_ascii=False, indent=4)
        print(f"Result saved to {save_path}")

def main():
    original_coco_path = os.path.join(os.getcwd(), '../fracture_Yet-Another-EfficientDet-Pytorch/datasets/2688plusChestX/annotations/instances_val.json')
    save_path = os.path.join(os.getcwd(), '../fracture_Yet-Another-EfficientDet-Pytorch/datasets/2688plusChestX/annotations/instances_val_changed.json')
    changeCocoCategoryId(original_coco_path, save_path)


if __name__ == '__main__':
    main()