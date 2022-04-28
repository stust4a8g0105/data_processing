import os
import json

#圖片檔名必須能轉換為整數
#image_id必須為檔名的整數型態
#分離後的json檔的info、licenses、categories皆與原來相同
#分離後的json檔將被存在save_path資料夾內的train.json、val.json、test.json
#分離後的json檔的encoding為utf-8，可自行更改
def separate_coco(train_img_path, val_img_path, json_path, save_path, ext='.jpg'):
    train_image_filenames = []
    val_image_filenames = []
    train_image_ids = []
    val_image_ids = []
    for img in os.listdir(train_img_path):
        train_image_filenames.append(img)

    for img in os.listdir(val_img_path):
        val_image_filenames.append(img)

    with open(json_path, encoding='utf-8') as json_f:
        coco_all_json = json.load(json_f, )
        coco_all_json_info = coco_all_json['info']
        coco_all_json_licenses = coco_all_json['licenses']
        coco_all_json_categories = coco_all_json['categories']
        coco_all_json_imgs = coco_all_json['images']
        coco_all_json_annotations = coco_all_json['annotations']

        coco_train_json = {}
        coco_val_json = {}

        coco_train_json_imgs = []
        coco_val_json_imgs = []

        coco_train_json_annotations = []
        coco_val_json_annotations = []

        for coco_img in coco_all_json_imgs:
            if coco_img["file_name"] in train_image_filenames:
                coco_train_json_imgs.append(coco_img)
                train_image_ids.append(coco_img["id"])
            elif coco_img["file_name"] in val_image_filenames:
                coco_val_json_imgs.append(coco_img)
                val_image_ids.append(coco_img["id"])
            else:
                print("image id went wrong on coco_img part!!")

        for coco_annotation in coco_all_json_annotations:
            if coco_annotation["image_id"] in train_image_ids:
                coco_train_json_annotations.append(coco_annotation)
            elif coco_annotation["image_id"] in val_image_ids:
                coco_val_json_annotations.append(coco_annotation)
            else:
                print("annotation image id went wrong on coco_annotation part!!")

        coco_train_json['info'] = coco_all_json_info
        coco_train_json['licenses'] = coco_all_json_licenses
        coco_train_json['categories'] = coco_all_json_categories
        coco_train_json['images'] = coco_train_json_imgs
        coco_train_json['annotations'] = coco_train_json_annotations

        coco_val_json['info'] = coco_all_json_info
        coco_val_json['licenses'] = coco_all_json_licenses
        coco_val_json['categories'] = coco_all_json_categories
        coco_val_json['images'] = coco_val_json_imgs
        coco_val_json['annotations'] = coco_val_json_annotations

        coco_train_json_save_path = os.path.join(save_path, './train.json')
        coco_val_json_save_path = os.path.join(save_path, './val.json')

        with open(coco_train_json_save_path, 'w', encoding='utf-8') as f:
            json.dump(coco_train_json, f, ensure_ascii=False, indent=4)

        with open(coco_val_json_save_path, 'w', encoding='utf-8') as f:
            json.dump(coco_val_json, f, ensure_ascii=False, indent=4)




if __name__ == '__main__':
    train_img_path = os.path.join(os.getcwd(), "../Dataset/images/train")
    val_img_path = os.path.join(os.getcwd(), "../Dataset/images/val")
    json_path = os.path.join(os.getcwd(), "../Dataset/label/orchid_all_annotation.json")
    save_path = os.path.join(os.getcwd(), "../Dataset/label/seperated_labels")
    separate_coco(train_img_path, val_img_path, json_path, save_path, ext='.jpg')