import os
import json

#圖片檔名必須能轉換為整數
#image_id必須為檔名的整數型態
#分離後的json檔的info、licenses、categories皆與原來相同
#分離後的json檔將被存在save_path資料夾內的train.json、val.json、test.json
#分離後的json檔的encoding為utf-8，可自行更改
def separate_coco(train_img_path, val_img_path, test_img_path, json_path, save_path, ext='.png'):
    train_img_list = []
    val_img_list = []
    test_img_list = []

    for img in os.listdir(train_img_path):
        if img.endswith(ext):
            train_img_list.append(int(os.path.splitext(os.path.basename(img))[0]))

    for img in os.listdir(val_img_path):
        if img.endswith(ext):
            val_img_list.append(int(os.path.splitext(os.path.basename(img))[0]))

    for img in os.listdir(test_img_path):
        if img.endswith(ext):
            test_img_list.append(int(os.path.splitext(os.path.basename(img))[0]))


    with open(json_path, encoding='utf-8') as json_f:
        coco_all_json = json.load(json_f, )
        coco_all_json_info = coco_all_json['info']
        coco_all_json_licenses = coco_all_json['licenses']
        coco_all_json_categories = coco_all_json['categories']
        coco_all_json_imgs = coco_all_json['images']
        coco_all_json_annotations = coco_all_json['annotations']

        coco_train_json = {}
        coco_val_json = {}
        coco_test_json = {}

        coco_train_json_imgs = []
        coco_val_json_imgs = []
        coco_test_json_imgs = []

        coco_train_json_annotations = []
        coco_val_json_annotations = []
        coco_test_json_annotations = []

        for coco_img in coco_all_json_imgs:
            if coco_img["id"] in train_img_list:
                coco_train_json_imgs.append(coco_img)
            elif coco_img["id"] in val_img_list:
                coco_val_json_imgs.append(coco_img)
            elif coco_img["id"] in test_img_list:
                coco_test_json_imgs.append(coco_img)
            else:
                print("image id went wrong on coco_img part!!")

        for coco_annotation in coco_all_json_annotations:
            if coco_annotation["image_id"] in train_img_list:
                coco_train_json_annotations.append(coco_annotation)
            elif coco_annotation["image_id"] in val_img_list:
                coco_val_json_annotations.append(coco_annotation)
            elif coco_annotation["image_id"] in test_img_list:
                coco_test_json_annotations.append(coco_annotation)
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

        coco_test_json['info'] = coco_all_json_info
        coco_test_json['licenses'] = coco_all_json_licenses
        coco_test_json['categories'] = coco_all_json_categories
        coco_test_json['images'] = coco_test_json_imgs
        coco_test_json['annotations'] = coco_test_json_annotations

        coco_train_json_save_path = os.path.join(save_path, './train.json')
        coco_val_json_save_path = os.path.join(save_path, './val.json')
        coco_test_json_save_path = os.path.join(save_path, './test.json')

        with open(coco_train_json_save_path, 'w', encoding='utf-8') as f:
            json.dump(coco_train_json, f, ensure_ascii=False, indent=4)

        with open(coco_val_json_save_path, 'w', encoding='utf-8') as f:
            json.dump(coco_val_json, f, ensure_ascii=False, indent=4)

        with open(coco_test_json_save_path, 'w', encoding='utf-8') as f:
            json.dump(coco_test_json, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    train_img_path = os.path.join(os.getcwd(), './ChestX_train')
    val_img_path = os.path.join(os.getcwd(), './ChestX_val')
    test_img_path = os.path.join(os.getcwd(), './ChestX_test')

    json_path = os.path.join(os.getcwd(), './ChestX_relabling/ChestX_relabling.json')

    save_path = os.path.join(os.getcwd(), './ChestX_relabling/')

    separate_coco(train_img_path, val_img_path, test_img_path, json_path, save_path, ext='.png')