import os
import json

if __name__ == '__main__':
    json_path = os.path.join(os.getcwd(), '../TBrain_AI/Dataset/labels/val.json')
    json_save_path = os.path.join(os.getcwd(), '../TBrain_AI/Dataset/labels/val_new.json')

    with open(json_path, encoding='utf-8') as json_f:
        json_dict = json.load(json_f)
        json_images = json_dict['images']
        for json_image in json_images:
            json_image['width'] = 640
            json_image['height'] = 480
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
