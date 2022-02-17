import cv2
import math
import json
import os
import numpy as np
from separateData import separateData
from datetime import datetime
from Data_augmentation import Data_augmentation

JSON_PATH = './ChestX_Det_train.json'
# JSON_PATH = 'resource/test.json'
IMAGE_PATH = './train'
# IMAGE_PATH = 'resource/test_data'

# BLACK_LIST = ["39120.png", "39449.png", "39458.png", "39459.png", "41944.png", "44861.png", "45193.png", "45195.png",
#               "45488.png", "45784.png", "45787.png", "457887.png", "45819.png", "45824.png", "45828.png", "45846.png",
#               "45847.png", "45848.png", "45850.png", "45851.png", "45852.png", "46030.png", "46652.png", "46940.png",
#               "46982.png", "47010.png", "47012.png", "47013.png", "47014.png", "47017.png", "47020.png", "47021.png",
#               "47196.png", "47263.png", "48908.png", "56590.png", "56687.png", "56688.png", "56689.png", "56700.png",
#               "567880.png", "57467.png", "57741.png", "57876.png", "57881.png", "57882.png", "57883.png", "57891.png",
#               "57895.png", "59227.png", "59442.png", "40050.png", "44864.png", "44865.png", "48872.png", "55840.png",
#               "62061.png", "69916.png", "59623.png", "61912.png", "69154.png", "69156.png", "69491.png", "61692.png",
#               "68882.png", "68883.png", "68884.png"]

# TRAIN = ['36237.png', '36285.png', '36338.png', '36342.png', '36343.png', '36345.png', '39402.png', '39405.png',
#          '39406.png', '39411.png', '39413.png', '39416.png', '39505.png', '39508.png', '39512.png', '39794.png',
#          '39803.png', '40006.png', '40019.png', '40410.png', '40415.png', '40417.png', '40419.png', '40453.png',
#          '40845.png', '40846.png', '40848.png', '41097.png', '41098.png', '41129.png', '41498.png', '41520.png',
#          '41802.png', '41820.png', '41823.png', '41959.png', '41960.png', '48925.png', '48942.png', '48943.png',
#          '45268.png', '45269.png', '45271.png', '45272.png', '45275.png', '45276.png', '45279.png', '45281.png',
#          '45477.png', '45478.png', '45479.png', '45493.png', '45496.png', '45497.png', '45498.png', '45523.png',
#          '45530.png', '45594.png', '45605.png', '45607.png', '45672.png', '45673.png', '45674.png', '45717.png',
#          '45785.png', '45786.png', '45788.png', '45789.png', '45791.png', '45818.png', '45821.png', '45823.png',
#          '45831.png', '45849.png', '45971.png', '45988.png', '45994.png', '46028.png', '46029.png', '46031.png',
#          '46032.png', '46035.png', '46036.png', '46037.png', '46038.png', '46039.png', '46040.png', '46042.png',
#          '46047.png', '46048.png', '46362.png', '46912.png', '46984.png', '46985.png', '47210.png', '48839.png',
#          '48844.png', '48845.png', '48873.png', '48883.png', '48884.png', '48887.png', '48889.png', '48890.png',
#          '48896.png', '55801.png', '55816.png', '55997.png', '56003.png', '56575.png', '56736.png', '56737.png',
#          '56740.png', '56749.png', '56776.png', '56784.png', '56787.png', '56788.png', '56793.png', '57550.png',
#          '57601.png', '57711.png', '57720.png', '57729.png', '57745.png', '57794.png', '57838.png', '57846.png',
#          '57850.png', '57852.png', '57873.png', '57878.png', '57900.png', '57901.png', '57903.png', '57905.png',
#          '57918.png', '57926.png', '57933.png', '57936.png', '57941.png', '57942.png', '57943.png', '57945.png',
#          '59055.png', '58879.png', '58881.png', '59059.png', '59361.png', '59422.png', '59424.png', '59425.png',
#          '59506.png', '59655.png', '59658.png', '59726.png', '60220.png', '60224.png', '60738.png', '60753.png',
#          '60758.png', '60759.png', '60761.png', '60763.png', '60768.png', '60779.png', '60780.png', '60902.png',
#          '60904.png', '60997.png', '61698.png', '61701.png', '61702.png', '61704.png', '61706.png', '61707.png',
#          '61708.png', '61709.png', '61992.png', '62039.png', '62043.png', '62049.png', '62050.png', '62069.png',
#          '62079.png', '67514.png', '67515.png', '67516.png', '67548.png', '68874.png', '68875.png', '68885.png',
#          '68886.png', '68888.png', '68889.png', '68890.png', '68892.png', '68914.png', '68917.png', '68923.png',
#          '68928.png', '68931.png', '68942.png', '68943.png', '68944.png', '68945.png', '68946.png', '68947.png',
#          '68948.png', '68996.png', '68997.png', '69024.png', '69127.png', '69133.png', '69136.png', '69377.png',
#          '69396.png', '69397.png', '69399.png', '69401.png', '69402.png', '69430.png', '69514.png', '69546.png',
#          '69547.png', '69548.png', '69549.png', '69553.png', '69559.png', '69579.png', '69581.png', '69582.png',
#          '69584.png', '69841.png', '69914.png', '69915.png', '69917.png', '69936.png', '69937.png', '69938.png',
#          '69939.png']

# VAL = ['69940.png', '69942.png', '70015.png', '70022.png', '70058.png', '70316.png', '70339.png', '70394.png',
#        '70397.png', '70417.png', '70424.png', '70449.png', '70460.png', '70488.png', '70920.png', '36365.png',
#        '39408.png', '39412.png', '39414.png', '39748.png', '39790.png', '39994.png', '40111.png', '40421.png',
#        '41128.png', '41165.png', '41518.png', '41798.png', '48932.png', '45208.png', '45495.png', '45529.png',
#        '45606.png', '45742.png', '45783.png', '45817.png', '45820.png', '45822.png', '45827.png', '45926.png',
#        '45969.png', '46041.png', '46938.png', '46986.png', '48868.png', '55908.png', '56582.png', '56779.png',
#        '56790.png', '57885.png', '57897.png', '57906.png', '59355.png', '59420.png', '59421.png', '59573.png',
#        '59576.png', '59665.png', '60752.png', '60755.png', '62055.png', '68891.png', '69139.png', '69400.png',
#        '69568.png', '69580.png', '70019.png', '70020.png', '70326.png', '39511.png', '39796.png', '40367.png',
#        '40412.png', '45816.png', '46939.png', '56618.png', '56752.png', '56796.png', '56806.png', '60766.png',
#        '61695.png', '69544.png', '70016.png']

BLACK_LIST = []

TRAIN, VAL, TEST = separateData(os.path.join(os.getcwd(), './train'), '.png', (.8, .1, .1))

STYLE = [[0, False], [5, False], [10, False], [-5, False], [-10, False], [0, True], [5, True], [10, True], [-5, True], [-10, True]]

dataset_info = {
    "description": "Fracture ChestX_Det Dataset",
    "url": "",
    "version": "1.0",
    "year": 2022,
    "contributor": "",
    "data_created": "2022/01/19"
}

dataset_licenses = [{
    "url": "https://www.kaggle.com/mathurinache/chestxdetdataset/version/1",
    "id": 1,
    "name": "Kaggle Fracture ChestX_Det Dataset"
}]

images = []

annotations = []

dataset_catgories = [
    {"id": 0, "name": "fracture"}
]

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def main():
    with open(JSON_PATH) as j:
        json_data = json.load(j)
        annotation_id = 0
        for content in json_data:
            # 確定這組中有骨裂
            if 'Fracture' in content['syms'] and content['file_name'] not in BLACK_LIST:

                """
                file_name   --圖片名稱
                boxes       --標記位置(左上、右下)
                syms        --標記類別(骨裂、氣胸...)
                """

                file_name = content['file_name']
                image_id = int(os.path.splitext(file_name)[0])
                boxes = content['boxes']
                polygons = content['polygons']
                syms = content['syms']
                fracture_index_list = []
                H, W, _ = cv2.imread(os.path.join(os.getcwd(), IMAGE_PATH, f'./{file_name}')).shape
                now_dt = datetime.now()
                date_captured = now_dt.strftime("%Y-%m-%d %H:%M:%S")
                image = {
                    "id": image_id,
                    "license": 1,
                    "file_name": file_name,
                    "height": H,
                    "width": W,
                    "date_captured": date_captured
                }

                images.append(image)

                print("Image JSON: ", image)

                # if file_name in VAL:
                #     SAVE_PATH = os.path.join(os.getcwd(), './resource/val')
                # elif file_name in TRAIN:
                #     SAVE_PATH = os.path.join(os.getcwd(), './resource/train')
                # elif file_name in TEST:
                #     SAVE_PATH = os.path.join(os.getcwd(), './resource/test')

                index_number = 0
                # 紀錄骨裂在syms陣列中的index
                for s in syms:
                    if s == 'Fracture':
                        fracture_index_list.append(index_number)
                    index_number += 1
                print("file_name :", file_name, "  fracture_quantity :", len(fracture_index_list))
                image = cv2.imread(IMAGE_PATH + '/' + file_name)

                polygon_points = []
                bbox_points = []
                # 對有骨裂的座標做迴圈
                for index in fracture_index_list:

                    # Convert bbox to coco format
                    box = boxes[index]
                    bbox_x = box[0]
                    bbox_y = box[1]
                    bbox_width = abs(box[2] - box[0])
                    bbox_height = abs(box[3] - box[1])
                    bbox_point = [bbox_x, bbox_y, bbox_width, bbox_height]
                    bbox_points.append(bbox_point)
                    print("bbox_point: ", str(bbox_point))

                    # Convert polygon to coco format
                    polygon_point = np.array(polygons[index])
                    polygon_point_t = np.transpose(polygon_point)
                    area = PolyArea(polygon_point_t[0], polygon_point_t[1])
                    polygon_point = polygon_point.flatten().tolist()
                    polygon_points.append(polygon_point)
                    print("area: ", area)

                    annotation = {
                        "segmentation": [polygon_point],
                        "area": area,
                        "iscrowd": 0,
                        "image_id": image_id,
                        "bbox": bbox_point,
                        "category_id": 0,
                        "id": annotation_id
                    }

                    annotations.append(annotation)

                    annotation_id += 1

    coco_json_format = {
        "info": dataset_info,
        "licenses": dataset_licenses,
        "images": images,
        "annotations": annotations,
        "categories": dataset_catgories
    }


    with open('./ChestX_Det_COCO.json', 'w', encoding='utf-8') as f:
        json.dump(coco_json_format, f, ensure_ascii=False, indent=4)





if __name__ == '__main__':
    main()