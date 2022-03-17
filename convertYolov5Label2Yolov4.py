import os
import cv2

def convertYolov5Label2Yolov4(yolov5_label_dir, images_dir, yolov4_label_save_path, yolov4_image_path_prefix=""):
    # get all labels filenames in yolov5_label_dir
    label_filenames = getAllYolov5LabelFilenames(yolov5_label_dir)

    with open(yolov4_label_save_path, 'a') as yolov4_label_file:
        for label_filename_index, label_filename in enumerate(label_filenames):
            # 將每個yolov5的label.txt檔案轉換成yolov4的label.txt檔案內的每一行
            # image_path x1,y1,x2,y2,id x1,y1,x2,y2,id
            image_path = negotiateImagePath(images_dir, label_filename)
            H, W = getImageShape(image_path)
            yolov5_label_path = os.path.join(yolov5_label_dir, label_filename)
            with open(yolov5_label_path, 'r') as yolov5_label_file:
                yolov5_label_content = yolov5_label_file.read()
                if checkYolov5LabelNotEmpty(yolov5_label_content):
                    yolov5_labels = yolov5_label_content.splitlines()
                    yolov4_label = getYolov4ImagePath(image_path, yolov4_image_path_prefix)
                    for yolov5_label in yolov5_labels:
                        # id cx cy w h
                        yolov5_bbox = yolov5_label.split(" ")
                        for index in range(1, 5):
                            yolov5_bbox[index] = float(yolov5_bbox[index])
                        yolov4_classId, yolov4_x1, yolov4_x2, yolov4_y1, yolov4_y2 = calculateYolov4Label(H, W,
                                                                                                          yolov5_bbox)
                        yolov4_label = f"{yolov4_label} {yolov4_x1},{yolov4_y1},{yolov4_x2},{yolov4_y2},{yolov4_classId}"
                    yolov4_label_file.write(f"{yolov4_label}\n")
                    print('Write data into label file')


def getYolov4ImagePath(image_path, yolov4_image_path_prefix):
    yolov4_label = f"{yolov4_image_path_prefix}{os.path.basename(image_path)}"
    return yolov4_label


def checkYolov5LabelNotEmpty(yolov5_label_content):
    return yolov5_label_content is not ""


def calculateYolov4Label(H, W, yolov5_bbox):
    yolov4_x1 = int(yolov5_bbox[1] * W - (yolov5_bbox[3] * W) / 2)
    yolov4_y1 = int(yolov5_bbox[2] * H - (yolov5_bbox[4] * H) / 2)
    yolov4_x2 = int(yolov5_bbox[1] * W + (yolov5_bbox[3] * W) / 2)
    yolov4_y2 = int(yolov5_bbox[2] * H + (yolov5_bbox[4] * H) / 2)
    yolov4_classId = yolov5_bbox[0]
    return yolov4_classId, yolov4_x1, yolov4_x2, yolov4_y1, yolov4_y2


def negotiateImagePath(images_dir, label_filename):
    image_name = f"{os.path.splitext(label_filename)[0]}.jpg"
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        image_name = f"{os.path.splitext(label_filename)[0]}.png"
        image_path = os.path.join(images_dir, image_name)
    print(f'Processing {os.path.basename(image_path)}')
    return image_path


def getImageShape(image_path):
    image = cv2.imread(image_path)
    (H, W, _) = image.shape
    print(f'Shape of {os.path.basename(image_path)}: {H}, {W}')
    return H, W


def getAllYolov5LabelFilenames(yolov5_label_dir):
    label_filenames = os.listdir(yolov5_label_dir)
    return label_filenames


def main():
    yolov5_label_dir = os.path.join(os.getcwd(), '../Datasets/2688_plus_ChestX_relabling_histo/labels/ChestX_test')
    images_path = os.path.join(os.getcwd(), '../Datasets/yolov4_2688_plus_ChestX_relabling_histo/images/ChestX_test')
    yolov4_label_save_path = os.path.join(os.getcwd(), '../Datasets/yolov4_2688_plus_ChestX_relabling_histo/labels/ChestX_test.txt')
    yolov4_image_path_prefix = 'ChestX_test/'
    convertYolov5Label2Yolov4(yolov5_label_dir, images_path, yolov4_label_save_path, yolov4_image_path_prefix=yolov4_image_path_prefix)

if __name__ == '__main__':
    main()