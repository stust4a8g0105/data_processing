from calculateIOU import yolov5_calculateIOU_using_intersection
from imageExtNegotiate import imageExtNegotiate
import os
import cv2

def classifyImageBasedOnIoU(image_path, image_save_dir, iou):
    iou_class = f"{int(iou * 10)}"
    image_save_dir_with_iou_class = os.path.join(image_save_dir, iou_class)
    if not os.path.exists(image_save_dir_with_iou_class):
        os.mkdir(image_save_dir_with_iou_class)
    image = cv2.imread(image_path)
    image_save_path = os.path.join(image_save_dir_with_iou_class, os.path.basename(image_path))
    cv2.imwrite(image_save_path, image)
    print(f"{os.path.basename(image_path)} saved")


if __name__ == '__main__':
    image_dir = os.path.join(os.getcwd(), '../yolov5/runs/detect/2688_plus_ChestX_relabling_without_histo_K_Fold_0/detect_with_answer')
    answer_label_dir = os.path.join(os.getcwd(), "../Datasets/K_Fold/For_yolov5/K_Fold_0/2688_plus_ChestX_histo/labels/val")
    result_label_dir = os.path.join(os.getcwd(), "../yolov5/runs/detect/2688_plus_ChestX_relabling_without_histo_K_Fold_0/labels")
    image_save_dir = os.path.join(os.getcwd(), '../Datasets/ClassifyImageBasedOnIoU/K_Fold_without_histo_0')

    image_filenames = os.listdir(image_dir)


    n_answer = len(os.listdir(answer_label_dir))

    for image_filename in image_filenames:
        result_name = os.path.splitext(image_filename)[0]
        result_label_path = os.path.join(result_label_dir, f'{result_name}.txt')
        image_path = imageExtNegotiate(os.path.join(image_dir, result_name))
        answer_label_path = os.path.join(answer_label_dir, f'{result_name}.txt')
        if os.path.exists(result_label_path):
            iou = yolov5_calculateIOU_using_intersection(image_path, answer_label_path, result_label_path)
            print(f"{image_path} IoU: {iou}")
            classifyImageBasedOnIoU(image_path, image_save_dir, iou)
        else:
            print(f"{image_path} IoU: {0}")
            classifyImageBasedOnIoU(image_path, image_save_dir, 0)

