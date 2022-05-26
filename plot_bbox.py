import cv2
import os
from imageExtNegotiate import imageExtNegotiate


def annotateRectangle(img_dir, label_dir, save_path, ext='png', color=(0, 255, 0)):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 取得所有label文字檔的路徑
    label_path = [os.path.join(label_dir, label) for label in os.listdir(label_dir) if label.endswith('.txt')]

    for label_file in label_path:
        labels = []
        label_basename = os.path.splitext(os.path.basename(label_file))[0]
        with open(label_file, 'r') as f:
            for label in f.read().split('\n'):
                if label:
                    print(label)
                    info = label.split()
                    print(info)
                    tl_x = float(info[1]) - float(info[3]) / 2
                    tl_y = float(info[2]) - float(info[4]) / 2
                    br_x = float(info[1]) + float(info[3]) / 2
                    br_y = float(info[2]) + float(info[4]) / 2
                    labels.append([tl_x, tl_y, br_x, br_y])

        img_path = imageExtNegotiate(os.path.join(img_dir, f'./{label_basename}.{ext}'))
        print(img_path)
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            (H, W, _) = image.shape
            for i, label in enumerate(labels):
                cv2.rectangle(image, (int(W * label[0]), int(H * label[1])), (int(W * label[2]), int(H * label[3])), color, 3)
                image_save_path = os.path.join(save_path, os.path.basename(img_path))
            cv2.imwrite(image_save_path, image)
            print(f"saved result to {image_save_path}")



if __name__ == '__main__':
    img_dir = os.path.join(os.getcwd(), '../yolov5/runs/detect/24times_augmentation_K_Fold_0')
    label_dir = os.path.join(os.getcwd(), '../Datasets/K_Fold/24times_augmentation_for_yolov5/K_Fold_0/labels/val')
    save_path = os.path.join(os.getcwd(), '../yolov5/runs/detect/24times_augmentation_K_Fold_0/augmented_val_with_answer')
    annotateRectangle(img_dir, label_dir, save_path, ext='png', color=(0, 255, 0))