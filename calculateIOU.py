import os
import cv2
from imageExtNegotiate import imageExtNegotiate

def yolov5_calculateIOU(image_path, answer_label_path, result_label_path):
    image = cv2.imread(image_path)
    (H, W, _) = image.shape
    iou = 0
    with open(answer_label_path, 'r') as answer_f, open(result_label_path, 'r') as result_f:
        answer_labels_str = answer_f.read()
        result_labels_str = result_f.read()
        answer_labels = answer_labels_str.split('\n')
        result_labels = result_labels_str.split('\n')

        for result_label_str in result_labels:
            for answer_label_str in answer_labels:
                if answer_label_str and result_label_str:
                    answer_label = answer_label_str.split(' ')[1:5]
                    result_label = result_label_str.split(' ')[1:5]
                    answer_width = int(W * float(answer_label[2]))
                    answer_height = int(H * float(answer_label[3]))
                    answer_ltx = int(W * float(answer_label[0]) - (answer_width / 2))
                    answer_lty = int(H * float(answer_label[1]) - (answer_height / 2))
                    answer_rdx = int(W * float(answer_label[0]) + (answer_width / 2))
                    answer_rdy = int(H * float(answer_label[1]) + (answer_height / 2))

                    result_width = int(W * float(result_label[2]))
                    result_height = int(H * float(result_label[3]))
                    result_ltx = int(W * float(result_label[0]) - (result_width / 2))
                    result_lty = int(H * float(result_label[1]) - (result_height / 2))
                    result_rdx = int(W * float(result_label[0]) + (result_width / 2))
                    result_rdy = int(H * float(result_label[1]) + (result_height / 2))

                    xA = max(answer_ltx, result_ltx)
                    yA = max(answer_lty, result_lty)
                    xB = min(answer_rdx, result_rdx)
                    yB = min(answer_rdy, result_rdy)

                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                    answerArea = answer_width * answer_height
                    resultArea = result_width * result_height

                    iou += interArea / float(answerArea + resultArea - interArea)

        iou = iou / len(answer_labels)
    return iou

def yolov5_calculateDice(image_path, answer_label_path, result_label_path):
    image = cv2.imread(image_path)
    (H, W, _) = image.shape
    dice = 0
    with open(answer_label_path, 'r') as answer_f, open(result_label_path, 'r') as result_f:
        answer_labels_str = answer_f.read()
        result_labels_str = result_f.read()
        answer_labels = answer_labels_str.split('\n')
        result_labels = result_labels_str.split('\n')

        for result_label_str in result_labels:
            for answer_label_str in answer_labels:
                if answer_label_str and result_label_str:
                    answer_label = answer_label_str.split(' ')[1:5]
                    result_label = result_label_str.split(' ')[1:5]
                    answer_width = int(W * float(answer_label[2]))
                    answer_height = int(H * float(answer_label[3]))
                    answer_ltx = int(W * float(answer_label[0]) - (answer_width / 2))
                    answer_lty = int(H * float(answer_label[1]) - (answer_height / 2))
                    answer_rdx = int(W * float(answer_label[0]) + (answer_width / 2))
                    answer_rdy = int(H * float(answer_label[1]) + (answer_height / 2))

                    result_width = int(W * float(result_label[2]))
                    result_height = int(H * float(result_label[3]))
                    result_ltx = int(W * float(result_label[0]) - (result_width / 2))
                    result_lty = int(H * float(result_label[1]) - (result_height / 2))
                    result_rdx = int(W * float(result_label[0]) + (result_width / 2))
                    result_rdy = int(H * float(result_label[1]) + (result_height / 2))

                    xA = max(answer_ltx, result_ltx)
                    yA = max(answer_lty, result_lty)
                    xB = min(answer_rdx, result_rdx)
                    yB = min(answer_rdy, result_rdy)

                    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                    answerArea = answer_width * answer_height
                    resultArea = result_width * result_height

                    dice += (2 * interArea)/ float(answerArea + resultArea)

        dice = dice / len(answer_labels)
    return dice


# darknet result format
# filename_without_ext  conf  ltx  lty  rbx  rby

# answer in same folder with image, with same filename
# format
# class  center_x_ratio  center_y_ratio  width_ratio  height_ratio
def darknet_calculateIOU(image_path, answer_label_path, result_label_path):
    print('...')



def main():
    result_label_dir = os.path.join(os.getcwd(), '../yolov5/runs/detect/2688_plus_ChestX_relabling_histo_K_Fold_0__epoch_249_/labels')
    result_label_filenames = os.listdir(result_label_dir)
    # result_names = [ os.path.splitext(filename)[0] for filename in result_label_filenames]

    answer_label_dir = os.path.join(os.getcwd(), '../Datasets/K_Fold/For_yolov5/K_Fold_0/2688_plus_ChestX_histo/labels/val')
    image_dir = os.path.join(os.getcwd(), '../Datasets/K_Fold/For_yolov5/K_Fold_0/2688_plus_ChestX_histo/images/val')

    iou_cache = []
    dice_cache = []

    n_answer = len(os.listdir(answer_label_dir))
    total_iou = 0
    total_dice = 0

    for result_label_filename in result_label_filenames:
        result_name = os.path.splitext(result_label_filename)[0]
        image_path = imageExtNegotiate(os.path.join(image_dir, result_name))
        answer_label_path = os.path.join(answer_label_dir, f'{result_name}.txt')
        result_label_path = os.path.join(result_label_dir, f'{result_name}.txt')

        iou = yolov5_calculateIOU(image_path, answer_label_path, result_label_path)
        dice = yolov5_calculateDice(image_path, answer_label_path, result_label_path)

        iou_cache.append({
            'filename': result_name,
            'iou': iou
        })
        dice_cache.append({
            'filename': result_name,
            'dice': dice
        })

        total_iou += iou
        total_dice += dice

    print(iou_cache)
    print("total iou: ", total_iou / n_answer)
    print("total dice: ", total_dice / n_answer)


if __name__ == '__main__':
    main()