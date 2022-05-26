import numpy as np
import os

def calcMultiClassRecallPrecision(gt_label_dir, pred_result_path):
    filepath_gt_pred_mapper = {}
    gt_filenames = os.listdir(gt_label_dir)
    gt_filenames = list(filter(lambda filename: filename.endswith(".txt"), gt_filenames)) # get all .txt labels filename
    gt_filepaths = list(map(lambda filename: os.path.join(gt_label_dir, filename), gt_filenames)) # convert gt_filenames to abs file path

    for gt_filepath in gt_filepaths:
        with open(gt_filepath, 'r') as gt_file:
            gt_line = gt_file.readline() # read the first line of gt label file
            if gt_line:
                gt = int(gt_line.split(" ")[0])  # get the ground truth
                gt_filename = os.path.splitext(os.path.basename(gt_filepath))[0]
                filepath_gt_pred_mapper[gt_filename] = {
                    "gt": gt
                }

    with open(pred_result_path, 'r') as pred_file:
        pred_lines = pred_file.readlines()
        for pred_line in pred_lines:
            pred_line_contents = pred_line.split(" ")
            pred_filename = pred_line_contents[0]
            pred_filename = os.path.splitext(pred_filename)[0]
            pred = int(pred_line_contents[1])
            if pred >= 0 and pred_filename in filepath_gt_pred_mapper:
                filepath_gt_pred_mapper[pred_filename]["pred"] = pred


    confusion_matrix = np.zeros((219, 219))

    for filename, gt_pred_dict in filepath_gt_pred_mapper.items():
        gt = gt_pred_dict["gt"]
        pred = gt_pred_dict["pred"]
        confusion_matrix[gt][pred] += 1

    fp = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    fn = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    tp = np.diag(confusion_matrix)
    tn = confusion_matrix.sum() - (fp + fn + tp)
    fp = fp.astype(float)
    fn = fn.astype(float)
    tp = tp.astype(float)
    tn = tn.astype(float)
    Accuracy = (tp + tn) / (tp + tn + fn + fp)
    Precision = tp / (tp + fp)
    for i in range(219):
        if np.isnan(Precision[i]):
            Precision[i] = 0
    Recall = tp / (tp + fn)
    F1_score = 2 * ((Precision * Recall) / (Precision + Recall))
    for i in range(219):
        if np.isnan(F1_score[i]):
            F1_score[i] = 0
    print('Accuracy : ', sum(Accuracy) / 219)
    print('Precision : ', sum(Precision) / 219)
    print('Recall : ', sum(Recall) / 219)
    print('F1_score : ', sum(F1_score) / 219)

if __name__ == '__main__':
    gt_label_dir = os.path.join(os.getcwd(), "../TBrain_AI/Dataset/orchid/labels/augmented_val")
    pred_result_path = os.path.join(os.getcwd(), "../yolov5/runs/detect/orchid_yolov5x_mosaic/result.txt")
    calcMultiClassRecallPrecision(gt_label_dir, pred_result_path)