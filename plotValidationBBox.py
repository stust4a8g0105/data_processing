import os
import cv2

def plotValidationBBox(detection_txt_path, image_path, result_save_path, conf_threshold=0.01):
    with open(detection_txt_path, 'r') as detection_f:
        detection_lines = detection_f.readlines()

        # each line format: image_filename_wihout_ext conf ltx lty rbx rby
        for i, detection_line in enumerate(detection_lines):
            detection_line = detection_line.split(" ")
            detection_conf = float(detection_line[1])

            if detection_conf >= conf_threshold:
                detected_image_path = imageExtNegotiate(os.path.join(image_path, detection_line[0]))
                saved_image_path = os.path.join(result_save_path, os.path.basename(detected_image_path))
                if not os.path.exists(saved_image_path):
                    detected_image = cv2.imread(detected_image_path)
                else:
                    detected_image = cv2.imread(saved_image_path)
                points = list(map(float, detection_line[2:]))
                ltx, lty, rbx, rby = list(map(int, points))
                left_top_point = (ltx, lty)
                right_bottom_point = (rbx, rby)
                cv2.rectangle(detected_image, left_top_point, right_bottom_point, color=(255, 0, 0), thickness=2)
                print(detected_image_path)
                cv2.imwrite(saved_image_path, detected_image)
                print(f"plot rectangle {left_top_point}, {right_bottom_point} to {saved_image_path}")


def imageExtNegotiate(image_path):
    image_path_png = f"{os.path.splitext(image_path)[0]}.png"
    if os.path.exists(image_path_png):
        return image_path_png
    else:
        return f"{os.path.splitext(image_path)[0]}.jpg"

def main():
    detection_txt_path = os.path.join(os.getcwd(), './results/2688plusChestX_lr_5e4/predicted_txt/2688_test.txt')
    # image_path = os.path.join(os.getcwd(), './data/fracture/2688_plus_ChestX_relabling_histo/val')
    image_path = os.path.join(os.getcwd(), './data/fracture/2688_plus_ChestX_relabling_histo/2688_test')
    result_save_path = os.path.join(os.getcwd(), './results/2688plusChestX_lr_5e4/2688_test')
    plotValidationBBox(detection_txt_path, image_path, result_save_path)

if __name__ == '__main__':
    main()
