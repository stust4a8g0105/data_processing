import cv2
import math


class Data_augmentation:
    def __init__(self, image, angle, flip_image, point=None, keep_size=False):
        """

        需要參數:
            image       --需要是已經讀入的圖片array。
            angle       --旋轉角度。

        可給也可以不給:
            point       --左上右下座標
                            一組 : point = [ [left_top_X,left_top_Y , right_down_X,right_down_Y] ]。

                            多組 : point = [ [left_top_X,left_top_Y , right_down_X,right_down_Y],
                                            [left_top_X,left_top_Y , right_down_X,right_down_Y],
                                            [left_top_X,left_top_Y , right_down_X,right_down_Y] ]。

            keep_size   --圖片是否要超出邊界。

        -------------------------------
        GET -可以取得的資源

        get_after_spin_image()  --取得旋轉後的圖片。
        get_after_spin_point()  --取的旋轉後的座標。
        get_yolo_point()  --取得將原始座標成yolo座標

        -------------------------------
        SAVE -可以儲存的資源
        save_yolo_point_txt(save_path)  --將座標轉換成yolo座標後輸出txt至指定路徑

        """
        self.image = image
        self.angle = angle
        self.point = point
        self.keep_size = keep_size
        self.after_spin_image = None
        self.after_spin_image_point = None
        self.yolo_point = []

        if flip_image:
            self.flip()

        self.spin_image()
        if self.point is not None:
            self.calculate_after_spin_point()

    def flip(self):
        (H, W) = self.image.shape[:2]
        self.image = cv2.flip(self.image, 1)

        flip_point = []
        for point in self.point:
            left_top_x = point[0]
            left_top_y = point[1]
            right_down_x = point[2]
            right_down_y = point[3]

            new_left_top_x = W - right_down_x
            new_left_top_y = left_top_y
            new_right_down_x = W - left_top_x
            new_right_down_y = right_down_y
            flip_point.append([new_left_top_x, new_left_top_y, new_right_down_x, new_right_down_y])

        self.point = flip_point

    def spin_image(self):
        (H, W) = self.image.shape[:2]

        M = cv2.getRotationMatrix2D((W / 2, H / 2), self.angle, 1)
        if self.keep_size:

            # 根據旋轉矩陣進行仿射變換
            img_arr = cv2.warpAffine(self.image, M, (W, H))
            self.after_spin_image = img_arr
        else:
            new_H = int(
                W * math.fabs(math.sin(math.radians(self.angle))) + H * math.fabs(
                    math.cos(math.radians(self.angle))))
            new_W = int(
                H * math.fabs(math.sin(math.radians(self.angle))) + W * math.fabs(
                    math.cos(math.radians(self.angle))))
            M[0, 2] += (new_W - W) / 2
            M[1, 2] += (new_H - H) / 2
            img_arr = cv2.warpAffine(self.image, M, (new_W, new_H))

            self.after_spin_image = img_arr

    def get_after_spin_image(self):
        return self.after_spin_image

    def calculate_after_spin_point(self):
        angel_abs = abs(self.angle)

        # # y位移量
        y_displacement = int(self.after_spin_image.shape[0] - (self.image.shape[0] * math.cos(math.radians(angel_abs))))
        # X位移量
        x_displacement = int((self.image.shape[1] * math.sin(math.radians(angel_abs))))

        new_point = []
        for point in self.point:
            left_top_x = point[0]
            left_top_y = point[1]
            right_down_x = point[2]
            right_down_y = point[3]
            if self.angle < 0:
                new_left_top_x = left_top_x * math.cos(math.radians(angel_abs)) - left_top_y * math.sin(
                    math.radians(angel_abs)) + x_displacement
                new_left_top_y = left_top_y * math.cos(math.radians(angel_abs)) + left_top_x * math.sin(
                    math.radians(angel_abs))

                new_right_down_x = right_down_x * math.cos(math.radians(angel_abs)) - right_down_y * math.sin(
                    math.radians(angel_abs)) + x_displacement
                new_right_down_y = right_down_y * math.cos(math.radians(angel_abs)) + right_down_x * math.sin(
                    math.radians(angel_abs))

                new_point.append([int(new_left_top_x), int(new_left_top_y),
                                  int(new_right_down_x), int(new_right_down_y)])
            else:
                new_left_top_x = left_top_x * math.cos(math.radians(angel_abs)) + left_top_y * math.sin(
                    math.radians(angel_abs))
                new_left_top_y = left_top_y * math.cos(math.radians(angel_abs)) - left_top_x * math.sin(
                    math.radians(angel_abs)) + y_displacement
                new_right_down_x = right_down_x * math.cos(math.radians(angel_abs)) + right_down_y * math.sin(
                    math.radians(angel_abs))
                new_right_down_y = right_down_y * math.cos(math.radians(angel_abs)) - right_down_x * math.sin(
                    math.radians(angel_abs)) + y_displacement

                new_point.append([int(new_left_top_x), int(new_left_top_y),
                                  int(new_right_down_x), int(new_right_down_y)])

        self.after_spin_image_point = new_point

    def get_after_spin_point(self):
        return self.after_spin_image_point

    def format_point_to_yolo(self, save_name=None):
        (H, W) = self.after_spin_image.shape[:2]
        yolo_point_list = []
        for after_spin_point in self.after_spin_image_point:
            # 左上（xmin、ymin）
            x_min = after_spin_point[0]
            y_min = after_spin_point[1]
            # 右下（xmax、ymax）
            x_max = after_spin_point[2]
            y_max = after_spin_point[3]
            x = (x_min + (x_max - x_min) / 2) * 1.0 / W
            y = (y_min + (y_max - y_min) / 2) * 1.0 / H
            w1 = (x_max - x_min) * 1.0 / W
            h1 = (y_max - y_min) * 1.0 / H
            yolo_point_list.append([0, x, y, w1, h1])
            # print(0, ",", x, ",", y, ",", w1, ",", h1)

            if save_name is not None:
                with open(save_name + '.txt', 'a') as file:
                    file.write(str(0) + " " + str(x) + " " + str(y) + " " + str(w1) + " " + str(h1))
                    file.write("\n")
                    file.close()

        self.yolo_point = yolo_point_list

    def save_yolo_point_txt(self, save_name):
        if len(self.yolo_point) == 0:
            self.format_point_to_yolo(save_name)

    def get_yolo_point(self):
        self.format_point_to_yolo()

        return self.yolo_point
