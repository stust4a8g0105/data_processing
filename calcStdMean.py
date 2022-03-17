import cv2
import os
import numpy as np

def calcStdMean(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    channels = ('R', 'G', 'B')
    for c in range(image.shape[2]):
        image_single_channel = image[:, :, c]
        single_channel_std = np.std(image_single_channel) / 255.
        single_channel_mean = np.mean(image_single_channel) / 255.
        print(f'Channel {channels[c]} std: {single_channel_std}, mean: {single_channel_mean}')

def main():
    image_path = os.path.join(os.getcwd(), "./2688_test/6479719.jpg")
    calcStdMean(image_path)


if __name__ == '__main__':
    main()