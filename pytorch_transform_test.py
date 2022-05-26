from torchvision import transforms
import os
import cv2
import numpy as np
import random


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5, saturation=200, hue=0.2, contrast=0.4)
    ])
    image = cv2.imread(os.path.join(os.getcwd(), "./pytorch_transform_test/images/69937.png"))
    for i in range(3):
        # for j in range(3):
        #     transformed_image = transform(image).permute(1, 2, 0)
        #     transformed_image = (transformed_image.cpu().detach().numpy() * 255).astype(int)
        #     transformed_image[:, :, j] = random.randint(50, 100)
        #     print(f"min: {transformed_image.min()} max:{transformed_image.max()}")
        #     cv2.imwrite(os.path.join(os.getcwd(), f"./pytorch_transform_test/test_channel{i}{j}.png"), transformed_image)
        transformed_image = transform(image).permute(1, 2, 0)
        transformed_image = (transformed_image.cpu().detach().numpy() * 255).astype(int)
        cv2.imwrite(os.path.join(os.getcwd(), f"./pytorch_transform_test/test{0}{i}.png"), transformed_image)
        transformed_image = (np.vectorize(lambda p: 255 - p))(transformed_image)
        cv2.imwrite(os.path.join(os.getcwd(), f"./pytorch_transform_test/test{1}{i}.png"), transformed_image)