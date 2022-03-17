import cv2
import numpy as np
import matplotlib.pyplot as plt

def addimage(image_path,image1_path):
    img=cv2.imread(image_path)
    img1=cv2.imread(image1_path)
    h,w,_=img.shape
    img2=cv2.resize(img1,(w,h),interpolation=cv2.INTER_AREA)
    alpha = 0.3
    beta = 1 - alpha
    gamma = 0
    img_add = cv2.addWeighted(img,alpha,img2,beta,gamma)
    cv2.imshow('img_add',img_add)
    cv2.waitKey()
    cv2.destroyAllWindows()

def combineMask(correct_path, predicted_path):
    correct = cv2.imread(correct_path)
    predicted = cv2.imread(predicted_path)

    _, predicted = cv2.threshold(predicted, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    # copy where we'll assign the new values
    correct_with_predicted = np.copy(correct)
    # boolean indexing and assignment based on mask
    correct_with_predicted[(predicted == 255).all(-1)] = [0, 255, 0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(cv2.cvtColor(correct, cv2.COLOR_BGR2RGB))
    ax[1].imshow(cv2.cvtColor(correct_with_predicted, cv2.COLOR_BGR2RGB))
    plt.show()


combineMask('./img/2455_label.png', './img/204_predict.png')