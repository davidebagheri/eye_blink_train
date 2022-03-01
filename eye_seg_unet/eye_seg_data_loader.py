import numpy as np
import cv2 as cv
from pathlib import Path
import os

"""
Data Loader class for retrieving Eye_Segmentation_Database data (https://arxiv.org/pdf/1910.05283.pdf)
"""

class EyeSegDataLoader:
    def __init__(self, data_path, img_size=(80,40), normalize=True):
        x_train_path = [str(path) for path in Path(os.path.join(data_path, "training", "images")).glob("*.png")]
        y_train_path = [str(path) for path in Path(os.path.join(data_path, "training", "gts")).glob("*.png")]
        x_test_path = [str(path) for path in Path(os.path.join(data_path, "testing", "images")).glob("*.png")]
        y_test_path = [str(path) for path in Path(os.path.join(data_path, "testing", "gts")).glob("*.png")]

        if len(x_train_path) != len(y_train_path) or len(x_test_path) != len(y_test_path):
            print("Something wrong with data..")
        else:
            self.x_train = np.empty([len(x_train_path)] + list(img_size[::-1]) + [3])
            self.y_train = np.empty([len(y_train_path)] + list(img_size[::-1]) + [3])
            self.x_test = np.empty([len(x_test_path)] + list(img_size[::-1]) + [3])
            self.y_test = np.empty([len(y_test_path)] + list(img_size[::-1]) + [3])

            self.load_set(x_train_path, self.x_train, img_size)
            self.load_set(y_train_path, self.y_train, img_size)
            self.load_set(x_test_path, self.x_test, img_size)
            self.load_set(y_test_path, self.y_test, img_size)

            if normalize:
                for set in [self.x_train, self.x_test]:
                    set /= 255.0

    def load_set(self, data_set_path, preallocated_space, img_size):
        n = len(preallocated_space)
        for i in range(n):
            image = self.load_image(data_set_path[i], img_size)
            preallocated_space[i] = image

    def load_image(self, path, img_size):
        image = cv.imread(path)
        image = cv.resize(image, img_size, cv.INTER_NEAREST)
        return image
