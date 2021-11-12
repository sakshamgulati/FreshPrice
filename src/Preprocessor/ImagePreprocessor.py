import logging
import confuse
import os

import numpy as np
import os
import cv2


class preprocessor:
    """
    this class is used to preprocess the input images
    to be trained for the base CNN
    """

    def __init__(self, config_file="./FreshPrice/conf/ML/preprocessor.yaml"):
        self.logger = logging.getLogger(__name__)
        self.IMG_WIDTH = 200
        self.IMG_HEIGHT = 200
        self.config = confuse.Configuration("FreshPrice", __name__)
        self.config.set_file(config_file)
        self.img_folder = self.config["img_folder"].get(str)

    def create_dataset(self, img_folder):

        img_data_array = []
        class_name = []

        for dir1 in os.listdir(self.img_folder):
            print("Collecting images for: ", dir1)
            self.logger.info("Image preprocessing step started")
            for file in os.listdir(os.path.join(img_folder, dir1)):

                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                try:
                    image = cv2.resize(
                        image,
                        (self.IMG_HEIGHT, self.IMG_WIDTH),
                        interpolation=cv2.INTER_AREA,
                    )
                except:
                    break
                image = np.array(image)
                image = image.astype("float64")
                image /= 255
                img_data_array.append(image)
                class_name.append(dir1)
        return np.array(img_data_array), np.array(class_name)

    @staticmethod
    def product_mapping(a):
        if a == "over_riped_bananas":
            return 1
        else:
            return 0
