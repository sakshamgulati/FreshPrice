# script to break down images to test and train
# split images into training and test folders
import os
import numpy as np
import shutil
import random


def train_test_split_folder(classes_dir):
    """

    :param classes_dir: input the name of the labels you want to test and train
    :return: Images divided into train and test
    """

    root_dir = "./Freshprice/Data/Image_data"
    # classes_dir = os.listdir(root_dir)

    test_ratio = 0.20

    for cls in classes_dir:
        if not os.path.exists(root_dir + "/train/" + cls):
            os.makedirs(root_dir + "/train/" + cls)
            os.makedirs(root_dir + "/test/" + cls)
        else:
            shutil.rmtree(root_dir + "/train/" + cls)  # Removes all the subdirectories!
            os.makedirs(root_dir + "/train/" + cls, exist_ok=True)
            os.makedirs(root_dir + "/test/" + cls, exist_ok=True)

        src = root_dir + "/" + cls
        print("data being copied from: ", src)

        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, test_FileNames = np.split(
            np.array(allFileNames), [int(len(allFileNames) * (1 - test_ratio))]
        )

        train_FileNames = [src + "/" + name for name in train_FileNames.tolist()]
        test_FileNames = [src + "/" + name for name in test_FileNames.tolist()]

        print("*****************************")
        print("Total images for %s: %d" % (cls, len(allFileNames)))
        print("Training images for %s: %d " % (cls, len(train_FileNames)))
        print("Testing images for %s: %d" % (cls, len(test_FileNames)))
        print("*****************************")

        for name in train_FileNames:
            shutil.copy(name, root_dir + "/train/" + cls)

        for name in test_FileNames:
            shutil.copy(name, root_dir + "/test/" + cls)
    print("Copying Done!")
