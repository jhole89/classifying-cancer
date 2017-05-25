import os
from glob import glob
import numpy as np
import tflearn
from skimage import color, io
from scipy.misc import imresize
from sklearn.model_selection import train_test_split

def train():

    file_path = 'images/'

    malignant_file_path = os.path.join(file_path, 'SOB_M_*.png')
    benign_file_path = os.path.join(file_path, 'SOB_B_*.png')

    malignant_files = sorted(glob(malignant_file_path))
    benign_files = sorted(glob(benign_file_path))

    n_files = len(malignant_files) + len(benign_files)
    print(n_files)

    image_height = 460
    image_width = 700
    colour_channels = 3

    allX = np.zeros((n_files, image_height, image_width, colour_channels), dtype='float64')
    allY = np.zeros(n_files)
    count = 0

    for file in malignant_files:
        try:
            img = io.imread(file)
            new_img = imresize(img, (image_height, image_width, colour_channels))
            allX[count] = np.array(new_img)
            count += 1
        except:
            continue

    for file in benign_files:
        try:
            img = io.imread(file)
            new_img = imresize(img, (image_height, image_width, colour_channels))
            allX[count] = np.array(new_img)
            count += 1
        except:
            continue
