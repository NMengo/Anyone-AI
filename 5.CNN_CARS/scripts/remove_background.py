"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""
from utils.utils import walkdir
from utils.detection import *
import argparse
import os
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    if not os.path.exists(output_data_folder):
        os.mkdir(output_data_folder)
        os.mkdir(output_data_folder+'/train')
        os.mkdir(output_data_folder+'/test')

    for dir_path, img_name in walkdir(data_folder):
        img_path = os.path.normpath(dir_path+'/'+img_name)
        new_img_path = dir_path.replace('car_ims_v1', 'car_ims_v2')
        if not os.path.exists(new_img_path):
            os.mkdir(new_img_path)
        else:
            pass
        new_img_path = os.path.normpath(new_img_path+'/'+img_name)
        img = cv2.imread(img_path)
        l, t, r, b = get_vehicle_coordinates(img)
        cropped_img = img[int(t):int(b), int(l):int(r)]
        print(new_img_path)
        cv2.imwrite(new_img_path, cropped_img)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
