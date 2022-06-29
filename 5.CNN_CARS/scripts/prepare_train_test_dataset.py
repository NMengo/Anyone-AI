import argparse
import pandas as pd
import os
"""
This script will be used to separate and copy images coming from
`car_ims.tgz` (extract the .tgz content first) between `train` and `test`
folders according to the column `subset` from `car_dataset_labels.csv`.
It will also create all the needed subfolders inside `train`/`test` in order
to copy each image to the folder corresponding to its class.

The resulting directory structure should look like this:
    data/
    ├── car_dataset_labels.csv
    ├── car_ims
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── ...
    ├── car_ims_v1
    │   ├── test
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000046.jpg
    │   │   │   ├── 000047.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000450.jpg
    │   │   │   ├── 000451.jpg
    │   │   │   ├── ...
    │   ├── train
    │   │   ├── AM General Hummer SUV 2000
    │   │   │   ├── 000001.jpg
    │   │   │   ├── 000002.jpg
    │   │   │   ├── ...
    │   │   ├── Acura Integra Type R 2001
    │   │   │   ├── 000405.jpg
    │   │   │   ├── 000406.jpg
    │   │   │   ├── ...
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. E.g. "
            "`/home/app/src/data/car_ims/`."
        ),
    )
    parser.add_argument(
        "labels",
        type=str,
        help=(
            "Full path to the CSV file with data labels. E.g. "
            "`/home/app/src/data/car_dataset_labels.csv`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "train/test splits. E.g. `/home/app/src/data/car_ims_v1/`."
        ),
    )
    args = parser.parse_args()
    return args


def main(data_folder, labels, output_data_folder):
    if not os.path.exists(output_data_folder):
        os.mkdir(output_data_folder)
        os.mkdir(output_data_folder+'/train')
        os.mkdir(output_data_folder+'/test')

    cars_labels = pd.read_csv(labels)
    car_class = cars_labels['class'].unique()

    for classes in car_class:
        if not os.path.exists(output_data_folder+'/train'+'/'+str(classes)):
            os.mkdir(output_data_folder+'/train'+'/'+str(classes))
        if not os.path.exists(output_data_folder+'/test'+'/'+str(classes)):
            os.mkdir(output_data_folder+'/test'+'/'+str(classes))
        
    for i in range(len(cars_labels)):
        src = data_folder+'/'+str(cars_labels.loc[i, 'img_name'])
        dst = (output_data_folder+'/'+str(cars_labels.loc[i, 'subset'])
        +'/'+str(cars_labels.loc[i, 'class'])+'/'+str(cars_labels.loc[i, 'img_name']))

        if not os.path.exists(dst):
            os.link(src, dst)

if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.labels, args.output_data_folder)
