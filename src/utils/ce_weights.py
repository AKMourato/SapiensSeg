"""Contains all functionality to precalculate weights
for weighted CE"""
import torchio as tio
import numpy as np


def calculate_ce_weights(root_dir, files, num_classes=8):
    """ Function to calculate weights for weighted CE
    Parameters
    ----------
    root_dir: string
        root directory of the data
    files: list
        patients (paths) retrieved by query
    num_classes: int
        nuber of classes in segmentation task
    """
    print("Calculating weights for Weighted Cross Entropy")
    sums_array = np.zeros(num_classes)
    for file in files:
        dot_index = file.index(".")
        if "-cropped" in file:
            labelmap_path = file[:(dot_index-8)] + "-heart-cropped-label.nii.gz"
        else:
            labelmap_path = file[:dot_index] + "-heart-label.nii.gz"
        labelmap = tio.LabelMap(root_dir + labelmap_path)[tio.DATA]
        for classnum in range(num_classes):
            sums_array[classnum] += (labelmap == classnum).sum()

    total = sums_array.sum()

    weight = sums_array/total
    weight = 1/weight
    weight = weight/weight.sum()
    print("Calculated weights: ")
    print(weight)

    return weight
