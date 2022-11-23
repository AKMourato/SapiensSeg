"""
This file should contain all code related to experiments creation stuff.
In the future maybe combined as one class.
"""
import os
import sys
import shutil
import numpy as np
import yaml
import nibabel as nib
# get test patients to exclude them from training
with open("./config.yaml", "r", encoding="utf-8") as yml:
    cfg = yaml.full_load(yml)
TEST_PATIENTS = cfg["benchmark"]["test_patients"]


def create_experiment(experiment_name):
    """Checks and creates folder structure for the experiment

    Parameters
    ----------
    experiment_name: string
        How to name the experiment folders

    """
    # first check, if there is already a root-experiment folder

    os.makedirs("../experiments", exist_ok=True)

    # create experiment folder
    try:
        os.mkdir(os.path.join("../experiments/", experiment_name))
        os.mkdir(os.path.join("../experiments/", experiment_name, "weights"))
    except OSError:
        user_input = input(
            f"There is already an experiment folder for {experiment_name}. Do you wish to overwrite it? "
        )
        if user_input.lower() in ["yes", "y", "true", "1"]:
            shutil.rmtree(os.path.join("../experiments/", experiment_name))
            os.mkdir(os.path.join("../experiments/", experiment_name))
            os.mkdir(os.path.join("../experiments/", experiment_name, "weights"))
        else:
            sys.exit("...  Exiting.")

    # copy config file in experiment folder
    shutil.copy("config.yaml", os.path.join("../experiments", experiment_name))

    # rename to avoid complications in inference.py
    os.rename(
        os.path.join("../experiments", experiment_name, "config.yaml"),
        os.path.join("../experiments", experiment_name, "network_parameters.yaml"),
    )

    # no return value, as this function just creates the needed folder structure
    return True


def build_file_paths(patients, bodypart):
    """This function builds the paths for reading in
    images and labels according to the query results.
    Note: There must be correspondence between the naming of the files
    and the entry in the excel file. See README for more details

    Parameter
    ---------
    patients: list
        Returned list from QueryCSV.query()
    """
    all_files = []
    for sample in patients:

        # 174, 177 and 194 are too small for cropping! Think of better handling!
        # 822, 941, 942 are special hearts and cant be included
        if f"{int(sample[0]):03}" in ["174", "177", "197", "822", "941", "942"] and bodypart == 'heart':
            continue
        # exclude fixed test patients
        if f"{int(sample[0]):03}" in TEST_PATIENTS and bodypart == 'heart':
            continue

        path = f"{int(sample[0]):05}" + "/dl-data/pat" + f"{int(sample[0]):03}"
        # see if series number is specified
        if sample[1] != "None":
            if sample[1] == "merged":
                path = path + "-ser_merged"
            elif sample[1] > 99:
                path = path + "-ser" + str(sample[1])
            else:
                path = path + "-ser" + f"{int(sample[1]):03}"
        # see if frame is specified
        if sample[2] != "None":
            path = path + "-frame" + f"{int(sample[2]):03}"
        # just get the cropped images
        if sample[3] != "None" and bodypart == "heart":
            if sample[3] == "yes":
                path = path + "-cropped"
        path = path + ".nii.gz"
        # The path should now look like
        # "00006/pat006-ser003-frame054.nii.gz"
        all_files.append(path)

    return all_files


def build_file_paths_verse(verse_dir):
    """This function builds the paths for reading in
    images for the Spine segmentation.
    """
    files = []
    for i in os.listdir(os.path.join(verse_dir,"images")):
        img = nib.load(os.path.join(verse_dir,"images", i))
        # ignoring samples that have smaller size than 48 px.
        if img.shape[2] >= 48:
            files.append(i)

    return files


def random_split_files(all_files, ratio, separate_cine_train_val):
    """Randomly split training and validation images

    Parameters
    ----------
    all_files: list
        list with all image files
    ratio: float
        desired ratio how to split
    separate_cine_train_val: boolean
        if True: same patient will not appear
        training and validation set
    """
    # First calculate desired length for validation set
    n_training_files = int(ratio * len(all_files))
    n_validataion_files = len(all_files) - n_training_files

    # Easy case, just shuffle the files and split accordind to ratio
    if not separate_cine_train_val:
        np.random.shuffle(all_files)
        return all_files[:n_training_files], all_files[n_training_files:]

    # Second case: separate_cine_train_val == True
    validation_files = []
    training_files = all_files

    while len(validation_files) < n_validataion_files:
        # 1. Select random file from list
        random_file = np.random.choice(training_files)
        validation_files.append(random_file)
        training_files.remove(random_file)
        # 2. Put all samples from the same patient into list
        files_to_remove = []
        for file in training_files:
            # [0:6] = patxxx
            if file[0:6] == random_file[0:6]:
                validation_files.append(file)
                files_to_remove.append(file)
        # remove shoud happen outside the first loop
        for file in files_to_remove:
            training_files.remove(file)
    return training_files, validation_files
