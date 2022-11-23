"""This script should be used to benchmark the latest model."""
import os
import datetime
import yaml
import torch
import numpy as np
import pandas as pd
from monai.metrics import DiceMetric, compute_hausdorff_distance
from monai.transforms import AsDiscrete
from utils.postprocessing_utils import PostProcessing
from utils.inference_utils import infer, prepare_model_for_inference, find_gpu_for_inference

if __name__ == "__main__":

    gpu = find_gpu_for_inference()
    print("Benchmarking with GPU " + str(gpu))

    # Load config file and collect settings
    with open("./config.yaml", "r", encoding="utf-8") as yml:
        cfg = yaml.full_load(yml)

    TEST_FILES = []

    ROOT_DIR = cfg["data"]["root_dir"]
    MODEL_WEIGHTS = cfg["benchmark"]["model_weights"]
    POST_PROCESSING = cfg["benchmark"]["post_processing"]
    OUT_DIR = cfg["benchmark"]["out_dir"]
    LABELS = ["Background", "LV", "LA", "PA", "RV", "RA", "Aorta", "Myocardium"]

    # check if out_dir exists, if not create it
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # load model and weights --> parameters can be neglected in this step
    model = prepare_model_for_inference(model_weights=MODEL_WEIGHTS, bodypart_setup='heart', gpu=gpu)

    # prepare matrices to store the output
    dice_matrix = np.zeros((len(TEST_FILES), 8))
    hausdorff_matrix = np.zeros((len(TEST_FILES), 7))

    for i, file in enumerate(TEST_FILES):
        print("Predicting on file: " + file)
        # get ground truth labelmap filename
        dot_index = file.index(".")
        if "-cropped" in file:
            label_file = file[: (dot_index - 8)] + "-heart-cropped-label.nii.gz"
        else:
            label_file = file[:dot_index] + "-heart-label.nii.gz"

        data = {
            "image": os.path.join(ROOT_DIR, file),
            "label": os.path.join(ROOT_DIR, label_file),
        }

        data, _ = infer(model,data,(96,96,96), pix_dim=(1.0, 1.0, 1.0), gpu=gpu)

        arg_max_transform = AsDiscrete(argmax=True)
        one_hot_transform = AsDiscrete(to_onehot=8)

        data["prediction"] = arg_max_transform(data["prediction"][0]).cpu()

        if POST_PROCESSING:
            post_processor = PostProcessing()
            data["prediction"] = post_processor(data["prediction"])

        data["prediction"] = one_hot_transform(data["prediction"])
        data["prediction"] = torch.unsqueeze(data["prediction"], 0)

        data["label"] = one_hot_transform(data["label"])
        data["label"] = torch.unsqueeze(data["label"], 0)

        dice = DiceMetric(reduction="none")
        hausdorff_score = compute_hausdorff_distance(data["prediction"], data["label"])
        dice_score = dice(data["prediction"], data["label"])

        dice_matrix[i, :] = dice_score
        hausdorff_matrix[i, :] = hausdorff_score

    print("Median Dice: " + str(np.mean(dice_matrix)))
    print("Median Hausdorff: " + str(np.median(hausdorff_matrix)))

    # bring the metrics in a nice form to output
    dice_df = pd.DataFrame(dice_matrix)
    hausdorff_df = pd.DataFrame(hausdorff_matrix)

    dice_df.name = "Dice Score"
    hausdorff_df.name = "Hausdorff Distance"

    dice_df.columns = LABELS
    hausdorff_df.columns = LABELS[1:]

    filename = (
        datetime.datetime.today().strftime("%Y-%m-%d")
        + ("_wP" if POST_PROCESSING else "")
        + "_network_metrics.xlsx"
    )
    writer = pd.ExcelWriter(os.path.join(OUT_DIR, filename), engine="xlsxwriter")  # pylint: disable=abstract-class-instantiated
    workbook = writer.book
    worksheet = workbook.add_worksheet("Result")
    writer.sheets["Result"] = worksheet
    worksheet.write_string(0, 0, dice_df.name)

    dice_df.to_excel(writer, sheet_name="Result", startrow=1, startcol=0)
    worksheet.write_string(dice_df.shape[0] + 4, 0, hausdorff_df.name)
    hausdorff_df.to_excel(
        writer, sheet_name="Result", startrow=dice_df.shape[0] + 5, startcol=1
    )
    writer.save()
