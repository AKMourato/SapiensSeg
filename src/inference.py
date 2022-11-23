"""Script to generate WH model predictions. Settings should be done in config.py"""
import os
import argparse
import sys
from ast import literal_eval
import yaml
from monai.transforms import SaveImage
from utils.postprocessing_utils import PostProcessing
from utils.inference_utils import (
    infer,
    inverse_preprocessing,
    prepare_model_for_inference,
    find_gpu_for_inference,
)

# pylint: disable=duplicate-code
def inference_parser():
    """
    Argparser function for the measurement JSON generation process.

    Args
    ----------


    Return
    -------
        parser (:obj: Namespace)

    """

    parser = argparse.ArgumentParser(
        prog="body_segmentation",
        usage="%(prog)s [options]",
        description="Initialize inference",
    )

    parser.add_argument(
        "-b",
        "--bodypart",
        metavar="Body part",
        choices=["heart", "thorax", "spine", "diaphragm"],
        default="heart",
        help="part of the body to predict",
    )

    parser.add_argument(
        "-c",
        "--config",
        metavar="Config file",
        default="./config.yaml",
        help="path to the config file",
    )

    return parser


def setup(cfg_file):
    """
    Config loading function.
    """
    # Load all settings from config file
    with open(cfg_file, "r", encoding="utf-8") as yml:
        cfg = yaml.full_load(yml)
    return cfg


def main(bodypart, config):
    """
    Inference caller function.
    """

    cfg = setup(config)

    gpu = find_gpu_for_inference()
    print("Predicting with GPU " + str(gpu))

    # collect files to predict on
    try:
        files = sorted(
            os.listdir(os.path.join(cfg["inference"]["root_dir_prediction"], "ct_test"))
        )
    except FileNotFoundError:
        sys.exit("Couldn't import inference files, check path.")

    # create directory if not exists
    os.makedirs(cfg["inference"]["out_dir"], exist_ok=True)

    # setup WH model, load weights and set mode eval
    model = prepare_model_for_inference(
        cfg["setup"][bodypart], cfg["inference"]["model_weights"], gpu
    )

    # start predicting
    for _, file in enumerate(files):

        print("Predicting on file: " + file)
        # follow monai style of dict for handling images, transforms, ...
        data = {
            "image": os.path.join(
                cfg["inference"]["root_dir_prediction"], "ct_test", file
            )
        }

        # predicted image will be found in data["prediction"]
        data, transforms = infer(
            model,
            data,
            literal_eval(cfg["setup"][bodypart]["roi_size"]),
            literal_eval(cfg["setup"][bodypart]["pix_dim"]),
            gpu,
        )

        # get most likely label and back to cpu()
        data["prediction"] = data["prediction"].argmax(dim=1).cpu()

        # invert transforms, needed as CropForegroundd is not updating affine matrix
        data = inverse_preprocessing(data=data, transforms=transforms)

        # postprocessing
        if bodypart == "heart":
            post_processor = PostProcessing(verbose=True)
            data["prediction"] = post_processor(data["prediction"])

        # save prediction
        saver = SaveImage(
            output_dir=cfg["inference"]["out_dir"],
            output_postfix=f"{bodypart}-nqc-label",
            resample=False,
            writer="ITKWriter"
        )
        saver(data["prediction"], meta_data=data["prediction_meta_dict"])


if __name__ == "__main__":
    args = inference_parser().parse_args()
    print("Command Line Args:", args)
    main(args.bodypart, args.config)
