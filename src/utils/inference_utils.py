"""benchmark.py & inference.py need partly the same functionality, which should be bundled here."""
import os
from ast import literal_eval
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import allow_missing_keys_mode, Invertd
from utils.preprocessing_utils import eval_transforms
from model_definition.segmentation_model import SegmentationModel


def infer(model, data, roi_size, pix_dim, gpu):
    """Function to infer on new data
    Parameters
    ----------
    model: DL model
        Model with loaded weights,
        most likely retrieved from
        prepare_model_for_inference
    data: dict
        dict with key "image", network
        prediction will be stored in
        data["prediction"]
    roi_size: triple
        UNETR patching size
    pix_dim: triple
        data resolution / spacing
    gpu: int
        which gpu should be used for
        prediction.
    """
    # define gpu device
    device = torch.device("cuda:" + str(gpu))

    # preprocess the data
    transforms = eval_transforms(pix_dim)
    with allow_missing_keys_mode(transforms):
        data = transforms(data)
    # add batch dimension and infer using sliding window
    data["image"] = torch.unsqueeze(data["image"], 0).to(device=device)
    with torch.no_grad():
        data["prediction"] = sliding_window_inference(
            data["image"], roi_size, 4, model, overlap=0.8
        )

    return data, transforms


def inverse_preprocessing(data, transforms):
    """Function to inverse preprocessing transforms.
    data["prediction_meta_dict"] must be used
    for saving the labelmap.

    Parameters
    ----------
    data: dict
        dictionary with keys: image, image_meta_dict, prediction
    transforms monai compose
        transforms applied to the data
    """
    # invert transform
    eval_transforms_inverted = Invertd(
        keys="prediction", transform=transforms, orig_keys="image"
    )
    # apply transform
    data = eval_transforms_inverted(data)
    return data


def prepare_model_for_inference(bodypart_setup, model_weights, gpu):
    """Function to prepare model for inference
    Parameters
    ----------
    bodypart_setup: dict
        setting for bodypart
    model_weights: dict
        trained model weights to load in
        Segmentation Model
    gpu: int
        which gpu should be used for
        prediction.
    """
    # define gpu device
    device = torch.device("cuda:" + str(gpu))

    # load model and weights --> parameters can be neglected in this step
    model = SegmentationModel(
        roi_size=literal_eval(bodypart_setup["roi_size"]),
        n_classes=bodypart_setup["n_classes"],
    )

    state_dict = torch.load(model_weights, map_location="cuda:" + str(gpu))[
        "state_dict"
    ]
    model.load_state_dict(state_dict)
    model.to(device)

    # deactivate dropout, ...
    model.eval()

    return model


def find_gpu_for_inference():
    """Function to find the gpu with
    the most available memory

    Parameters
    ----------

    """
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >gpu_memory_summary")

    with open("gpu_memory_summary", "r", encoding="utf-8") as tmp_file:
        gpu_summary = tmp_file.readlines()

    memory_available = [int(x.split()[2]) for x in gpu_summary]
    os.remove("gpu_memory_summary")
    return np.argmax(memory_available)
