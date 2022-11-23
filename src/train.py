"""Main training script for the body segmentation model."""
import argparse
import os
from ast import literal_eval
import torch
import pytorch_lightning as pl
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from model_definition.segmentation_model import SegmentationModel
from utils.experiment_utils import create_experiment
from data_transfer.query_csv import QueryCSV

#pylint: disable=duplicate-code
def train_parser():
    """
    Argparser function for the measurement JSON generation process.

    Args
    ----------
    bodypart: str
    config: str
    gpu_device: str

    Return
    -------
        parser (:obj: Namespace)

    """

    parser = argparse.ArgumentParser(
        prog="body_segmentation",
        usage="%(prog)s [options]",
        description="Initialize training",
    )

    parser.add_argument(
        "-b",
        "--bodypart",
        metavar="Body part",
        choices=["heart", "thorax", "spine", "diaphragm"],
        default="heart",
        help="part of the body to segment",
    )

    parser.add_argument(
        "-c",
        "--config",
        metavar="Config file",
        default="./config.yaml",
        help="path to the config file",
    )

    parser.add_argument(
        "-g",
        "--gpu_device",
        metavar="GPU device number",
        help="GPU device number to train at",
        choices=[0, 1, 2],
        default=0,
        type=int,
    )

    return parser


def setup(cfg_file):
    """
    Config loading function.
    """
    with open(cfg_file, "r", encoding="utf-8") as yml:
        cfg = yaml.full_load(yml)
    return cfg


def main(bodypart, config, gpu):
    # pylint: disable=too-many-locals
    """
    Inference caller function.
    """
    cfg = setup(config)
    # define seed for repeatable results
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if cfg["overall"]["save_experiment"]:
        create_experiment(experiment_name=cfg["overall"]["experiment_name"])
        logging_path = os.path.join(
            "../experiments/", cfg["overall"]["experiment_name"]
        )
        logger = TensorBoardLogger(logging_path, name="TensorboardLogs")
    else:
        logger = TensorBoardLogger("tb_logs", name="Model")

    # find files to train on
    data_identifier = QueryCSV(
        cfg["sharepoint-download-credentials"], cfg["data"]["path_to_csv"]
    )
    patients = data_identifier.query(query=cfg["data"]["query"])

    if bodypart == "thorax":
        n_classes = cfg["setup"]["thorax"]["n_classes"]
        roi_size = literal_eval(cfg["setup"]["thorax"]["roi_size"])
        pix_dim = literal_eval(cfg["setup"]["thorax"]["pix_dim"])
        structures_log = cfg["setup"]["thorax"]["log_structures_order"]
        select_labels = cfg["setup"]["thorax"]["select_labels"]
    elif bodypart == "spine":
        n_classes = cfg["setup"]["spine"]["n_classes"]
        roi_size = literal_eval(cfg["setup"]["spine"]["roi_size"])
        pix_dim = literal_eval(cfg["setup"]["spine"]["pix_dim"])
        structures_log = cfg["setup"]["spine"]["log_structures_order"]
        select_labels = cfg["setup"]["spine"]["select_labels"]
    elif bodypart == "diaphragm":
        n_classes = cfg["setup"]["diaphragm"]["n_classes"]
        roi_size = literal_eval(cfg["setup"]["diaphragm"]["roi_size"])
        pix_dim = literal_eval(cfg["setup"]["diaphragm"]["pix_dim"])
        structures_log = cfg["setup"]["diaphragm"]["log_structures_order"]
        select_labels = cfg["setup"]["diaphragm"]["select_labels"]
    else:
        n_classes = cfg["setup"]["heart"]["n_classes"]
        roi_size = literal_eval(cfg["setup"]["heart"]["roi_size"])
        pix_dim = literal_eval(cfg["setup"]["heart"]["pix_dim"])
        structures_log = cfg["setup"]["heart"]["log_structures_order"]
        select_labels = cfg["setup"]["heart"]["select_labels"]

    # Load model
    model = SegmentationModel(
        patients=patients,
        bodypart=bodypart,
        n_classes=n_classes,
        select_labels=select_labels,
        roi_size=roi_size,
        pix_dim=pix_dim,
        structures_log=structures_log,
        root_dir=cfg["data"]["root_dir"],
        verse_dir=cfg["data"]["verse_dir"],
        batch_size=cfg["training"]["batch_size"],
        train_val_ratio=cfg["training"]["train_val_ratio"],
        calc_ce_weights=cfg["training"]["calculate_ce_weights"],
        separate_cine_train_val=cfg["training"]["separate_cine_train_val"],
        pretrained_model=cfg["training"]["pretrained_model"],
    )

    # Early stopping helps to prevent overfitting
    callbacks = [
        EarlyStopping(
            monitor="Validation_Loss",
            min_delta=0.00,
            patience=15,
            verbose=False,
            mode="min",
        )
    ]

    # store weights of the lowest validation loss
    if cfg["overall"]["save_experiment"]:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(logging_path, "weights"),
                save_top_k=1,
                monitor="Validation_Loss",
                mode="min",
            )
        )

    # pytorch lighting trainer
    trainer = pl.Trainer(
        max_epochs=500,
        logger=logger,
        accelerator="gpu",
        devices=[gpu],
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    # start training
    trainer.fit(model)


if __name__ == "__main__":
    args = train_parser().parse_args()
    print("Command Line Args:", args)
    main(args.bodypart, args.config, args.gpu_device)
