"""Contains lightning (and monai) model for training and validating
body segmentation."""
import torchmetrics
import torch
import pytorch_lightning as pl
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR
from monai.data import CacheDataset, list_data_collate
from datasets.dataset import BDataset
from utils.experiment_utils import (
    build_file_paths,
    random_split_files,
)
from utils.preprocessing_utils import train_transforms, val_transforms
from utils.ce_weights import calculate_ce_weights


class SegmentationModel(pl.LightningModule):
    """
    A class that contains methods
    to train and validate UNETR
    ...

    Attributes
    ----------
    patients: list
        result of query
    bodypart: str
        target part of the body
    n_classes: int
        number of classes to predict
    roi_size: str
        dimensions of patch cube
    pix_dim: str
        data spacing / resolution
    structures_log: list
        name of each label according to target body part (orderly)
    root_dir: string
         directory where to find the data
    batch_size: int
        batch size
    train_val_ratio: float
        ratio for splitting into train and validations set
    calculate_ce_weights: boolean
        calculate weights for weighted cross entropy
    separate_cine_train_val: boolean
        if True: one patient cannot appear in train
        and validation set (that means, all series are either
        in train or validation)
    pretrained_model: string / None
        if a pretrained model should be loaded


    Methods
    -------
    forward(x)
        Forward pass of DL model
    training_step(batch, batch_idx)
        Lightning handled training step
    validation_step(batch, batch_idx)
        Lightning handled validation step
    prepare_data
        First function called after init
        -> all data related tasks are performed
    train_dataloader
        Dataloader
    val_dataloader
        Dataloader
    configure_optimizer
        Lightning handled optimizer
    """

    # pylint: disable=too-many-instance-attributes, too-many-ancestors, arguments-differ
    # Give the user enough flexability

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        see Class docstring
        """
        super().__init__()

        self.patients = kwargs.get("patients")
        self.bodypart = kwargs.get("bodypart")
        self.n_classes = kwargs.get("n_classes")
        self.select_labels = kwargs.get("select_labels")
        self.roi_size = kwargs.get("roi_size")
        self.pix_dim = kwargs.get("pix_dim")
        self.structures_log = kwargs.get("structures_log")
        self.root_dir = kwargs.get("root_dir")
        self.verse_dir = kwargs.get("verse_dir")
        self.batch_size = kwargs.get("batch_size")
        self.train_val_ratio = kwargs.get("train_val_ratio")
        self.calc_ce_weights = kwargs.get("calc_ce_weights")
        self.separate_cine_train_val = kwargs.get("separate_cine_train_val")
        self.pretrained_model = kwargs.get("pretrained_model")
        self.train_ds: object
        self.val_ds: object
        self.ce_weights: object

        self.model = UNETR(
            in_channels=1,
            out_channels=self.n_classes,
            img_size=self.roi_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            conv_block=True,
            dropout_rate=0.0,
        ).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        # pylint: disable=not-callable
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # pylint: disable=unused-argument
        images, labels = (batch["image"].cuda(), batch["label"].cuda())
        output = self.forward(images)

        loss_function = DiceCELoss(
            to_onehot_y=True, softmax=True, ce_weight=self.ce_weights
        )
        train_loss = loss_function(output, labels)

        # and log
        self.log(
            "Training_Loss",
            train_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        # pylint: disable=unused-argument
        images, labels = batch["image"], batch["label"]
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, self.roi_size, sw_batch_size, self.forward
        )

        if self.calc_ce_weights:
            self.ce_weights = (
                torch.from_numpy(self.weights_for_ce).to(device=self.device).float()
            )

        loss = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=self.ce_weights)
        validation_loss = loss(outputs, labels)

        # addtional metrics F1
        f1_valid = torchmetrics.F1Score(
            num_classes=self.n_classes, average=None, mdmc_average="samplewise"
        )

        f1_valid_score = f1_valid(outputs, labels.to(torch.int16))

        structures_log = {}
        for idx, sub in enumerate(self.structures_log):
            structures_log[sub] = f1_valid_score[idx]

        # and log
        self.log(
            "Validation_Loss",
            validation_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.batch_size,
        )
        self.log(
            "Validation_F1",
            structures_log,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        return validation_loss

    def prepare_data(self):
        # first we load a pretrained model if specified
        if self.pretrained_model is not None:
            print("Loading pretrained model")
            state_dict = torch.load(self.pretrained_model)["state_dict"]
            state_dict = {
                k.partition("model.")[2]: state_dict[k] for k in state_dict.keys()
            }
            self.model.load_state_dict(state_dict)

        all_files = build_file_paths(self.patients, self.bodypart)

        # --------- In case is needed to train on Verse----------#
        # uncomment the lines below and import build_file_paths_verse
        #
        # if self.bodypart == "spine":
        #    all_files = build_file_paths_verse(self.verse_dir)

        # randomly split train and validation set
        training_files, validation_files = random_split_files(
            all_files, self.train_val_ratio, self.separate_cine_train_val
        )

        print("Number of training files: " + str(len(training_files)))
        print("Number of validation files: " + str(len(validation_files)))

        print(training_files)
        print("Validating on: ")
        print(validation_files)

        if self.calc_ce_weights:
            self.ce_weights = calculate_ce_weights(self.root_dir, training_files)
        else:
            self.ce_weights = None

        # get the data as tio subjects
        subjects_train = BDataset(self.root_dir, self.bodypart, files=training_files,)
        subjects_validation = BDataset(
            self.root_dir, self.bodypart, files=validation_files
        )

        # --------- In case is needed to train on Verse----------#
        # uncomment the lines below and import VerseDataset
        #
        # if self.bodypart == 'spine':
        #    subjects_train_verse = VerseDataset(
        #    self.verse_dir,
        #    files=training_files_verse,
        #    )
        #    subjects_validation_verse = VerseDataset(
        #        self.verse_dir, files=validation_files_verse
        #    )

        self.train_ds = CacheDataset(
            data=subjects_train,
            transform=train_transforms(self.select_labels, self.roi_size, self.pix_dim),
            cache_num=len(training_files),
            cache_rate=1,
            num_workers=8,
        )

        self.val_ds = CacheDataset(
            data=subjects_validation,
            transform=val_transforms(self.select_labels, self.pix_dim),
            cache_num=len(validation_files),
            cache_rate=1,
            num_workers=8,
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=list_data_collate,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
        )

        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        lr_dict = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=30, gamma=0.5
            ),
            "interval": "epoch",
            "name": "BSP_learning_rate",
        }

        return [optimizer], [lr_dict]
