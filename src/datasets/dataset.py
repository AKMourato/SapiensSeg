"""Class to build Whole Heart, Thorax and Spine datasets for pytorch dataloader."""
import os
from torch.utils.data import Dataset


class BDataset(Dataset):
    """
    A class that is used to create
    a map-style dataset for dataloading

    ...

    Attributes
    ----------
    root_dir: string
        root directory of the data
    files: list
        path/filenames of patients to load
    bodypart: str
        part of the body

    Methods
    -------
    __getitem__(item)
        returns dict with image and label (paths)
    __len__
        returns number of patients == len(files)
    """

    def __init__(self, root_dir, bodypart, files):
        """
        Parameters
        ----------
        root_dir: string
            root directory of the data
        files: list
            path/filenames of patients to load
        """
        self.root_dir = root_dir
        self.files = files
        self.bodypart = bodypart

    # pylint: disable=inconsistent-return-statements
    def __getitem__(self, item):


        if self.bodypart == "heart":
            # read in the image and label
            # find label by renaming image file (adding -label)
            dot_index = self.files[item].index(".")
            if "-cropped" in self.files[item]:
                labelmap_path = (
                    self.files[item][: (dot_index - 8)] + "-heart-cropped-label.nii.gz"
                )
            else:
                labelmap_path = self.files[item][:dot_index] + "-heart-label.nii.gz"

            # create tio Subject
            return {
                "image": os.path.join(self.root_dir,self.files[item]),
                "label": os.path.join(self.root_dir,labelmap_path),
            }

        if self.bodypart in ["thorax","spine", "diaphragm"]:
            # read in the image and label
            # find label by renaming image file (adding -label)
            dot_index = self.files[item].index(".")
            if "-cropped" in self.files[item]:
                labelmap_path = (
                    self.files[item][: (dot_index - 8)] + "-thorax-cropped-label.nii.gz"
                )
            else:
                labelmap_path = self.files[item][:dot_index] + "-thorax-label.nii.gz"

            # create tio Subject
            return {
                "image": os.path.join(self.root_dir,self.files[item]),
                "label": os.path.join(self.root_dir,labelmap_path),
            }

    def __len__(self):
        """
        Parameters
        ----------
        """
        return len(self.files)
