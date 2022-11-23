"""Class to build Whole Heart, Thorax and Spine datasets for pytorch dataloader."""
import os
from torch.utils.data import Dataset


class VerseDataset(Dataset):
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

    def __init__(self,verse_dir,files):
        """
        Parameters
        ----------
        root_dir: string
            root directory of the data
        files: list
            path/filenames of patients to load
        """
        self.verse_dir = verse_dir
        self.files = files

    def __getitem__(self, item):


        for i in os.listdir(os.path.join(self.verse_dir,'labels')):
            if self.files[item][:-10] in i:
                labelmap_path = i

                # create tio Subject
        return {
            "image": os.path.join(self.verse_dir, "images", self.files[item]),
            "label": os.path.join(self.verse_dir, "labels", labelmap_path),
        }


    def __len__(self):
        """
        Parameters
        ----------
        """
        return len(self.files)
