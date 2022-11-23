"""Custom transform module to select specific sub-structures labelmaps."""
from monai.transforms.transform import Transform
import numpy as np

#pylint: disable=too-few-public-methods,duplicate-code
class SelectLabels(Transform):
    """
    Class to switch out certain
    labels.

    Attributes
    ----------
    keys: list
        Keys to apply tranform
    labels: list
        Labels to select
    inputs: dict
        Received from previous transform

    Methods
    -------
    __call__(input)
        Applies transform
        on specified keys
    """

    def __init__(self, keys, labels):
        """
        See class docstring
        """
        self.keys = keys
        self.labels = labels

    def __call__(self, inputs):
        """
        See class docstring
        """
        dic = dict(inputs)
        if self.labels:
            for key in self.keys:
                mask = np.isin(dic[key], np.array(self.labels), invert=True)
                dic[key][mask] = 0
                for i,value in enumerate(np.unique(dic[key])):
                    dic[key][dic[key]==value] = i
        return dic
