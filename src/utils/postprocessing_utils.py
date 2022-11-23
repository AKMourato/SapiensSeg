"""Class containing multiple postprocessing steps for WH segmentation"""
import torch
import numpy as np
import SimpleITK as sitk
from monai.transforms import KeepLargestConnectedComponent, GaussianSmooth

LABELS = [
    "Background",
    "Right Ventricle",
    "Right Atrium",
    "Pulmonary Artery",
    "Left Ventricle",
    "Left Atrium",
    "Aorta",
    "Myocardium",
]


class PostProcessing:
    """
    A class that contains methods
    for postprocessing the WH segmentation network output

    ...

    Attributes
    ----------
    transforms: list
        A list that contains all transforms that should be applied
    gaussian_sigma: int
        Sigma value for gaussian smoothing
    median_radius: int
        Radius of median filter
    verbose: Boolean
        If information should be printed to the console


    Methods
    -------
    __call__(network_prediction)
        Calls specified postprocessing transforms
    gaussian_smoothing(network_prediction)
        Gaussian labelmap smoothing (see Slicer3D)
    median_smoothing(network_prediction)
        Median labelmap smoothing (see Slicer3D)
    keep_largest_island(network_prediction)
        Keep largest island
    close_holes(network_prediction)
        Diameter closing operation
    myocardium_correction(network_prediction)
        Dilating ventricles trick
    """

    def __init__(
        self,
        transforms=None,
        gaussian_sigma=1,
        median_radius=5,
        verbose=False,
    ):
        """
        Parameters
        ----------
        transforms: list
            A list that contains all transforms that should be applied
        gaussian_sigma: int
            Sigma value for gaussian smoothing
        median_radius: int
            Radius of median filter
        verbose: Boolean
            Print out additional information
        """
        if transforms is None:
            self.transforms = [
                "keep_largest_island",
                "median_smoothing",
                "diameter_closing",
                "myocardium_correction",
            ]
        else:
            self.transforms = transforms

        self.gaussian_sigma = gaussian_sigma
        self.median_radius = median_radius
        self.verbose = verbose

    def __call__(self, network_prediction):
        """
        Parameters
        ----------
        network_prediction: tensor
            Tensor containing raw network output (after argmax)
        """
        for transform in self.transforms:

            if transform == "gaussian_smoothing":
                network_prediction = self.gaussian_smoothing(network_prediction)
            if transform == "keep_largest_island":
                network_prediction = self.keep_largest_island(network_prediction)
            if transform == "diameter_closing":
                network_prediction = self.close_holes(network_prediction)
            if transform == "myocardium_correction":
                network_prediction = self.myocardium_correction(network_prediction)
            if transform == "median_smoothing":
                network_prediction = self.median_smoothing(network_prediction)

        return network_prediction

    def gaussian_smoothing(self, network_prediction):
        """
        Parameters
        ----------
        network_prediction: tensor
            Tensor containing raw network output (after argmax)
        """
        # init monai filter
        gaussian = GaussianSmooth(sigma=2)
        # do smoothing for all segments
        for label in range(1, 7):
            if self.verbose:
                print("Smoothing segment: " + LABELS[label])
            # use temporary binary labelmap representation
            binary_labelmap = np.zeros(network_prediction.shape)
            binary_labelmap[network_prediction == label] = 1
            # apply filter
            binary_labelmap = gaussian(binary_labelmap)
            # clean up network_prediction and fill in smoothed labelmap
            network_prediction[network_prediction == label] = 0
            network_prediction[0][binary_labelmap > 0.5] = label

        return network_prediction

    def median_smoothing(self, network_prediction):
        """
        Parameters
        ----------
        network_prediction: tensor
            Tensor containing raw network output (after argmax)
        """
        # ignore batch dimension
        if len(network_prediction.shape) == 4:
            network_prediction = network_prediction[0]

        # init sitk filter
        median = sitk.MedianImageFilter()
        median.SetRadius(self.median_radius)

        for label in range(1, 8):
            if self.verbose:
                print("Median smoothing segment: " + LABELS[label])
            # use temporary binary labelmap representation
            binary_labelmap = np.zeros(network_prediction.shape).astype(np.int16)
            binary_labelmap[network_prediction == label] = 1
            # execute sitk filter
            binary_labelmap = sitk.GetImageFromArray(binary_labelmap)
            binary_labelmap = median.Execute(binary_labelmap)
            binary_labelmap = sitk.GetArrayFromImage(binary_labelmap)
            # clean up network_prediction and fill in smoothed labelmap
            network_prediction[network_prediction == label] = 0
            network_prediction[binary_labelmap > 0.5] = label

        # add batch dimension again for proper saving
        if len(network_prediction.shape) == 3:
            network_prediction = torch.unsqueeze(network_prediction, 0)

        return network_prediction

    def keep_largest_island(self, network_prediction):
        """
        Parameters
        ----------
        network_prediction: tensor
            Tensor containing raw network output (after argmax)
        """
        if self.verbose:
            print("Keep Largest Island")
        # init monai filter (exlude Aorta!)
        keep_largest_island = KeepLargestConnectedComponent(
            applied_labels=[1, 2, 3, 4, 5, 7]
        )
        # apply filter
        network_prediction = keep_largest_island(network_prediction)

        return network_prediction

    def close_holes(self, network_prediction):
        """
        Parameters
        ----------
        network_prediction: tensor
            Tensor containing raw network output (after argmax)
        """
        # ignore batch dimension
        if len(network_prediction.shape) == 4:
            network_prediction = network_prediction[0]

        # init sitk filter
        bmcif = sitk.BinaryMorphologicalClosingImageFilter()
        bmcif.SetForegroundValue(1)
        bmcif.SetKernelType(sitk.sitkBall)
        bmcif.SetKernelRadius(3)

        for label in range(1, 8):
            if self.verbose:
                print("Diameter closing segment: " + LABELS[label])
            # use temporary binary labelmap representation
            binary_labelmap = np.zeros(network_prediction.shape).astype(np.int16)
            binary_labelmap[network_prediction == label] = 1
            # execute sitk filter
            binary_labelmap = sitk.GetImageFromArray(binary_labelmap)
            binary_labelmap = bmcif.Execute(binary_labelmap)
            binary_labelmap = sitk.GetArrayFromImage(binary_labelmap)
            # clean up network_prediction and fill in smoothed labelmap
            network_prediction[network_prediction == label] = 0
            network_prediction[binary_labelmap > 0] = label

        # add batch dimension again for proper saving
        if len(network_prediction.shape) == 3:
            network_prediction = torch.unsqueeze(network_prediction, 0)

        return network_prediction

    def myocardium_correction(self, network_prediction):
        """
        Parameters
        ----------
        network_prediction: tensor
            Tensor containing raw network output (after argmax)
        """
        if self.verbose:
            print("Myocardium Correction")
        # ignore batch dimension
        if len(network_prediction.shape) == 4:
            network_prediction = network_prediction[0]

        # use temporary binary labelmap representation
        binary_labelmap = np.zeros(network_prediction.shape).astype(np.int16)
        binary_labelmap[(network_prediction == 1) | (network_prediction == 4)] = 1
        # set up filter
        dilate_filter = sitk.BinaryDilateImageFilter()
        dilate_filter.SetKernelRadius(2)
        dilate_filter.SetKernelType(sitk.sitkBall)
        dilate_filter.SetForegroundValue(1)

        binary_labelmap = sitk.GetImageFromArray(binary_labelmap)

        # execute and get back as array
        binary_labelmap = dilate_filter.Execute(binary_labelmap)
        binary_labelmap = sitk.GetArrayFromImage(binary_labelmap)

        # add myocardium, if dilated and before just background
        network_prediction[
            ((network_prediction == 0) & (binary_labelmap == 1)).bool()
        ] = 7

        # add channel dimension again, needed for smooth
        network_prediction = torch.unsqueeze(network_prediction, 0)

        binary_labelmap = np.zeros(network_prediction.shape)
        binary_labelmap[network_prediction == 7] = 1

        # init monai filter
        gaussian = GaussianSmooth(sigma=1)
        binary_labelmap = gaussian(binary_labelmap)

        network_prediction[0][binary_labelmap > 0.5] = 7

        return network_prediction
